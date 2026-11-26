"""Homo-GE2PE inference in pure NumPy — no PyTorch, no transformers.

Homo-GE2PE is a 4-layer byte-level T5 (2 encoder + 2 decoder layers,
d_model 512, ~8 M parameters). That is small enough that a straight NumPy
forward pass runs in milliseconds per sentence on a CPU. This module reads the
checkpoint's ``model.safetensors`` directly and re-implements exactly the T5
operations the checkpoint uses:

* RMS layer norm (no bias, no mean subtraction),
* unscaled dot-product attention with T5 bucketed relative position bias
  (bidirectional in the encoder, causal in the decoder; computed by the first
  layer and shared by the rest),
* gated-GELU feed-forward (``gelu_new`` tanh approximation),
* untied LM head, byte-level vocabulary (ByT5: ``id = byte + 3``).

Weights: ``model-weights/homo-ge2pe.zip`` from
https://github.com/MahtaFetrat/Homo-GE2PE-Persian (MIT). Unzip it and point
:data:`~ava_voice.linguistics.neural_phonemizer.MODEL_PATH_ENV` at the folder.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["NumpyT5", "load_safetensors", "byt5_encode", "byt5_decode"]

_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": None, "I64": np.int64}


def load_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Read a ``.safetensors`` file into float32 NumPy arrays (memory-mapped)."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    (n,) = struct.unpack("<Q", bytes(raw[:8]))
    header = json.loads(bytes(raw[8 : 8 + n]))
    base = 8 + n
    out: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        buf = raw[base + start : base + end]
        if info["dtype"] == "BF16":
            arr = (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            arr = np.frombuffer(buf, dtype=_DTYPES[info["dtype"]])
        out[name] = arr.reshape(info["shape"]).astype(np.float32)
    return out


def _rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def _gelu_new(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _bucket(rel: np.ndarray, bidirectional: bool, num_buckets: int, max_distance: int) -> np.ndarray:
    """T5 relative-position bucketing (transformers' ``_relative_position_bucket``)."""
    buckets = np.zeros_like(rel)
    if bidirectional:
        num_buckets //= 2
        buckets += (rel > 0).astype(rel.dtype) * num_buckets
        rel = np.abs(rel)
    else:
        rel = -np.minimum(rel, 0)
    max_exact = num_buckets // 2
    is_small = rel < max_exact
    with np.errstate(divide="ignore"):
        large = max_exact + (
            np.log(np.maximum(rel, 1) / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).astype(rel.dtype)
    large = np.minimum(large, num_buckets - 1)
    return buckets + np.where(is_small, rel, large)


class NumpyT5:
    """Encoder–decoder T5 forward pass for a ``T5ForConditionalGeneration`` checkpoint."""

    def __init__(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())
        self.cfg = cfg
        self.w = load_safetensors(model_dir / "model.safetensors")
        self.h = cfg["num_heads"]
        self.dk = cfg["d_kv"]
        self.eps = cfg["layer_norm_epsilon"]
        self.nb = cfg["relative_attention_num_buckets"]
        self.md = cfg["relative_attention_max_distance"]
        self.n_enc = cfg["num_layers"]
        self.n_dec = cfg.get("num_decoder_layers", self.n_enc)
        self.start = cfg.get("decoder_start_token_id", 0)
        self.eos = cfg.get("eos_token_id", 1)
        self.emb = self.w["shared.weight"]
        self.lm_head = self.w.get("lm_head.weight", self.emb)
        if cfg.get("tie_word_embeddings", True):
            self.lm_scale = cfg["d_model"] ** -0.5
        else:
            self.lm_scale = 1.0

    # ---------------------------------------------------------------- blocks
    def _attn(self, prefix: str, xq: np.ndarray, xkv: np.ndarray, bias: np.ndarray | None,
              mask: np.ndarray | None) -> np.ndarray:
        w = self.w
        q = xq @ w[f"{prefix}.q.weight"].T
        k = xkv @ w[f"{prefix}.k.weight"].T
        v = xkv @ w[f"{prefix}.v.weight"].T
        def split(t: np.ndarray) -> np.ndarray:  # (T, H*dk) -> (H, T, dk)
            return t.reshape(t.shape[0], self.h, self.dk).transpose(1, 0, 2)
        q, k, v = split(q), split(k), split(v)
        scores = q @ k.transpose(0, 2, 1)  # T5: no 1/sqrt(dk)
        if bias is not None:
            scores = scores + bias
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        ctx = _softmax(scores) @ v  # (H, Tq, dk)
        ctx = ctx.transpose(1, 0, 2).reshape(xq.shape[0], self.h * self.dk)
        return ctx @ w[f"{prefix}.o.weight"].T

    def _ff(self, prefix: str, x: np.ndarray) -> np.ndarray:
        w = self.w
        g = _gelu_new(x @ w[f"{prefix}.wi_0.weight"].T) * (x @ w[f"{prefix}.wi_1.weight"].T)
        return g @ w[f"{prefix}.wo.weight"].T

    def _bias(self, table: str, q_len: int, k_len: int, bidirectional: bool, q_offset: int = 0) -> np.ndarray:
        ctx = np.arange(q_offset, q_offset + q_len)[:, None]
        mem = np.arange(k_len)[None, :]
        b = _bucket(mem - ctx, bidirectional, self.nb, self.md)
        return self.w[table][b].transpose(2, 0, 1)  # (H, q, k)

    # ------------------------------------------------------------- encoder
    def encode(self, ids: Sequence[int]) -> np.ndarray:
        x = self.emb[np.asarray(ids)]
        bias = self._bias("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                          len(ids), len(ids), True)
        for i in range(self.n_enc):
            p = f"encoder.block.{i}.layer"
            x = x + self._attn(f"{p}.0.SelfAttention", _rms(x, self.w[f"{p}.0.layer_norm.weight"], self.eps),
                               _rms(x, self.w[f"{p}.0.layer_norm.weight"], self.eps), bias, None)
            x = x + self._ff(f"{p}.1.DenseReluDense", _rms(x, self.w[f"{p}.1.layer_norm.weight"], self.eps))
        return _rms(x, self.w["encoder.final_layer_norm.weight"], self.eps)

    # ------------------------------------------------------------- decoder
    def decode_logits(self, enc: np.ndarray, prefix_ids: Sequence[int]) -> np.ndarray:
        """Log-probabilities for the token after ``prefix_ids`` (full recompute; fine at this size)."""
        x = self.emb[np.asarray(prefix_ids)]
        t = len(prefix_ids)
        bias = self._bias("decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", t, t, False)
        causal = np.tril(np.ones((t, t), dtype=bool))[None]
        for i in range(self.n_dec):
            p = f"decoder.block.{i}.layer"
            h = _rms(x, self.w[f"{p}.0.layer_norm.weight"], self.eps)
            x = x + self._attn(f"{p}.0.SelfAttention", h, h, bias, causal)
            h = _rms(x, self.w[f"{p}.1.layer_norm.weight"], self.eps)
            x = x + self._attn(f"{p}.1.EncDecAttention", h, enc, None, None)
            x = x + self._ff(f"{p}.2.DenseReluDense", _rms(x, self.w[f"{p}.2.layer_norm.weight"], self.eps))
        x = _rms(x[-1], self.w["decoder.final_layer_norm.weight"], self.eps) * self.lm_scale
        logits = x @ self.lm_head.T
        logits = logits - logits.max()
        return logits - np.log(np.exp(logits).sum())

    def _step(self, enc_kv: list[tuple[np.ndarray, np.ndarray]], tokens: np.ndarray, pos: int,
              cache: list[list[np.ndarray]]) -> np.ndarray:
        """One incremental decoder step for a batch of beams.

        ``tokens``: (B,) last token per beam; ``cache[i] = [K, V]`` with shape
        (B, H, pos, dk) holding self-attention keys/values of earlier steps.
        Returns log-probabilities (B, vocab).
        """
        w, h, dk = self.w, self.h, self.dk
        x = self.emb[tokens]  # (B, D)
        B = x.shape[0]
        bias = self._bias("decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                          1, pos + 1, False, q_offset=pos)  # (H, 1, pos+1)
        for i in range(self.n_dec):
            p = f"decoder.block.{i}.layer"
            hdn = _rms(x, w[f"{p}.0.layer_norm.weight"], self.eps)
            a = f"{p}.0.SelfAttention"
            q = (hdn @ w[f"{a}.q.weight"].T).reshape(B, h, 1, dk)
            k = (hdn @ w[f"{a}.k.weight"].T).reshape(B, h, 1, dk)
            v = (hdn @ w[f"{a}.v.weight"].T).reshape(B, h, 1, dk)
            K = np.concatenate([cache[i][0], k], axis=2)
            V = np.concatenate([cache[i][1], v], axis=2)
            cache[i] = [K, V]
            att = _softmax(q @ K.transpose(0, 1, 3, 2) + bias[None]) @ V  # (B,H,1,dk)
            x = x + att.reshape(B, h * dk) @ w[f"{a}.o.weight"].T
            hdn = _rms(x, w[f"{p}.1.layer_norm.weight"], self.eps)
            c = f"{p}.1.EncDecAttention"
            ek, ev = enc_kv[i]  # (H, S, dk)
            q = (hdn @ w[f"{c}.q.weight"].T).reshape(B, h, 1, dk)
            att = _softmax(q @ ek.transpose(0, 2, 1)[None]) @ ev[None]
            x = x + att.reshape(B, h * dk) @ w[f"{c}.o.weight"].T
            x = x + self._ff(f"{p}.2.DenseReluDense", _rms(x, w[f"{p}.2.layer_norm.weight"], self.eps))
        x = _rms(x, w["decoder.final_layer_norm.weight"], self.eps) * self.lm_scale
        logits = x @ self.lm_head.T
        logits = logits - logits.max(axis=-1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))

    def generate(self, ids: Sequence[int], num_beams: int = 5, max_length: int = 512,
                 length_penalty: float = 1.0) -> list[int]:
        """Beam search equivalent to ``model.generate(num_beams, early_stopping=True)``,
        with incremental (KV-cached) decoding."""
        enc = self.encode(ids)
        w, h, dk = self.w, self.h, self.dk
        enc_kv = []
        for i in range(self.n_dec):
            c = f"decoder.block.{i}.layer.1.EncDecAttention"
            ek = (enc @ w[f"{c}.k.weight"].T).reshape(-1, h, dk).transpose(1, 0, 2)
            ev = (enc @ w[f"{c}.v.weight"].T).reshape(-1, h, dk).transpose(1, 0, 2)
            enc_kv.append((ek, ev))
        empty = np.zeros((1, h, 0, dk), np.float32)
        cache = [[empty, empty] for _ in range(self.n_dec)]
        seqs: list[list[int]] = [[]]
        scores = np.zeros(1)
        last = np.array([self.start])
        done: list[tuple[float, list[int]]] = []
        for pos in range(max_length):
            lp = self._step(enc_kv, last, pos, cache)  # (B, V)
            total = scores[:, None] + lp
            flat = total.ravel()
            k = min(len(flat), 2 * num_beams)
            top = np.argpartition(-flat, k - 1)[:k]
            top = top[np.argsort(-flat[top])]
            new_seqs, new_scores, src, toks = [], [], [], []
            for idx in top:
                b, tok = divmod(int(idx), lp.shape[1])
                if tok == self.eos:
                    L = len(seqs[b]) + 1
                    done.append((float(flat[idx]) / (L ** length_penalty), seqs[b]))
                    continue
                new_seqs.append(seqs[b] + [tok])
                new_scores.append(float(flat[idx]))
                src.append(b)
                toks.append(tok)
                if len(new_seqs) == num_beams:
                    break
            if len(done) >= num_beams or not new_seqs:
                break
            src_a = np.array(src)
            cache = [[K[src_a], V[src_a]] for K, V in cache]
            seqs, scores, last = new_seqs, np.array(new_scores), np.array(toks)
        if not done:
            done = [(s / (len(q) ** length_penalty), q) for s, q in zip(scores, seqs)]
        return max(done, key=lambda d: d[0])[1]


def byt5_encode(text: str) -> list[int]:
    return [b + 3 for b in text.encode("utf-8")]


def byt5_decode(ids: Sequence[int]) -> str:
    return bytes(i - 3 for i in ids if 3 <= i < 259).decode("utf-8", "replace")
