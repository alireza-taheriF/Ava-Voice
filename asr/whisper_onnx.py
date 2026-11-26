"""Whisper ASR on ONNX Runtime — no PyTorch, no CTranslate2.

Runs the sherpa-onnx Whisper export (``*-encoder.onnx``, ``*-decoder.onnx``,
``*-tokens.txt``) with plain :mod:`onnxruntime` and a NumPy log-mel frontend,
so real Persian speech recognition works anywhere ``pip install onnxruntime``
does (Linux, macOS incl. Apple Silicon, Windows).

Used by Ava to *measure* its own voice (synthesize → recognize → character
error rate, see ``scripts/eval_intelligibility.py``) and by PVAP as its
speech recognizer.

Get a model (Whisper *turbo* is the best multilingual/Persian trade-off)::

    curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-turbo.tar.bz2
    tar xjf sherpa-onnx-whisper-turbo.tar.bz2

The feature pipeline and greedy decoding mirror sherpa-onnx's reference
``scripts/whisper/test.py``.
"""

from __future__ import annotations

import base64
import glob
import os
from functools import lru_cache

import numpy as np

__all__ = ["WhisperOnnxASR", "log_mel_spectrogram"]

_SR = 16_000
_N_FFT = 400
_HOP = 160


def _hz_to_mel(f: np.ndarray) -> np.ndarray:  # Slaney (librosa default)
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


@lru_cache(maxsize=4)
def _mel_filters(n_mels: int) -> np.ndarray:
    """librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels), Slaney-normalised."""
    fft_freqs = np.linspace(0, _SR / 2, 1 + _N_FFT // 2)
    mel_pts = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(_SR / 2), n_mels + 2))
    fdiff = np.diff(mel_pts)
    ramps = mel_pts[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0, np.minimum(lower, upper))
    w *= (2.0 / (mel_pts[2 : n_mels + 2] - mel_pts[:n_mels]))[:, None]
    return w.astype(np.float32)


def log_mel_spectrogram(audio: np.ndarray, n_mels: int = 128) -> np.ndarray:
    """Whisper log-mel features, shape ``(n_mels, frames)``; float audio at 16 kHz."""
    x = np.asarray(audio, dtype=np.float32)
    x = np.pad(x, (_N_FFT // 2, _N_FFT // 2), mode="reflect")
    window = np.hanning(_N_FFT + 1)[:-1].astype(np.float32)  # periodic Hann
    n_frames = 1 + (len(x) - _N_FFT) // _HOP
    idx = np.arange(_N_FFT)[None, :] + _HOP * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * window, axis=1)) ** 2  # (frames, 201)
    spec = spec[:-1]  # Whisper drops the last frame
    mel = spec @ _mel_filters(n_mels).T  # (frames, n_mels)
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).T.astype(np.float32)


class WhisperOnnxASR:
    def __init__(self, model_dir: str, language: str = "fa", threads: int | None = None, max_tokens: int = 200,
                 pad_frames: int = 600) -> None:
        import onnxruntime as ort

        def pick(pattern: str) -> str:
            hits = sorted(glob.glob(os.path.join(model_dir, pattern)))
            int8 = [h for h in hits if ".int8." in h]
            if not (int8 or hits):
                raise FileNotFoundError(f"{pattern} not found in {model_dir}")
            return (int8 or hits)[0]

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        # Never oversubscribe: on a 2-core box 4 threads made decoding 1.6x slower.
        opts.intra_op_num_threads = threads or max(1, os.cpu_count() or 1)
        self._enc = ort.InferenceSession(pick("*encoder*.onnx"), opts, providers=["CPUExecutionProvider"])
        self._dec = ort.InferenceSession(pick("*decoder*.onnx"), opts, providers=["CPUExecutionProvider"])
        meta = self._enc.get_modelmeta().custom_metadata_map
        self.name = f"whisper-onnx:{meta.get('model_type', os.path.basename(model_dir))}"
        self._n_mels = int(meta["n_mels"])
        self._n_layer = int(meta["n_text_layer"])
        self._n_ctx = int(meta["n_text_ctx"])
        self._n_state = int(meta["n_text_state"])
        self._eot = int(meta["eot"])
        self._suppress_always = [int(meta[k]) for k in ("no_timestamps", "sot", "no_speech", "translate")]
        self._blank = int(meta["blank_id"])
        langs = dict(zip(meta["all_language_codes"].split(","), map(int, meta["all_language_tokens"].split(","))))
        sot_seq = list(map(int, meta["sot_sequence"].split(",")))
        if language not in langs:
            raise ValueError(f"unsupported language {language!r}")
        sot_seq[1] = langs[language]
        self._sot = sot_seq + [int(meta["no_timestamps"])]
        self._max_tokens = max_tokens
        # sherpa-onnx pads 1500 frames (15 s) of silence. Fewer frames make the
        # encoder much cheaper; measured on 20 Persian utterances, 2 cores:
        # 1500 → 8.4 % CER / 3.0 s, 500 → 9.4 % / 1.9 s, 300 → 19.9 % (breaks).
        self._pad = pad_frames
        self._vocab: dict[int, bytes] = {}
        with open(pick("*tokens.txt"), encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2:
                    self._vocab[int(parts[1])] = base64.b64decode(parts[0])

    def _features(self, audio16k: np.ndarray) -> np.ndarray:
        mel = log_mel_spectrogram(audio16k, self._n_mels).T  # (frames, n_mels)
        mel = np.pad(mel, ((0, self._pad), (0, 0)))
        if mel.shape[0] > 3000:
            mel = np.pad(mel[: 3000 - 50], ((0, 50), (0, 0)))
        return mel.T[None].astype(np.float32)

    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str:
        """Transcribe int16 or float audio. Non-16 kHz input is resampled linearly;
        pass 16 kHz (or resample with a proper filter) for best accuracy."""
        x = np.asarray(pcm)
        audio = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
        if sample_rate != _SR:
            n = int(round(len(audio) * _SR / sample_rate))
            audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
        return self.transcribe_16k(audio)

    def transcribe_16k(self, audio: np.ndarray) -> str:
        """Transcribe float32 mono audio at 16 kHz."""
        audio = np.asarray(audio, np.float32)
        if len(audio) < _SR // 10:
            return ""
        cross_k, cross_v = self._enc.run(None, {self._enc.get_inputs()[0].name: self._features(audio)})
        k_cache = np.zeros((self._n_layer, 1, self._n_ctx, self._n_state), np.float32)
        v_cache = np.zeros_like(k_cache)
        names = [i.name for i in self._dec.get_inputs()]
        tokens = np.array([self._sot], dtype=np.int64)
        offset = np.zeros(1, dtype=np.int64)
        out: list[int] = []
        first = True
        for _ in range(self._max_tokens):
            logits, k_cache, v_cache = self._dec.run(
                None,
                dict(zip(names, (tokens, k_cache, v_cache, cross_k, cross_v, offset))),
            )[:3]
            offset = offset + tokens.shape[1]
            step = logits[0, -1].astype(np.float32)
            step[self._suppress_always] = -np.inf
            if first:
                step[[self._eot, self._blank]] = -np.inf
                first = False
            nxt = int(step.argmax())
            if nxt == self._eot:
                break
            out.append(nxt)
            tokens = np.array([[nxt]], dtype=np.int64)
        return b"".join(self._vocab.get(t, b"") for t in out).decode("utf-8", "replace").strip()
