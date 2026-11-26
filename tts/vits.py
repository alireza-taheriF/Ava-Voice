"""Neural Persian TTS: Ava text frontend → Homo-GE2PE → VITS (ONNX).

This is Ava's production voice path. Text goes through:

1. :func:`~ava_voice.linguistics.persian_normalizer.normalize` — character
   unification, ZWNJ, and number/date/time verbalization;
2. clause splitting on punctuation (kept as prosodic breaks);
3. **Homo-GE2PE** G2P in pure NumPy — restores the unwritten short vowels,
   ezafe and homograph readings (``مرد`` → *mard* / *mord* from context);
4. a mapping from GE2PE's ASCII scheme to the espeak-ng Persian IPA inventory
   the Piper/VITS Persian voices were trained on, with word-final stress;
5. a VITS acoustic model + vocoder in ``onnxruntime``.

Why not espeak-ng? The voices were trained on espeak phonemes, but espeak's
Persian G2P does not know unwritten short vowels. Feeding the voices
GE2PE-derived phonemes in the same notation gives them correct input. The
mapping was chosen by measuring intelligibility: synthesize → Whisper →
character error rate (``scripts/eval_intelligibility.py``).

Emotion-aware prosody: :meth:`PersianVITS.synthesize_plan` applies each
sentence's ``speaking_rate`` from Ava's prosody engine as VITS ``length_scale``.
"""

from __future__ import annotations

import json
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np

from ava_voice.linguistics.persian_normalizer import normalize

__all__ = ["PersianVITS", "ge2pe_word_to_ipa", "GE2PE_TO_IPA", "fix_ge2pe", "is_mi_verb"]

#: Homo-GE2PE symbol → espeak-ng Persian IPA (Piper voices' training alphabet).
GE2PE_TO_IPA: Final[dict[str, str]] = {
    "a": "ɑː", "/": "a", "e": "e", "o": "o", "i": "i", "u": "u",
    "@": "ʔ", "$": "ʃ", "c": "tʃ", ";": "ʒ", "j": "dʒ", "q": "q",
    "x": "x", "y": "j", "g": "ɡ", "r": "r", "h": "h",
}
_VOWELS: Final[frozenset[str]] = frozenset("a/eoiu")
_DROP: Final[frozenset[str]] = frozenset({"1", "-", "_", "‌"})
_BREAKS: Final[dict[str, str]] = {"،": ",", ",": ",", "؛": ",", ";": ",", ":": ",",
                                  ".": ".", "!": "!", "…": ".", "?": "?", "؟": "?"}
_CLAUSE_RE: Final = re.compile(r"([،,؛;:.!…?؟])")
_PERSIAN_RE: Final = re.compile(r"[ء-يپچژکگیآ‌ ]+")


#: Monosyllabic clitics and function words that carry no lexical stress.
_UNSTRESSED: Final[frozenset[str]] = frozenset({
    "be", "ba", "b/", "d/r", "@/z", "ra", "ta", "v/", "@o", "o", "ke", "@/st", "h/m", "ya", "y/", "na",
    "@/gr", "c/", "ce", "ham", "@i", "ye", "yek",
})


def _stress_cut(w: str) -> int | None:
    vowels = [i for i, ch in enumerate(w) if ch in _VOWELS]
    if not vowels:
        return None
    v = vowels[-1]
    return v - 1 if v > 0 and w[v - 1] not in _VOWELS else v


def _initial_cut(w: str) -> int | None:
    vowels = [i for i, ch in enumerate(w) if ch in _VOWELS]
    if not vowels:
        return None
    v = vowels[0]
    return v - 1 if v > 0 and w[v - 1] not in _VOWELS else v


_DIGRAPHS: Final[tuple[tuple[str, str], ...]] = (("sh", "$"), ("kh", "x"), ("zh", ";"), ("ch", "c"), ("ei", "ey"))


def fix_ge2pe(word: str) -> str:
    """Repair ASCII digraphs the model sometimes emits on colloquial input
    (``mishe`` → ``mi$e``, ``mikham`` → ``mixam``, ``xeili`` → ``xeyli``)."""
    for a, b in _DIGRAPHS:
        word = word.replace(a, b)
    return word


#: Colloquial and formal verb prefixes that take the word stress (Persian
#: verbs stress mi-, the negative ne-/na- and subjunctive be-, not the ending).
_MI_PREFIX_RE: Final = re.compile(r"^(?:نمی|می)‌?\S{2,}$")
_NOT_VERBS: Final[frozenset[str]] = frozenset({
    "میوه", "میله", "میمون", "میهن", "میزان", "میدان", "میدون", "میلیون", "میلیارد", "میراث", "میانه",
    "میز", "میل", "میخ", "میش", "میکروب", "میکروفون", "مینا", "میترا", "میلاد",
})


def is_mi_verb(src: str) -> bool:
    return bool(_MI_PREFIX_RE.match(src)) and src.replace("‌", "") not in _NOT_VERBS


def ge2pe_word_to_ipa(word: str, stress: bool | str = True) -> str:
    """Convert one GE2PE word (e.g. ``s/lam``) to IPA (``saˈlɑːm``).

    Stress goes on the last syllable of the *stem*: an ezafe suffix (marked
    ``1`` by the model) stays unstressed (``nobate1`` → ``noˈbate``), and
    clitics/prepositions (``be``, ``ba``, ``ra``, ``@/st`` …) get none.
    ``stress="initial"`` stresses the first syllable (verbs with mi-/nemi-),
    and an explicit ``ˈ`` in ``word`` is kept as written.
    """
    ezafe = word.endswith("1")
    if "ˈ" in word:
        w = "".join(ch for ch in word if ch not in _DROP)
        return "".join(GE2PE_TO_IPA.get(ch, ch) for ch in w)
    w = "".join(ch for ch in word if ch not in _DROP)
    if not w:
        return ""
    cut = None
    if stress == "initial":
        cut = _initial_cut(w)
    elif stress and w not in _UNSTRESSED:
        stem = w
        if ezafe:
            if stem.endswith("ye") and len(stem) > 2 and stem[-3] in _VOWELS:
                stem = stem[:-2]
            elif stem.endswith("e"):
                stem = stem[:-1]
        cut = _stress_cut(stem)
    out = []
    for i, ch in enumerate(w):
        if i == cut:
            out.append("ˈ")
        out.append(GE2PE_TO_IPA.get(ch, ch))
    return "".join(out)


class PersianVITS:
    """Persian text → float32 waveform with a Piper-format VITS voice."""

    #: Lower than Piper's defaults (0.667 / 0.8): on a 20-sentence Whisper
    #: round-trip this cut character error from ~8 % to ~5.5 % with no audible
    #: flattening. Pass ``None`` to use the voice's own values.
    NOISE_SCALE: Final[float] = 0.35
    NOISE_W: Final[float] = 0.55

    def __init__(self, voice_dir: str | Path | None = None, g2p_dir: str | Path | None = None,
                 threads: int = 2, noise_scale: float | None = NOISE_SCALE,
                 noise_w: float | None = NOISE_W, lexicon: object | None = "default") -> None:
        import onnxruntime as ort

        from ava_voice.linguistics.ge2pe_numpy import NumpyT5
        from ava_voice.tts import models

        vdir = Path(voice_dir) if voice_dir else models.voice_dir()
        gdir = Path(g2p_dir) if g2p_dir else models.g2p_dir()
        onnx = sorted(p for p in vdir.glob("*.onnx"))
        if not onnx:
            raise FileNotFoundError(f"no .onnx voice in {vdir} — run: python -m ava_voice.tts.models fetch")
        if not (gdir / "model.safetensors").exists():
            raise FileNotFoundError(f"no G2P model in {gdir} — run: python -m ava_voice.tts.models fetch")
        cfg = json.loads(Path(str(onnx[0]) + ".json").read_text(encoding="utf-8"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(str(onnx[0]), opts, providers=["CPUExecutionProvider"])
        self.name = onnx[0].stem
        self.sample_rate = int(cfg["audio"]["sample_rate"])
        inf = cfg.get("inference", {})
        self._noise = float(noise_scale if noise_scale is not None else inf.get("noise_scale", 0.667))
        self._length = float(inf.get("length_scale", 1.0))
        self._noise_w = float(noise_w if noise_w is not None else inf.get("noise_w", 0.8))
        self._ids: dict[str, int] = {}
        if (vdir / "tokens.txt").exists():  # sherpa-onnx layout
            for line in (vdir / "tokens.txt").read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                tok, _, idx = line.rpartition(" ")
                self._ids[tok if tok else " "] = int(idx)
        else:  # plain Piper layout (e.g. a voice you trained): map lives in the JSON config
            self._ids = {k: (v[0] if isinstance(v, list) else int(v)) for k, v in cfg["phoneme_id_map"].items()}
        self._g2p = NumpyT5(gdir)
        if lexicon == "default":
            from ava_voice.linguistics.lexicon import default_lexicon

            lexicon = default_lexicon()
        #: Pronunciation overrides (builtin colloquial + user + learned); ``None`` disables.
        self.lexicon = lexicon
        self._lock = threading.Lock()
        self._cache = lru_cache(maxsize=4096)(self._g2p_clause)

    # ------------------------------------------------------------ frontend
    def _g2p_clause(self, clause: str, overrides: dict[str, str] | None = None) -> str:
        from ava_voice.linguistics.ge2pe_numpy import byt5_decode, byt5_encode

        with self._lock:
            raw = byt5_decode(self._g2p.generate(byt5_encode(clause))).split()
        src = clause.split()
        aligned = len(raw) == len(src)
        words: list[tuple[str, bool | str]] = []
        for i, w in enumerate(raw):
            w = fix_ge2pe("".join(ch for ch in w if ch not in _DROP or ch == "1"))
            mode: bool | str = True
            if aligned:
                lex = (overrides or {}).get(src[i])
                if lex is None and self.lexicon is not None:
                    lex = self.lexicon.lookup(src[i])
                if lex:
                    ez = w.endswith("1") and not lex.endswith("1")
                    parts = lex.split()
                    for p in parts[:-1]:
                        words.append((p, True))
                    w = parts[-1] + (("ye" if parts[-1][-1] in _VOWELS else "e") + "1" if ez else "")
                if "ˈ" not in w and is_mi_verb(src[i]):
                    mode = "initial"
            # The conjunction "و" read as /o/ cliticises onto the previous word
            # (هزار و چهارصد → hezɑːro tʃahɑːrsad), with a /v/ glide after a vowel.
            if w in ("@o", "o") and words:
                prev, pmode = words[-1]
                prev = prev.rstrip("1")
                words[-1] = (prev + ("vo" if prev[-1] in _VOWELS else "o"), pmode)
                continue
            words.append((w, mode))
        return " ".join(filter(None, (ge2pe_word_to_ipa(w, m) for w, m in words)))

    def clear_cache(self) -> None:
        """Forget cached clause phonemizations (call after the lexicon changes)."""
        self._cache.cache_clear()

    def word_candidates(self, word: str, n: int = 5) -> list[str]:
        """Alternative pronunciations (GE2PE n-best for the word on its own)."""
        from ava_voice.linguistics.ge2pe_numpy import byt5_decode, byt5_encode

        with self._lock:
            beams = self._g2p.generate(byt5_encode(word), num_beams=max(n, 2), return_all=True)
        out: list[str] = []
        for b in beams:
            p = fix_ge2pe(byt5_decode(b).strip().replace("1", ""))
            if p and " " not in p and p not in out:
                out.append(p)
        return out[:n]

    def phonemize(self, text: str, overrides: dict[str, str] | None = None) -> str:
        """Persian text → IPA string with stress and punctuation breaks.

        ``overrides`` maps written words to GE2PE pronunciations for this call
        only (used to try candidate pronunciations); it bypasses the cache.
        """
        text = normalize(text)
        out = ""
        for part in _CLAUSE_RE.split(text):
            if not part.strip():
                continue
            if part in _BREAKS:
                out = out.rstrip()
                if out and out[-1] not in ",.!?":
                    out += _BREAKS[part]
                out += " "
                continue
            clause = " ".join(m.group(0).strip() for m in _PERSIAN_RE.finditer(part)).strip()
            clause = re.sub(r"\s+", " ", clause)
            if clause:
                out += (self._g2p_clause(clause, overrides) if overrides else self._cache(clause)) + " "
        return out.strip()

    def _ids_for(self, ipa: str) -> np.ndarray:
        pad, bos, eos = self._ids["_"], self._ids["^"], self._ids["$"]
        seq = [bos, pad]
        for ch in ipa:
            i = self._ids.get(ch)
            if i is not None:
                seq += [i, pad]
        seq.append(eos)
        return np.array([seq], dtype=np.int64)

    # ------------------------------------------------------------ synthesis
    def synthesize_ipa(self, ipa: str, length_scale: float | None = None, noise_scale: float | None = None,
                       noise_w: float | None = None) -> np.ndarray:
        if not ipa.strip():
            return np.zeros(0, np.float32)
        ids = self._ids_for(ipa)
        scales = np.array([
            self._noise if noise_scale is None else noise_scale,
            length_scale or self._length,
            self._noise_w if noise_w is None else noise_w,
        ], np.float32)
        out = self._sess.run(None, {"input": ids, "input_lengths": np.array([ids.shape[1]], np.int64),
                                    "scales": scales})[0]
        return np.clip(out.reshape(-1), -1.0, 1.0).astype(np.float32)

    def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Text → float32 samples in ``[-1, 1]`` at :attr:`sample_rate`."""
        return self.synthesize_ipa(self.phonemize(text), self._length / max(0.5, min(2.0, speed)))

    def synthesize_plan(self, plan: object, sentence_gap_ms: float = 150.0) -> np.ndarray:
        """Render an Ava :class:`~ava_voice.core.speech_plan.SpeechPlan`.

        Each sentence is spoken at its prosody ``speaking_rate`` (clamped to
        ±20 % — VITS degrades beyond that), so emotion changes tempo.
        """
        gap = np.zeros(int(self.sample_rate * sentence_gap_ms / 1000), np.float32)
        chunks: list[np.ndarray] = []
        for i, s in enumerate(getattr(plan, "sentences")):
            rate = float(getattr(s.prosody.profile, "speaking_rate", 1.0))
            if i:
                chunks.append(gap)
            chunks.append(self.synthesize(s.text, speed=max(0.8, min(1.2, rate))))
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def _main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m ava_voice.tts.vits "متن فارسی" out.wav [--voice-dir DIR]``."""
    import argparse
    import wave

    ap = argparse.ArgumentParser(description="Persian text → WAV with Ava's neural voice")
    ap.add_argument("text")
    ap.add_argument("out")
    ap.add_argument("--voice-dir")
    ap.add_argument("--g2p-dir")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--show-ipa", action="store_true")
    args = ap.parse_args(argv)
    tts = PersianVITS(args.voice_dir, args.g2p_dir)
    if args.show_ipa:
        print(tts.phonemize(args.text))
    wav = tts.synthesize(args.text, speed=args.speed)
    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(tts.sample_rate)
        w.writeframes((wav * 32767).astype("<i2").tobytes())
    print(f"wrote {args.out}: {len(wav) / tts.sample_rate:.2f} s @ {tts.sample_rate} Hz ({tts.name})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
