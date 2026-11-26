"""Ava listens to herself and fixes her own pronunciation.

:class:`PronunciationLearner` closes the loop between speaking and hearing:

1. **audit** — synthesize a sentence, recognize it with an ASR model
   (Whisper), and align the recognized words with the written ones. Written
   words the recognizer did not hear are *suspects*.
2. **repair** — for each suspect, try alternative pronunciations: the G2P
   model's n-best readings of the word, plus vowel-swap variants of the current
   reading. Each candidate is spoken *inside the original sentence* and scored
   by recognition error, averaged over trials because VITS is stochastic.
3. **learn** — a candidate is kept only if it beats the current pronunciation
   by a clear margin, and it is written to the learned lexicon
   (:mod:`ava_voice.linguistics.lexicon`). Every later sentence uses it.

Run it on what a conversational agent actually said, for example after each
PVAP call. Ava's pronunciation then improves on exactly the words that come up
in real conversations. ``python -m ava_voice.tts.self_check --whisper-dir ...
sentences.txt`` does the same from the command line.

Limits: the recognizer is the judge. Recognizer spelling variants (formal
``خانه`` for colloquial ``خونه``) are normalized away, and candidates that do
not *improve* recognition are never learned. Still, review the learned file
(``~/.config/ava-voice/learned_lexicon.tsv``) now and then.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

__all__ = ["PronunciationLearner", "Suspect", "LearnReport", "normalize_for_compare"]

_COLLOQUIAL_EQUIV = {
    "خونه": "خانه", "اون": "آن", "این": "این", "میشه": "میشود", "نمیشه": "نمیشود",
    "دیگه": "دیگر", "یه": "یک", "تموم": "تمام",
}


def normalize_for_compare(text: str) -> list[str]:
    """Words for alignment: letters unified, ZWNJ/diacritics/punctuation removed."""
    t = unicodedata.normalize("NFC", text)
    for a, b in (("ي", "ی"), ("ك", "ک"), ("ۀ", "ه"), ("أ", "ا"), ("إ", "ا"), ("ئ", "ی"), ("ؤ", "و")):
        t = t.replace(a, b)
    t = t.replace("‌", "")
    t = re.sub(r"[ً-ٰٟ]", "", t)
    t = re.sub(r"[^ء-ۿ\s]", " ", t)
    return [_COLLOQUIAL_EQUIV.get(w, w) for w in t.split()]


def _wer(ref: list[str], hyp: list[str]) -> float:
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return 1.0 - matched / max(1, len(ref))


def _cer(ref: list[str], hyp: list[str]) -> float:
    r, h = "".join(ref), "".join(hyp)
    sm = difflib.SequenceMatcher(a=r, b=h, autojunk=False)
    return 1.0 - sum(b.size for b in sm.get_matching_blocks()) / max(1, len(r))


@dataclass
class Suspect:
    word: str
    sentence: str
    heard: str


@dataclass
class LearnReport:
    audited: int = 0
    suspects: list[Suspect] = field(default_factory=list)
    learned: dict[str, tuple[str, str, float, float]] = field(default_factory=dict)  # word → (old, new, err_old, err_new)

    def summary(self) -> str:
        lines = [f"audited {self.audited} sentences, {len(self.suspects)} suspect words, learned {len(self.learned)}"]
        for w, (old, new, eo, en) in self.learned.items():
            lines.append(f"  {w}: {old} → {new}  (error {eo:.2f} → {en:.2f})")
        return "\n".join(lines)


_VOWEL_SWAPS = {"/": "eo", "e": "/o", "o": "/e", "a": "/", "i": "e", "u": "o"}


def _vowel_variants(pron: str, limit: int = 6) -> list[str]:
    out = []
    for i, ch in enumerate(pron):
        for alt in _VOWEL_SWAPS.get(ch, ""):
            out.append(pron[:i] + alt + pron[i + 1 :])
    return out[:limit]


class PronunciationLearner:
    def __init__(self, voice: object, asr: object, lexicon: object | None = None, trials: int = 3,
                 margin: float = 0.5, min_gain: float = 0.02, max_candidates: int = 6) -> None:
        self.voice = voice
        self.asr = asr
        self.lexicon = lexicon if lexicon is not None else voice.lexicon
        self.trials = trials
        self.margin = margin  # required relative drop of the sentence error (0.5 = halve it)
        self.min_gain = min_gain  # and at least this much absolute
        self.max_candidates = max_candidates

    # ------------------------------------------------------------- hearing
    def _hear(self, text: str, overrides: dict[str, str] | None = None) -> str:
        ipa = self.voice.phonemize(text, overrides=overrides) if overrides else self.voice.phonemize(text)
        wav = self.voice.synthesize_ipa(ipa)
        return self.asr.transcribe(wav, self.voice.sample_rate)

    def _errors(self, text: str, overrides: dict[str, str] | None = None) -> list[float]:
        ref = normalize_for_compare(text)
        return [_cer(ref, normalize_for_compare(self._hear(text, overrides))) for _ in range(self.trials)]

    def audit(self, sentence: str) -> list[Suspect]:
        heard = self._hear(sentence)
        ref, hyp = normalize_for_compare(sentence), normalize_for_compare(heard)
        raw_words = [w for w in re.split(r"[\s،,.!?؟؛:]+", unicodedata.normalize("NFC", sentence)) if w]
        if len(raw_words) != len(ref):  # keep original spelling for the lexicon key
            raw_words = ref
        sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
        out = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("replace", "delete"):
                for i in range(i1, i2):
                    if len(ref[i]) >= 2:
                        out.append(Suspect(raw_words[i], sentence, " ".join(hyp[j1:j2])))
        return out

    # -------------------------------------------------------------- repair
    def _current(self, word: str) -> str | None:
        lex = self.lexicon.lookup(word) if self.lexicon is not None else None
        if lex:
            return lex
        cands = self.voice.word_candidates(word, 1)
        return cands[0] if cands else None

    def repair(self, s: Suspect) -> tuple[str, str, float, float] | None:
        current = self._current(s.word)
        if not current:
            return None
        cands = [c for c in self.voice.word_candidates(s.word, self.max_candidates) if c != current]
        cands += [v for v in _vowel_variants(current.lstrip("ˈ")) if v not in cands and v != current]
        cands = cands[: self.max_candidates]
        if not cands:
            return None
        base_errs = self._errors(s.sentence, {s.word: current})
        base = float(np.mean(base_errs))
        best, best_err, best_errs = None, base, base_errs
        for c in cands:
            errs = self._errors(s.sentence, {s.word: c})
            if float(np.mean(errs)) < best_err:
                best, best_err, best_errs = c, float(np.mean(errs)), errs
        # Accept only a clear, consistent win: the mean error drops by `margin`
        # (relative) and `min_gain` (absolute), AND the candidate's worst trial
        # beats the current reading's best one.
        # VITS is stochastic and a recognizer can prefer a wrong reading by
        # chance; a lucky trial must not rewrite the lexicon.
        gain = base - best_err
        if (best is not None and gain >= self.min_gain and gain >= self.margin * base
                and max(best_errs) < min(base_errs)):
            return current, best, base, best_err
        return None

    def learn(self, sentences: Iterable[str], max_words: int = 12) -> LearnReport:
        rep = LearnReport()
        seen: set[str] = set()
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            rep.audited += 1
            for sus in self.audit(sent):
                rep.suspects.append(sus)
                if sus.word in seen or len(rep.learned) >= max_words:
                    continue
                if self.lexicon is not None and getattr(self.lexicon, "is_curated", lambda w: False)(sus.word):
                    continue  # a human already decided this word
                seen.add(sus.word)
                fix = self.repair(sus)
                if fix:
                    rep.learned[sus.word] = fix
                    if self.lexicon is not None:
                        self.lexicon.learn(sus.word, fix[1])
                    self.voice.clear_cache()
        return rep


def _main(argv: list[str] | None = None) -> int:
    import argparse

    from ava_voice.asr.whisper_onnx import WhisperOnnxASR
    from ava_voice.tts.vits import PersianVITS

    ap = argparse.ArgumentParser(description="Let Ava find and fix her own mispronunciations")
    ap.add_argument("sentences", help="text file, one sentence per line")
    ap.add_argument("--whisper-dir", required=True)
    ap.add_argument("--voice-dir")
    ap.add_argument("--g2p-dir")
    ap.add_argument("--max-words", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true", help="report, don't write the learned lexicon")
    args = ap.parse_args(argv)
    voice = PersianVITS(args.voice_dir, args.g2p_dir)
    learner = PronunciationLearner(voice, WhisperOnnxASR(args.whisper_dir))
    if args.dry_run:
        learner.lexicon = None
    lines = [l for l in open(args.sentences, encoding="utf-8") if l.strip()]
    print(learner.learn(lines, args.max_words).summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
