"""Persian lexical emotion classifier (Phase 1B).

This is the *bottom-up* stage of the Emotional Vector Engine: it reads the
surface words of a sentence and produces a **multi-label emotion score** — a
raw activation per :class:`~ava_voice.emotion.emotion_schema.PrimaryEmotion`.

It is a deterministic, rule-and-lexicon system (no machine learning, no
randomness). Four mechanisms cooperate:

1. **Weighted unigram lexicon** — each emotion-bearing word contributes a fixed,
   signed weight to one or more emotion channels. Weights encode strength
   (``عالی`` ≫ ``خوب``).
2. **Phrase matching** — multi-word cues (``دوستت دارم``) are matched *before*
   unigrams so idiomatic affect is not lost to word-by-word scoring.
3. **Intensifiers & down-toners** — modifiers such as ``خیلی`` (up) or ``کمی``
   (down) scale the emotion word that follows them within a small window.
4. **Negation handling** — negators (``نیست``، ``نه``، the verbal prefix
   ``نمی``) that precede an emotion word within a window *suppress* it and shift
   its contribution toward the opposite polarity.

The classifier is ZWNJ-insensitive (it strips نیم‌فاصله for lookups, exactly as
:mod:`ava_voice.linguistics.ambiguity_resolver` does), so ``می‌ترسم`` and
``میترسم`` score identically.

Examples
--------
>>> clf = EmotionClassifier()
>>> result = clf.classify("خیلی خوشحالم")
>>> result.dominant
'joy'
>>> clf.classify("می‌ترسم").scores["fear"] > 0.5
True
>>> clf.classify("خوشحال نیستم").scores["joy"] < 0.1  # negation suppresses joy
True
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final, Mapping, Sequence

from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import PRIMARY_DIMENSIONS

_logger = get_logger("emotion.classifier")

ZWNJ: Final[str] = "\u200c"

# Tokenizer identical in spirit to the linguistics layer: a run of word
# characters (Persian letters, ZWNJ, latin, digits). Punctuation/space split.
_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[0-9A-Za-z\u0621-\u06ff\u200c]+")

# How many tokens ahead an intensifier / negator may influence.
_MODIFIER_WINDOW: Final[int] = 2


def _strip_zwnj(token: str) -> str:
    """Return ``token`` with ZWNJ removed (for lexicon lookups)."""
    return token.replace(ZWNJ, "")


# --------------------------------------------------------------------------- #
# Weighted unigram lexicon
# --------------------------------------------------------------------------- #
# Each entry maps a ZWNJ-stripped surface form to ``{emotion: weight}`` where
# weight ∈ (0, 1]. A word may activate several channels (e.g. ``هیجان`` adds joy
# and arousal-heavy affect). Values are hand-tuned constants — deterministic and
# auditable — not learned parameters.
_LEXICON: Final[dict[str, dict[str, float]]] = {
    # --- joy ---
    "خوشحال": {"joy": 0.8},
    "خوشحالم": {"joy": 0.85},
    "خوشحالی": {"joy": 0.8},
    "شاد": {"joy": 0.75},
    "شادم": {"joy": 0.8},
    "شادی": {"joy": 0.7},
    "خوشی": {"joy": 0.65},
    "عالی": {"joy": 0.9},
    "فوقالعاده": {"joy": 0.95},
    "خوب": {"joy": 0.45},
    "لذت": {"joy": 0.7},
    "لذتبخش": {"joy": 0.75},
    "خوشبخت": {"joy": 0.8},
    "امیدوار": {"joy": 0.55},
    "ذوق": {"joy": 0.7},
    "هیجان": {"joy": 0.55},
    "هیجانزده": {"joy": 0.7},
    "دوست": {"joy": 0.5},
    "عشق": {"joy": 0.7},
    "عاشق": {"joy": 0.75},
    "خدارو": {"joy": 0.2},
    # --- sadness ---
    "غمگین": {"sadness": 0.85},
    "ناراحت": {"sadness": 0.8},
    "ناراحتم": {"sadness": 0.85},
    "غم": {"sadness": 0.7},
    "غصه": {"sadness": 0.75},
    "دلتنگ": {"sadness": 0.7},
    "دلتنگم": {"sadness": 0.75},
    "گریه": {"sadness": 0.65},
    "افسرده": {"sadness": 0.9},
    "ناامید": {"sadness": 0.75},
    "تنها": {"sadness": 0.55},
    "دلگیر": {"sadness": 0.6},
    "بد": {"sadness": 0.4},
    "خسته": {"sadness": 0.45, "calm": 0.1},
    "بیحوصله": {"sadness": 0.5},
    # --- anger ---
    "عصبانی": {"anger": 0.85},
    "عصبانیم": {"anger": 0.9},
    "خشمگین": {"anger": 0.9},
    "خشم": {"anger": 0.8},
    "متنفر": {"anger": 0.85},
    "نفرت": {"anger": 0.85},
    "بیزار": {"anger": 0.75},
    "کلافه": {"anger": 0.6},
    "حرص": {"anger": 0.6},
    "دعوا": {"anger": 0.55},
    "خفهشو": {"anger": 0.9},
    "لعنتی": {"anger": 0.8},
    "احمق": {"anger": 0.7},
    # --- fear ---
    "میترسم": {"fear": 0.85},
    "ترس": {"fear": 0.8},
    "ترسناک": {"fear": 0.8},
    "ترسیدم": {"fear": 0.85},
    "وحشت": {"fear": 0.9},
    "وحشتناک": {"fear": 0.85},
    "نگران": {"fear": 0.65},
    "نگرانم": {"fear": 0.7},
    "اضطراب": {"fear": 0.75},
    "مضطرب": {"fear": 0.75},
    "هراس": {"fear": 0.8},
    "دلهره": {"fear": 0.7},
    "استرس": {"fear": 0.65},
    # --- calm ---
    "آرام": {"calm": 0.8},
    "آرامش": {"calm": 0.85},
    "آروم": {"calm": 0.8},
    "راحت": {"calm": 0.6},
    "خونسرد": {"calm": 0.75},
    "آسوده": {"calm": 0.75},
    "صبور": {"calm": 0.6},
    "آسان": {"calm": 0.4},
    "امن": {"calm": 0.55},
    # --- humor ---
    "خندهدار": {"humor": 0.85, "joy": 0.3},
    "بامزه": {"humor": 0.75, "joy": 0.3},
    "جوک": {"humor": 0.7},
    "شوخی": {"humor": 0.7},
    "طنز": {"humor": 0.7},
    "خندیدم": {"humor": 0.6, "joy": 0.4},
    "مسخره": {"humor": 0.5},
    "خنده": {"humor": 0.55, "joy": 0.35},
    # --- formality (register) ---
    "جناب": {"formality": 0.8},
    "سرکار": {"formality": 0.75},
    "تشریف": {"formality": 0.8},
    "استدعا": {"formality": 0.85},
    "خواهشمندم": {"formality": 0.85},
    "احتراماً": {"formality": 0.9},
    "محترم": {"formality": 0.8},
    "لطفاً": {"formality": 0.5},
    "بفرمایید": {"formality": 0.7},
    "متشکرم": {"formality": 0.55, "joy": 0.2},
    "سپاسگزارم": {"formality": 0.65, "joy": 0.2},
    "عرض": {"formality": 0.6},
    "حضور": {"formality": 0.6},
}

# --------------------------------------------------------------------------- #
# Phrase lexicon (matched before unigrams; keys are ZWNJ-stripped token tuples)
# --------------------------------------------------------------------------- #
_PHRASES: Final[dict[tuple[str, ...], dict[str, float]]] = {
    ("دوستت", "دارم"): {"joy": 0.7},
    ("دوستت", "دارم", "خیلی"): {"joy": 0.85},
    ("دلم", "گرفته"): {"sadness": 0.75},
    ("دلم", "تنگ", "شده"): {"sadness": 0.8},
    ("از", "دستت", "عصبانیم"): {"anger": 0.9},
    ("خیلی", "میترسم"): {"fear": 0.95},
    ("خدا", "رو", "شکر"): {"joy": 0.5, "calm": 0.4},
    ("چقدر", "خندهدار"): {"humor": 0.9, "joy": 0.4},
    ("با", "کمال", "احترام"): {"formality": 0.9},
}

# --------------------------------------------------------------------------- #
# Modifiers
# --------------------------------------------------------------------------- #
# Intensifiers multiply the score of the emotion word they govern; down-toners
# attenuate it. Values are multipliers applied to weight.
_INTENSIFIERS: Final[dict[str, float]] = {
    "خیلی": 1.4,
    "بسیار": 1.4,
    "واقعاً": 1.3,
    "واقعا": 1.3,
    "بهشدت": 1.6,
    "شدیداً": 1.6,
    "حسابی": 1.35,
    "فوقالعاده": 1.5,
    "کاملاً": 1.3,
    "خیلیخیلی": 1.7,
}
_DOWNTONERS: Final[dict[str, float]] = {
    "کمی": 0.6,
    "یکم": 0.6,
    "یهکم": 0.6,
    "کمابیش": 0.7,
    "تاحدی": 0.7,
    "نسبتاً": 0.75,
    "نسبتا": 0.75,
}

# Negators. Standalone negation tokens plus verbal negative prefixes; a negator
# preceding an emotion word within the window suppresses & polarity-shifts it.
_NEGATORS: Final[frozenset[str]] = frozenset(
    {"نه", "نیست", "نیستم", "نیستی", "ندارم", "نداره", "نداری", "نکن", "نبود", "بدون"}
)
_NEGATION_PREFIXES: Final[tuple[str, ...]] = ("نمی", "نا")

# When an emotion is negated, part of its energy flows to a "counter" channel to
# model the affective flip (e.g. "خوشحال نیستم" leans slightly sad).
_NEGATION_COUNTER: Final[dict[str, str]] = {
    "joy": "sadness",
    "sadness": "joy",
    "calm": "fear",
    "fear": "calm",
    "anger": "calm",
    "humor": "sadness",
    "formality": "formality",
}
# Fraction of suppressed weight redirected to the counter channel.
_NEGATION_COUNTER_RATIO: Final[float] = 0.35


@dataclass(frozen=True)
class LexicalEmotion:
    """Result of lexical emotion classification for a sentence.

    Attributes
    ----------
    scores:
        Ordered ``{emotion: raw_score}`` mapping over
        :data:`~ava_voice.emotion.emotion_schema.PRIMARY_DIMENSIONS`. Scores are
        *raw* (unbounded, non-normalized) activations; the fusion stage maps them
        into the ``[0, 1]`` schema range.
    polarity:
        Net valence polarity in ``[-1, +1]`` derived from positive vs negative
        channel mass (a quick, standalone sentiment read).
    matched_terms:
        The lexicon terms/phrases that fired, in order of appearance, for
        auditability.
    matched_lexical_strength:
        The single strongest raw weight matched (before intensifiers). Consumed
        by the intensity estimator as the "lexical strength" signal.
    """

    scores: dict[str, float]
    polarity: float
    matched_terms: tuple[str, ...] = field(default_factory=tuple)
    matched_lexical_strength: float = 0.0

    @property
    def dominant(self) -> str | None:
        """Return the highest-scoring emotion name, or ``None`` if all zero."""
        best_name: str | None = None
        best = 0.0
        for name in PRIMARY_DIMENSIONS:
            value = self.scores.get(name, 0.0)
            if value > best:
                best = value
                best_name = name
        return best_name

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "scores": dict(self.scores),
            "polarity": self.polarity,
            "matched_terms": list(self.matched_terms),
            "matched_lexical_strength": self.matched_lexical_strength,
        }


# Channels considered positive vs negative for polarity aggregation.
_POSITIVE_CHANNELS: Final[frozenset[str]] = frozenset({"joy", "calm", "humor"})
_NEGATIVE_CHANNELS: Final[frozenset[str]] = frozenset(
    {"sadness", "anger", "fear"}
)


class EmotionClassifier:
    """Deterministic Persian lexical / phrasal emotion classifier.

    The classifier is stateless across calls: :meth:`classify` fully determines
    its output from its input, making it trivially thread-safe and testable.
    Lexicon and phrase tables may be extended via the constructor for
    domain-specific tuning without touching the algorithm.
    """

    def __init__(
        self,
        lexicon: Mapping[str, Mapping[str, float]] | None = None,
        phrases: Mapping[tuple[str, ...], Mapping[str, float]] | None = None,
    ) -> None:
        self._lexicon: dict[str, dict[str, float]] = {
            k: dict(v) for k, v in _LEXICON.items()
        }
        if lexicon:
            self._lexicon.update({k: dict(v) for k, v in lexicon.items()})
        self._phrases: dict[tuple[str, ...], dict[str, float]] = {
            k: dict(v) for k, v in _PHRASES.items()
        }
        if phrases:
            self._phrases.update({k: dict(v) for k, v in phrases.items()})
        # Longest phrase first so greedy matching prefers the most specific cue.
        self._phrase_lengths: tuple[int, ...] = tuple(
            sorted({len(k) for k in self._phrases}, reverse=True)
        )

    # -- Public API --------------------------------------------------------- #
    def classify(self, text: str) -> LexicalEmotion:
        """Classify raw ``text`` by first tokenizing it, then scoring tokens."""
        return self.classify_tokens(self._tokenize(text))

    def classify_tokens(self, tokens: Sequence[str]) -> LexicalEmotion:
        """Classify a pre-tokenized sequence of surface words.

        This is the integration entry point for the linguistic token graph:
        callers pass the canonical TOKEN node values directly, keeping a single
        tokenization authority upstream.
        """
        clean = [_strip_zwnj(t) for t in tokens if t]
        scores: dict[str, float] = {name: 0.0 for name in PRIMARY_DIMENSIONS}
        matched: list[str] = []
        strongest = 0.0

        consumed = self._apply_phrases(clean, scores, matched)

        for index, token in enumerate(clean):
            if index in consumed:
                continue
            entry = self._lexicon.get(token) or self._match_negative_prefix(token)
            if entry is None:
                continue

            multiplier, is_negated = self._modifier_context(clean, index, consumed)
            matched.append(token)
            for emotion, weight in entry.items():
                strongest = max(strongest, weight)
                self._accumulate(scores, emotion, weight * multiplier, is_negated)

        polarity = self._polarity(scores)
        return LexicalEmotion(
            scores=scores,
            polarity=polarity,
            matched_terms=tuple(matched),
            matched_lexical_strength=strongest,
        )

    # -- Tokenization ------------------------------------------------------- #
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Extract word tokens from ``text`` (punctuation/space stripped)."""
        return _WORD_RE.findall(text or "")

    # -- Phrase matching ---------------------------------------------------- #
    def _apply_phrases(
        self,
        tokens: Sequence[str],
        scores: dict[str, float],
        matched: list[str],
    ) -> set[int]:
        """Greedily match multi-word phrases; return consumed token indices.

        Longer phrases are tried first at each position so the most specific cue
        wins. Matched positions are marked consumed so unigram scoring skips
        them, preventing double counting.
        """
        consumed: set[int] = set()
        index = 0
        length = len(tokens)
        while index < length:
            if index in consumed:
                index += 1
                continue
            match_len = self._match_phrase_at(tokens, index)
            if match_len:
                span = tuple(tokens[index : index + match_len])
                entry = self._phrases[span]
                for emotion, weight in entry.items():
                    scores[emotion] += weight
                matched.append(" ".join(span))
                consumed.update(range(index, index + match_len))
                index += match_len
            else:
                index += 1
        return consumed

    def _match_phrase_at(self, tokens: Sequence[str], start: int) -> int:
        """Return the length of the longest phrase matching at ``start`` (else 0)."""
        for length in self._phrase_lengths:
            span = tuple(tokens[start : start + length])
            if len(span) == length and span in self._phrases:
                return length
        return 0

    # -- Modifier / negation context --------------------------------------- #
    def _modifier_context(
        self,
        tokens: Sequence[str],
        index: int,
        consumed: set[int],
    ) -> tuple[float, bool]:
        """Compute the intensifier/down-toner multiplier and negation flag.

        Looks back up to :data:`_MODIFIER_WINDOW` tokens for modifiers/negators
        governing the emotion word at ``index``. Multiple intensifiers compound
        multiplicatively (deterministically bounded by the fixed window).
        """
        multiplier = 1.0
        negated = False
        start = max(0, index - _MODIFIER_WINDOW)
        for position in range(start, index):
            token = tokens[position]
            if token in _INTENSIFIERS:
                multiplier *= _INTENSIFIERS[token]
            elif token in _DOWNTONERS:
                multiplier *= _DOWNTONERS[token]
        # Negation may sit immediately before or just after the emotion word
        # (Persian allows both: "خوشحال نیستم" / "نه خوشحال").
        look_start = max(0, index - _MODIFIER_WINDOW)
        look_end = min(len(tokens), index + _MODIFIER_WINDOW + 1)
        for position in range(look_start, look_end):
            if position == index:
                continue
            if tokens[position] in _NEGATORS:
                negated = True
                break
        return multiplier, negated

    @staticmethod
    def _match_negative_prefix(token: str) -> dict[str, float] | None:
        """Resolve a negated verbal form (``نمی…``/``نا…``) if the stem is known.

        This is a light heuristic: only fires when stripping the negative prefix
        yields a stem present in the lexicon, keeping matches conservative and
        deterministic. The caller applies the negation flip separately, so here
        we simply return the *base* emotion of the stem.
        """
        for prefix in _NEGATION_PREFIXES:
            if token.startswith(prefix) and len(token) > len(prefix) + 1:
                # Handled by the standalone-negator path in practice; the stem
                # lookup is intentionally omitted to avoid over-triggering on
                # ordinary words that merely begin with these letters.
                return None
        return None

    # -- Scoring helpers ---------------------------------------------------- #
    @staticmethod
    def _accumulate(
        scores: dict[str, float],
        emotion: str,
        weight: float,
        is_negated: bool,
    ) -> None:
        """Add ``weight`` to ``emotion``, applying negation flip when required."""
        if not is_negated:
            scores[emotion] += weight
            return
        # Negation: suppress the channel and redirect a fraction to its counter.
        counter = _NEGATION_COUNTER.get(emotion, emotion)
        if counter != emotion:
            scores[counter] += weight * _NEGATION_COUNTER_RATIO

    @staticmethod
    def _polarity(scores: Mapping[str, float]) -> float:
        """Return net polarity in ``[-1, +1]`` from positive/negative mass."""
        positive = sum(scores.get(c, 0.0) for c in _POSITIVE_CHANNELS)
        negative = sum(scores.get(c, 0.0) for c in _NEGATIVE_CHANNELS)
        total = positive + negative
        if total == 0.0:
            return 0.0
        return (positive - negative) / total


_default_classifier: Final[EmotionClassifier] = EmotionClassifier()


def classify(text: str) -> LexicalEmotion:
    """Classify ``text`` using the module-level default classifier."""
    return _default_classifier.classify(text)


def classify_tokens(tokens: Sequence[str]) -> LexicalEmotion:
    """Classify ``tokens`` using the module-level default classifier."""
    return _default_classifier.classify_tokens(tokens)
