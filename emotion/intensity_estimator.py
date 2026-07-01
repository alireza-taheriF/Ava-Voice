"""Emotional intensity estimation (Phase 1B).

Intensity is *orthogonal* to which emotion is present: ``خوبه`` and
``عاااالی!!!`` may share the ``joy`` channel yet differ enormously in expressive
magnitude. This module estimates that magnitude — a single scalar in ``[0, 1]``
— from surface, typographic and lexical cues.

.. important::
   The estimator must run on the **raw** (pre-normalization) text. The Persian
   normalizer deliberately collapses repeated sentence-final punctuation
   (``!!!`` → ``!``); running intensity estimation afterwards would erase the
   very signal it depends on. Elongation (repeated letters) survives
   normalization but is also read from the raw form for consistency.

Signals (each mapped to a sub-score in ``[0, 1]``)
--------------------------------------------------
* **Punctuation** — count of exclamation/question marks and their repetition
  (``!!!`` ≫ ``!``).
* **Repetition** — the same word repeated for emphasis (``برو برو برو``).
* **Elongation** — repeated letters within a word (``عاااالی``, ``نههههه``).
* **Emphasis** — presence of intensifier tokens (``خیلی``، ``بسیار``).
* **Lexical strength** — the strongest emotion weight matched by the classifier
  (supplied by the caller; strong words are inherently intense).

Fusion
------
Sub-scores combine with a **probabilistic OR** (noisy-or):

.. math::  I = 1 - \\prod_i (1 - w_i \\cdot s_i)

so multiple weak cues accumulate with diminishing returns and any single strong
cue already pushes intensity high — matching how humans read emphasis. The
computation is deterministic and closed-form (no ML, no randomness).

Examples
--------
>>> est = IntensityEstimator()
>>> est.estimate("عاااالی!!!").intensity > 0.8
True
>>> est.estimate("خوبه").intensity < 0.3
True
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import clamp

_logger = get_logger("emotion.intensity")

ZWNJ: Final[str] = "\u200c"

# Word tokenizer (kept local so intensity can run on raw text independently).
_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[0-9A-Za-z\u0621-\u06ff\u200c]+")

# Emphatic terminal punctuation whose repetition scales intensity.
_EMPHATIC_MARKS: Final[frozenset[str]] = frozenset({"!", "؟", "?"})

# Intensifier tokens (ZWNJ-stripped) that mark deliberate emphasis.
_EMPHASIS_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "خیلی",
        "بسیار",
        "واقعاً",
        "واقعا",
        "بهشدت",
        "شدیداً",
        "حسابی",
        "فوقالعاده",
        "کاملاً",
        "خیلیخیلی",
        "بینهایت",
    }
)

# Per-signal fusion weights (how much a saturated sub-score can contribute).
_W_PUNCTUATION: Final[float] = 0.75
_W_REPETITION: Final[float] = 0.70
_W_ELONGATION: Final[float] = 0.85
_W_EMPHASIS: Final[float] = 0.65
_W_LEXICAL: Final[float] = 0.80

# Saturation constants: how many "units" of a cue reach a near-1 sub-score.
_PUNCT_SATURATION: Final[float] = 4.0   # ~4 emphatic marks -> saturated
_ELONG_SATURATION: Final[float] = 4.0   # ~4 extra repeated chars -> saturated
_REPEAT_SATURATION: Final[float] = 2.0  # ~2 extra repeats -> saturated


@dataclass(frozen=True)
class IntensityEstimate:
    """The estimated intensity plus its decomposed, auditable sub-scores.

    Attributes
    ----------
    intensity:
        The fused overall intensity in ``[0, 1]``.
    punctuation, repetition, elongation, emphasis, lexical:
        The individual signal sub-scores, each in ``[0, 1]``, retained so the
        result is explainable (important for a patent-grade specification and
        for debugging prosody output).
    """

    intensity: float
    punctuation: float
    repetition: float
    elongation: float
    emphasis: float
    lexical: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable representation."""
        return {
            "intensity": self.intensity,
            "punctuation": self.punctuation,
            "repetition": self.repetition,
            "elongation": self.elongation,
            "emphasis": self.emphasis,
            "lexical": self.lexical,
        }


class IntensityEstimator:
    """Deterministic, closed-form emotional intensity estimator."""

    def estimate(
        self, raw_text: str, *, lexical_strength: float = 0.0
    ) -> IntensityEstimate:
        """Estimate intensity from ``raw_text`` and an optional lexical strength.

        Parameters
        ----------
        raw_text:
            The *original* text, before orthographic normalization, so
            punctuation repetition is preserved.
        lexical_strength:
            The strongest emotion weight matched by the classifier, in
            ``[0, 1]``. Defaults to ``0`` for a purely typographic estimate.
        """
        text = raw_text or ""
        tokens = _WORD_RE.findall(text)

        punctuation = self._punctuation_score(text)
        repetition = self._repetition_score(tokens)
        elongation = self._elongation_score(tokens)
        emphasis = self._emphasis_score(tokens)
        lexical = clamp(lexical_strength, 0.0, 1.0)

        intensity = self._fuse(
            (punctuation, _W_PUNCTUATION),
            (repetition, _W_REPETITION),
            (elongation, _W_ELONGATION),
            (emphasis, _W_EMPHASIS),
            (lexical, _W_LEXICAL),
        )
        return IntensityEstimate(
            intensity=intensity,
            punctuation=punctuation,
            repetition=repetition,
            elongation=elongation,
            emphasis=emphasis,
            lexical=lexical,
        )

    # -- Individual signals ------------------------------------------------- #
    @staticmethod
    def _punctuation_score(text: str) -> float:
        """Score emphatic-mark usage, rewarding repetition (``!!!`` > ``!``)."""
        count = sum(1 for ch in text if ch in _EMPHATIC_MARKS)
        if count == 0:
            return 0.0
        return clamp(count / _PUNCT_SATURATION, 0.0, 1.0)

    @staticmethod
    def _repetition_score(tokens: list[str]) -> float:
        """Score immediate word repetition (``برو برو برو``)."""
        if len(tokens) < 2:
            return 0.0
        extra_repeats = 0
        for previous, current in zip(tokens, tokens[1:]):
            if _strip(previous) == _strip(current) and _strip(current):
                extra_repeats += 1
        if extra_repeats == 0:
            return 0.0
        return clamp(extra_repeats / _REPEAT_SATURATION, 0.0, 1.0)

    @staticmethod
    def _elongation_score(tokens: list[str]) -> float:
        """Score the strongest intra-word letter elongation across ``tokens``.

        A run of ``n`` identical letters contributes ``n - 2`` "extra" units
        (two identical letters can be lexical, e.g. gemination; three or more is
        expressive), matching the linguistics stress detector's threshold.
        """
        max_extra = 0
        for token in tokens:
            surface = _strip(token)
            run = 1
            best = 1
            for previous, current in zip(surface, surface[1:]):
                run = run + 1 if current == previous else 1
                best = max(best, run)
            max_extra = max(max_extra, max(0, best - 2))
        if max_extra == 0:
            return 0.0
        return clamp(max_extra / _ELONG_SATURATION, 0.0, 1.0)

    @staticmethod
    def _emphasis_score(tokens: list[str]) -> float:
        """Score the presence of explicit intensifier tokens (saturating)."""
        hits = sum(1 for t in tokens if _strip(t) in _EMPHASIS_TOKENS)
        if hits == 0:
            return 0.0
        # One intensifier is already a strong cue; a second saturates.
        return clamp(0.7 + 0.3 * (hits - 1), 0.0, 1.0)

    # -- Fusion ------------------------------------------------------------- #
    @staticmethod
    def _fuse(*weighted: tuple[float, float]) -> float:
        """Combine ``(sub_score, weight)`` pairs via a noisy-or aggregation."""
        product = 1.0
        for score, weight in weighted:
            contribution = clamp(score, 0.0, 1.0) * clamp(weight, 0.0, 1.0)
            product *= 1.0 - contribution
        return clamp(1.0 - product, 0.0, 1.0)


def _strip(token: str) -> str:
    """Return ``token`` with ZWNJ removed (for stable comparisons)."""
    return token.replace(ZWNJ, "")


_default_estimator: Final[IntensityEstimator] = IntensityEstimator()


def estimate_intensity(
    raw_text: str, *, lexical_strength: float = 0.0
) -> IntensityEstimate:
    """Estimate intensity using the module-level default estimator."""
    return _default_estimator.estimate(raw_text, lexical_strength=lexical_strength)
