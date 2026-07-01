"""Emotion → tempo (speaking rate) mapping (Phase 1C).

Speaking rate is a primary affective cue: aroused, positive states speed speech
up; low, depressed states slow it down; anxiety makes it irregular. This module
deterministically maps an
:class:`~ava_voice.emotion.emotion_schema.EmotionalVector` to a speaking-rate
multiplier plus a regularity read-out.

Emotion rules (per specification)
---------------------------------
* **joy** → faster.
* **sadness** → slower.
* **anger** → faster, forceful.
* **fear** → irregular (rate itself near-neutral, but high variability).
* **calm** → stable, slightly slowed.

Method
------
The rate deviation from baseline is the emotion-weighted average of per-emotion
anchors, amplified by arousal (an aroused speaker exaggerates the deviation).
Rhythmic *regularity* is a separate weighted signal — dominated by ``fear`` —
that flips the :class:`TempoRegularity` flag when it crosses a threshold. All
outputs are pure functions of the input vector (no randomness, no ML).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final

from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import EmotionalVector, clamp
from ava_voice.prosody.prosody_schema import blend_by_emotion

_logger = get_logger("prosody.tempo")


class TempoRegularity(str, enum.Enum):
    """Whether the rhythmic pace is steady or fluctuating."""

    STABLE = "stable"
    IRREGULAR = "irregular"


# Per-emotion rate *deviation* anchors (added to the 1.0 baseline multiplier).
_RATE_DELTA_ANCHORS: Final[dict[str, float]] = {
    "joy": 0.25,
    "sadness": -0.30,
    "anger": 0.20,
    "fear": 0.12,
    "calm": -0.08,
    "humor": 0.18,
    "formality": -0.12,
}

# Per-emotion rhythmic-variability anchors (fear dominates irregularity).
_VARIABILITY_ANCHORS: Final[dict[str, float]] = {
    "joy": 0.30,
    "sadness": 0.20,
    "anger": 0.55,
    "fear": 0.90,
    "calm": 0.05,
    "humor": 0.45,
    "formality": 0.15,
}

# Rhythmic-variability above this is reported as IRREGULAR.
_IRREGULAR_THRESHOLD: Final[float] = 0.50


@dataclass(frozen=True)
class TempoProfile:
    """Deterministic tempo description derived from emotion.

    Attributes
    ----------
    rate:
        Speaking rate as a multiplier of baseline, in ``[0.5, 2.0]``.
    variability:
        Normalized rhythmic variability in ``[0, 1]`` (higher = less even).
    regularity:
        The discrete :class:`TempoRegularity` flag derived from ``variability``.
    """

    rate: float
    variability: float
    regularity: TempoRegularity

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "rate": self.rate,
            "variability": self.variability,
            "regularity": self.regularity.value,
        }


class TempoMapper:
    """Deterministic mapper from an :class:`EmotionalVector` to a tempo profile."""

    def map(self, vector: EmotionalVector) -> TempoProfile:
        """Map ``vector`` to a :class:`TempoProfile`.

        The baseline rate deviation is amplified by arousal so that, e.g., joyful
        *and* aroused speech is faster than joyful-but-subdued speech. Rhythmic
        variability is an independent weighted signal (fear-dominant) that sets
        the regularity flag.
        """
        scores = vector.primary_scores

        delta = blend_by_emotion(scores, _RATE_DELTA_ANCHORS, default=0.0)
        # Arousal amplifies the tempo deviation from baseline.
        delta *= 0.7 + 0.6 * vector.arousal
        rate = clamp(1.0 + delta, 0.5, 2.0)

        variability = blend_by_emotion(scores, _VARIABILITY_ANCHORS, default=0.1)
        variability = clamp(variability * (0.7 + 0.3 * vector.intensity), 0.0, 1.0)
        regularity = (
            TempoRegularity.IRREGULAR
            if variability >= _IRREGULAR_THRESHOLD
            else TempoRegularity.STABLE
        )

        return TempoProfile(
            rate=rate, variability=variability, regularity=regularity
        )


_default_mapper: Final[TempoMapper] = TempoMapper()


def map_tempo(vector: EmotionalVector) -> TempoProfile:
    """Map ``vector`` to a tempo profile using the module-level default mapper."""
    return _default_mapper.map(vector)
