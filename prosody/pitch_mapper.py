"""Emotion → pitch mapping (Phase 1C).

Pitch (fundamental frequency, F0) is the single most salient carrier of emotion
in speech. This module deterministically maps an
:class:`~ava_voice.emotion.emotion_schema.EmotionalVector` onto three pitch
controls:

* **pitch height** (``base``) — a semitone offset from the speaker baseline.
* **pitch variability** (``variance``) — how much F0 moves (monotone ↔ animated).
* **movement tendency** (``tendency``) — the qualitative direction/stability of
  that movement (:class:`PitchTendency`).

Emotion rules (per specification)
---------------------------------
* **joy** → higher pitch, lively variability, rising tendency.
* **sadness** → lower pitch, compressed variability, falling tendency.
* **anger** → moderately raised pitch with *sharp jumps* (very high variability,
  unstable/broken movement).
* **fear** → raised pitch with *unstable* wavering (very high variability).
* **calm** → slightly lowered, compressed, level pitch.

Persian-specific expressive behavior
-------------------------------------
* **Register control** — the ``formality`` channel *compresses* pitch variance
  (رسمی speech in Persian is deliberately measured and less modulated), pulling
  the contour toward a controlled, level delivery.
* **Question anchoring** — Persian yes/no questions (``آیا …؟`` / bare rising
  ``؟``) carry their rise on the *final* boundary; this module supplies the
  emotion-driven variability, while the sentence-terminal rise itself is applied
  by :mod:`~ava_voice.prosody.contour_generator` from the punctuation signal, so
  the two concerns stay cleanly separated.
* **Elevation with intensity** — expressive intensity widens the pitch excursion
  symmetrically, matching the wide F0 range characteristic of emphatic Persian
  speech.

All outputs are pure functions of the input vector: no randomness, no ML.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final

from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import EmotionalVector, clamp
from ava_voice.prosody.prosody_schema import blend_by_emotion

_logger = get_logger("prosody.pitch")


class PitchTendency(str, enum.Enum):
    """Qualitative direction / stability of pitch movement."""

    RISING = "rising"
    FALLING = "falling"
    LEVEL = "level"
    UNSTABLE = "unstable"  # large, irregular excursions (anger/fear)


# Per-emotion pitch-height anchors, in semitones relative to baseline.
_PITCH_BASE_ANCHORS: Final[dict[str, float]] = {
    "joy": 4.0,
    "sadness": -4.5,
    "anger": 2.0,
    "fear": 3.0,
    "calm": -1.5,
    "humor": 3.0,
    "formality": -1.0,
}

# Per-emotion pitch-variability anchors, normalized to [0, 1].
_PITCH_VARIANCE_ANCHORS: Final[dict[str, float]] = {
    "joy": 0.60,
    "sadness": 0.22,
    "anger": 0.85,
    "fear": 0.90,
    "calm": 0.15,
    "humor": 0.70,
    "formality": 0.30,
}

# Thresholds used to classify the movement tendency.
_UNSTABLE_ACTIVATION: Final[float] = 0.35   # anger/fear activation for instability
_DIRECTIONAL_VALENCE: Final[float] = 0.15   # |valence| needed to pick rise/fall


@dataclass(frozen=True)
class PitchProfile:
    """Deterministic pitch description derived from emotion.

    Attributes
    ----------
    base:
        Pitch height as a semitone offset from baseline, in ``[-12, +12]``.
    variance:
        Normalized pitch variability in ``[0, 1]``.
    tendency:
        The qualitative :class:`PitchTendency` of pitch movement.
    """

    base: float
    variance: float
    tendency: PitchTendency

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "base": self.base,
            "variance": self.variance,
            "tendency": self.tendency.value,
        }


class PitchMapper:
    """Deterministic mapper from an :class:`EmotionalVector` to a pitch profile."""

    def map(self, vector: EmotionalVector) -> PitchProfile:
        """Map ``vector`` to a :class:`PitchProfile`.

        Height is the emotion-weighted anchor scaled by expressive intensity
        (low intensity compresses toward baseline). Variability is the
        emotion-weighted anchor widened by arousal/intensity and then *narrowed*
        by the formality register. Tendency is chosen from instability
        (anger/fear) first, then valence direction, then level.
        """
        scores = vector.primary_scores

        base = blend_by_emotion(scores, _PITCH_BASE_ANCHORS, default=0.0)
        # Intensity scales the excursion: neutral speech sits near baseline.
        base *= 0.5 + 0.5 * vector.intensity

        variance = blend_by_emotion(scores, _PITCH_VARIANCE_ANCHORS, default=0.2)
        variance = variance * (0.6 + 0.4 * vector.arousal) + 0.1 * vector.intensity
        # Persian register control: formality flattens modulation.
        variance *= 1.0 - 0.3 * vector.formality

        tendency = self._tendency(vector)

        return PitchProfile(
            base=clamp(base, -12.0, 12.0),
            variance=clamp(variance, 0.0, 1.0),
            tendency=tendency,
        )

    @staticmethod
    def _tendency(vector: EmotionalVector) -> PitchTendency:
        """Classify pitch-movement tendency deterministically."""
        # Anger and fear produce large, irregular excursions first.
        if max(vector.anger, vector.fear) >= _UNSTABLE_ACTIVATION:
            return PitchTendency.UNSTABLE
        if vector.valence >= _DIRECTIONAL_VALENCE:
            return PitchTendency.RISING
        if vector.valence <= -_DIRECTIONAL_VALENCE:
            return PitchTendency.FALLING
        return PitchTendency.LEVEL


_default_mapper: Final[PitchMapper] = PitchMapper()


def map_pitch(vector: EmotionalVector) -> PitchProfile:
    """Map ``vector`` to a pitch profile using the module-level default mapper."""
    return _default_mapper.map(vector)
