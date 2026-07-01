"""Emotion → vocal energy mapping (Phase 1C).

Vocal energy (perceived loudness / effort, correlated with sub-glottal pressure
and spectral tilt) conveys how *forceful* speech is. This module deterministically
maps an :class:`~ava_voice.emotion.emotion_schema.EmotionalVector` to a single
normalized energy level in ``[0, 1]``.

Emotion rules (per specification)
---------------------------------
* **anger** → high energy.
* **calm** → low energy.
* **joy** → medium-high energy.
* **sadness** → low energy.
* **fear** → elevated (tense) energy.

Method
------
Energy is the emotion-weighted average of per-emotion anchors, then lifted by
overall expressive intensity and arousal (an intense, aroused delivery is louder
regardless of category). The result is clamped to ``[0, 1]``. Pure function of
the input vector — no randomness, no ML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import EmotionalVector, clamp
from ava_voice.prosody.prosody_schema import blend_by_emotion

_logger = get_logger("prosody.energy")


# Per-emotion energy anchors, normalized to [0, 1].
_ENERGY_ANCHORS: Final[dict[str, float]] = {
    "joy": 0.70,
    "sadness": 0.28,
    "anger": 0.92,
    "fear": 0.68,
    "calm": 0.20,
    "humor": 0.65,
    "formality": 0.48,
}

# Resting energy when no emotion is active (relaxed neutral speech).
_NEUTRAL_ENERGY: Final[float] = 0.40

# Weights blending the categorical anchor with the intensity/arousal drives.
_W_ANCHOR: Final[float] = 0.70
_W_INTENSITY: Final[float] = 0.20
_W_AROUSAL: Final[float] = 0.10


@dataclass(frozen=True)
class EnergyProfile:
    """Deterministic vocal-energy description derived from emotion.

    Attributes
    ----------
    level:
        Normalized vocal energy in ``[0, 1]``.
    """

    level: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {"level": self.level}


class EnergyMapper:
    """Deterministic mapper from an :class:`EmotionalVector` to an energy level."""

    def map(self, vector: EmotionalVector) -> EnergyProfile:
        """Map ``vector`` to an :class:`EnergyProfile`.

        The categorical anchor supplies the base level; expressive intensity and
        arousal add the remaining drive, so an angry-and-intense delivery
        approaches full energy while calm-and-subdued speech stays quiet.
        """
        anchor = blend_by_emotion(
            vector.primary_scores, _ENERGY_ANCHORS, default=_NEUTRAL_ENERGY
        )
        level = (
            _W_ANCHOR * anchor
            + _W_INTENSITY * vector.intensity
            + _W_AROUSAL * vector.arousal
        )
        return EnergyProfile(level=clamp(level, 0.0, 1.0))


_default_mapper: Final[EnergyMapper] = EnergyMapper()


def map_energy(vector: EmotionalVector) -> EnergyProfile:
    """Map ``vector`` to an energy profile using the module-level default mapper."""
    return _default_mapper.map(vector)
