"""Context-aware emotional adjustment (Phase 1B).

A sentence's *raw* emotion (from the lexical/intensity stages) is not yet its
*spoken* emotion: real speakers carry affect across sentence boundaries, ease
into new moods, and only snap when something genuinely shifts. This module
applies that temporal shaping.

Given the current sentence's vector and the recent history (an
:class:`~ava_voice.emotion.emotion_memory.EmotionMemory`), it produces a
**context-adjusted vector** plus an explanation of what it did:

* **Previous-sentence influence / continuity** — when the new emotion is
  *consistent* with the recent trend, the analyzer blends the current vector
  toward that trend (stronger smoothing), yielding natural continuity instead of
  abrupt per-sentence resets.
* **Context-shift detection** — when the new emotion is *far* from the trend
  (large affect distance) or flips valence polarity, a shift is declared and
  smoothing is reduced so the new mood is honoured promptly rather than being
  washed out by history.
* **Momentum reinforcement** — when the mood is already moving in the same
  direction as the current valence (aligned momentum), the current valence is
  nudged slightly further, modelling emotional build-up.

Everything is a deterministic function of ``(current, memory)`` — no randomness,
no clock.

Examples
--------
>>> from ava_voice.emotion.emotion_memory import EmotionMemory
>>> from ava_voice.emotion.emotion_schema import EmotionalVector
>>> mem = EmotionMemory()
>>> _ = mem.push(EmotionalVector(joy=0.7, valence=0.7))
>>> analyzer = ContextualAnalyzer()
>>> out = analyzer.adjust(EmotionalVector(anger=0.8, valence=-0.6), mem)
>>> out.shift_detected
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_memory import EmotionMemory
from ava_voice.emotion.emotion_schema import EmotionalVector, clamp

_logger = get_logger("emotion.context")

# Affect distance (Euclidean, all dimensions) above which a context shift is
# declared even without a valence-polarity flip.
_SHIFT_DISTANCE_THRESHOLD: Final[float] = 0.9

# Minimum absolute valence on both sides for a sign flip to count as a shift
# (guards against noise around zero valence).
_VALENCE_FLIP_THRESHOLD: Final[float] = 0.25

# How strongly the current vector is pulled toward the recent trend.
# Continuity (no shift) smooths more; a detected shift smooths far less so the
# new emotion dominates.
_CONTINUITY_SMOOTHING: Final[float] = 0.40
_SHIFT_SMOOTHING: Final[float] = 0.12

# Momentum reinforcement: fraction of the (aligned) valence slope added to the
# current valence to model emotional build-up.
_MOMENTUM_GAIN: Final[float] = 0.30


@dataclass(frozen=True)
class ContextAdjustment:
    """The outcome of contextual adjustment.

    Attributes
    ----------
    vector:
        The context-adjusted :class:`EmotionalVector`.
    shift_detected:
        ``True`` if a context shift (large distance or valence flip) was found.
    continuity:
        ``True`` if the current emotion was treated as continuous with history.
        Mutually exclusive with ``shift_detected``; both are ``False`` only when
        there is no history to compare against.
    smoothing_weight:
        The blend weight actually applied toward the trend (``0`` = the current
        vector was untouched).
    distance:
        The measured affect distance between the current vector and the trend.
    """

    vector: EmotionalVector
    shift_detected: bool
    continuity: bool
    smoothing_weight: float
    distance: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "vector": self.vector.to_dict(),
            "shift_detected": self.shift_detected,
            "continuity": self.continuity,
            "smoothing_weight": self.smoothing_weight,
            "distance": self.distance,
        }


class ContextualAnalyzer:
    """Deterministic temporal shaper of per-sentence emotion vectors."""

    def __init__(
        self,
        shift_distance_threshold: float = _SHIFT_DISTANCE_THRESHOLD,
        valence_flip_threshold: float = _VALENCE_FLIP_THRESHOLD,
        continuity_smoothing: float = _CONTINUITY_SMOOTHING,
        shift_smoothing: float = _SHIFT_SMOOTHING,
        momentum_gain: float = _MOMENTUM_GAIN,
    ) -> None:
        self._shift_distance = shift_distance_threshold
        self._valence_flip = valence_flip_threshold
        self._continuity_smoothing = clamp(continuity_smoothing, 0.0, 1.0)
        self._shift_smoothing = clamp(shift_smoothing, 0.0, 1.0)
        self._momentum_gain = momentum_gain

    def adjust(
        self, current: EmotionalVector, memory: EmotionMemory
    ) -> ContextAdjustment:
        """Return the context-adjusted vector for ``current`` given ``memory``.

        The memory is treated as read-only here: the caller decides when to push
        the (adjusted or raw) result, keeping this stage a pure transformation.
        """
        if memory.is_empty:
            return ContextAdjustment(
                vector=current,
                shift_detected=False,
                continuity=False,
                smoothing_weight=0.0,
                distance=0.0,
            )

        trend = memory.trend()
        distance = current.distance(trend)
        shift = self._is_shift(current, trend, distance)

        smoothing = self._shift_smoothing if shift else self._continuity_smoothing
        # Blend the current vector toward the historical trend.
        adjusted = current.blend(trend, smoothing)

        # Momentum reinforcement only applies during continuity: a build-up in
        # the *same* direction as the current mood is amplified slightly.
        if not shift:
            adjusted = self._apply_momentum(adjusted, memory)

        return ContextAdjustment(
            vector=adjusted,
            shift_detected=shift,
            continuity=not shift,
            smoothing_weight=smoothing,
            distance=distance,
        )

    # -- Internals ---------------------------------------------------------- #
    def _is_shift(
        self, current: EmotionalVector, trend: EmotionalVector, distance: float
    ) -> bool:
        """Decide whether ``current`` represents a context shift from ``trend``."""
        if distance >= self._shift_distance:
            return True
        # Valence polarity flip with meaningful magnitude on both sides.
        if (
            current.valence * trend.valence < 0.0
            and abs(current.valence) >= self._valence_flip
            and abs(trend.valence) >= self._valence_flip
        ):
            return True
        return False

    def _apply_momentum(
        self, vector: EmotionalVector, memory: EmotionMemory
    ) -> EmotionalVector:
        """Nudge valence along an aligned momentum (emotional build-up)."""
        momentum = memory.momentum()
        slope = momentum.valence_slope
        if slope == 0.0 or vector.valence == 0.0:
            return vector
        # Only reinforce when momentum and current valence share a sign.
        if slope * vector.valence <= 0.0:
            return vector
        new_valence = clamp(
            vector.valence + self._momentum_gain * slope, -1.0, 1.0
        )
        return vector.with_values(valence=new_valence)


_default_analyzer: Final[ContextualAnalyzer] = ContextualAnalyzer()


def adjust(
    current: EmotionalVector, memory: EmotionMemory
) -> ContextAdjustment:
    """Adjust ``current`` against ``memory`` using the default analyzer."""
    return _default_analyzer.adjust(current, memory)
