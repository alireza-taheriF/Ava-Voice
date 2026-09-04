"""Short-term emotional memory (Phase 1B).

Speech emotion is *sequential*: a sentence is coloured by what preceded it.
This module holds a bounded, rolling history of recent
:class:`~ava_voice.emotion.emotion_schema.EmotionalVector` states and derives
two aggregate read-outs that the contextual analyzer consumes:

* **Emotional trend** — the (optionally recency-weighted) mean of the window; a
  smoothed estimate of the current affective baseline.
* **Emotional momentum** — the direction and rate of change (a delta vector plus
  a signed valence slope); positive valence momentum means the mood is
  brightening, negative means it is darkening.

Determinism
-----------
The memory records a monotonic **sequence index** with each state, never a wall
-clock timestamp, so a given sequence of pushes always yields identical trend
and momentum values. There is no randomness and no external I/O.

Examples
--------
>>> mem = EmotionMemory(window=4)
>>> _ = mem.push(EmotionalVector(joy=0.2, valence=0.2))
>>> _ = mem.push(EmotionalVector(joy=0.8, valence=0.8))
>>> mem.momentum().valence_slope > 0   # mood brightening
True
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Final, Mapping

from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import (
    PRIMARY_DIMENSIONS,
    SECONDARY_DIMENSIONS,
    EmotionalVector,
    clamp,
)

_logger = get_logger("emotion.memory")

#: Default number of recent states retained in the rolling window.
DEFAULT_WINDOW: Final[int] = 5

# Geometric recency decay for the weighted trend: weight_i = decay ** age_i.
# ``1.0`` yields a plain mean; smaller values emphasize the most recent states.
_DEFAULT_DECAY: Final[float] = 0.7

# Bounds for a *difference* of two bounded values: [0,1] fields differ by at most
# ±1; valence is itself signed in [-1,+1], so it differs by at most ±2.
_DELTA_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "valence": (-2.0, 2.0),
}


@dataclass(frozen=True)
class EmotionalState:
    """A single recorded emotional state with its sequence index.

    Attributes
    ----------
    index:
        Monotonic push counter (0-based). Used instead of a timestamp to keep
        the memory fully deterministic.
    vector:
        The recorded :class:`EmotionalVector`.
    """

    index: int
    vector: EmotionalVector

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {"index": self.index, "vector": self.vector.to_dict()}


@dataclass(frozen=True)
class EmotionalDelta:
    """Signed per-dimension change between two :class:`EmotionalVector` states.

    A dedicated type rather than an :class:`EmotionalVector`, because a vector
    clamps its fields to their *absolute* ranges — ``[0, 1]`` for the primaries
    and for arousal/dominance/intensity — which silently rectifies every decrease
    to ``0.0``. A difference of bounded values is signed, so the primaries and
    arousal/dominance/intensity live in ``[-1, +1]`` here and valence, itself
    signed, lives in ``[-2, +2]``.

    Field names and ordering match
    :data:`~ava_voice.emotion.emotion_schema.PRIMARY_DIMENSIONS` followed by
    :data:`~ava_voice.emotion.emotion_schema.SECONDARY_DIMENSIONS`, so
    :meth:`to_dict` is directly comparable with ``EmotionalVector.to_dict``.

    Examples
    --------
    >>> delta = EmotionalDelta(joy=-0.8)
    >>> delta.joy
    -0.8
    """

    joy: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    calm: float = 0.0
    humor: float = 0.0
    formality: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    intensity: float = 0.0

    def __post_init__(self) -> None:
        """Clamp every field to its signed difference range."""
        for name in (*PRIMARY_DIMENSIONS, *SECONDARY_DIMENSIONS):
            low, high = _DELTA_BOUNDS.get(name, (-1.0, 1.0))
            object.__setattr__(self, name, clamp(float(getattr(self, name)), low, high))

    @classmethod
    def zero(cls) -> EmotionalDelta:
        """Return the no-motion delta (every dimension exactly ``0.0``)."""
        return cls()

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> EmotionalDelta:
        """Build a delta from a ``{dimension: change}`` mapping."""
        return cls(
            **{
                name: float(data.get(name, 0.0))
                for name in (*PRIMARY_DIMENSIONS, *SECONDARY_DIMENSIONS)
            }
        )

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable representation in canonical field order."""
        return {
            name: getattr(self, name)
            for name in (*PRIMARY_DIMENSIONS, *SECONDARY_DIMENSIONS)
        }


@dataclass(frozen=True)
class EmotionalMomentum:
    """Direction and rate of emotional change across the window.

    Attributes
    ----------
    delta:
        Per-dimension **signed** change from the oldest to the newest retained
        state, as an :class:`EmotionalDelta`. Negative values mean the dimension
        decreased.
    valence_slope:
        Signed average per-step change in valence, naturally bounded by
        ``[-2, +2]``; positive ⇒ brightening, negative ⇒ darkening.
    magnitude:
        The Euclidean norm of the delta — how much affect moved overall,
        regardless of direction.
    """

    delta: EmotionalDelta
    valence_slope: float
    magnitude: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "delta": self.delta.to_dict(),
            "valence_slope": self.valence_slope,
            "magnitude": self.magnitude,
        }


class EmotionMemory:
    """A bounded rolling window of emotional states with trend/momentum read-outs.

    The window is a fixed-capacity deque: pushing beyond ``window`` evicts the
    oldest state in O(1). All derivations are pure functions of the current
    window contents.
    """

    def __init__(
        self, window: int = DEFAULT_WINDOW, decay: float = _DEFAULT_DECAY
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        self._decay = clamp(decay, 0.0, 1.0)
        self._states: Deque[EmotionalState] = deque(maxlen=window)
        self._counter = 0

    # -- Introspection ------------------------------------------------------ #
    @property
    def window(self) -> int:
        """Return the configured maximum window size."""
        return self._window

    @property
    def is_empty(self) -> bool:
        """Return ``True`` if no state has been recorded yet."""
        return not self._states

    def states(self) -> tuple[EmotionalState, ...]:
        """Return the retained states, oldest first."""
        return tuple(self._states)

    def latest(self) -> EmotionalVector | None:
        """Return the most recently pushed vector, or ``None`` if empty."""
        return self._states[-1].vector if self._states else None

    # -- Mutation ----------------------------------------------------------- #
    def push(self, vector: EmotionalVector) -> EmotionalState:
        """Append ``vector`` to the window and return its recorded state."""
        state = EmotionalState(index=self._counter, vector=vector)
        self._counter += 1
        self._states.append(state)
        return state

    def reset(self) -> None:
        """Clear the window and the sequence counter (fresh conversation)."""
        self._states.clear()
        self._counter = 0

    # -- Derivations -------------------------------------------------------- #
    def trend(self) -> EmotionalVector:
        """Return the recency-weighted mean vector of the window.

        The newest state carries weight ``decay**0 = 1``; each older state is
        discounted by an additional factor of ``decay``. With ``decay == 1`` this
        reduces to a plain arithmetic mean. Returns the neutral vector when the
        window is empty.
        """
        if not self._states:
            return EmotionalVector.neutral()

        ordered = list(self._states)  # oldest -> newest
        # Age 0 is the newest state; larger age = older.
        accum: dict[str, float] = {
            name: 0.0
            for name in (*PRIMARY_DIMENSIONS, *SECONDARY_DIMENSIONS)
        }
        total_weight = 0.0
        newest_index = len(ordered) - 1
        for position, state in enumerate(ordered):
            age = newest_index - position
            weight = self._decay ** age
            total_weight += weight
            data = state.vector.to_dict()
            for name, value in data.items():
                accum[name] += value * weight

        if total_weight == 0.0:
            return EmotionalVector.neutral()
        return EmotionalVector.from_dict(
            {name: value / total_weight for name, value in accum.items()}
        )

    def momentum(self) -> EmotionalMomentum:
        """Return the emotional momentum across the current window.

        With fewer than two states there is no motion, so a zero momentum is
        returned. Otherwise the delta is ``newest - oldest`` per dimension, the
        valence slope is that valence delta averaged over the number of steps,
        and the magnitude is the Euclidean norm of the full delta.

        Deltas are signed: a mood collapsing from ``joy=0.9`` to ``joy=0.1``
        reports ``delta.joy == -0.8``.

        Examples
        --------
        >>> mem = EmotionMemory()
        >>> _ = mem.push(EmotionalVector(joy=0.9))
        >>> _ = mem.push(EmotionalVector(joy=0.1))
        >>> round(mem.momentum().delta.joy, 4)
        -0.8
        """
        if len(self._states) < 2:
            return EmotionalMomentum(
                delta=EmotionalDelta.zero(), valence_slope=0.0, magnitude=0.0
            )

        oldest = self._states[0].vector
        newest = self._states[-1].vector
        steps = len(self._states) - 1

        deltas: dict[str, float] = {}
        squared = 0.0
        for name in (*PRIMARY_DIMENSIONS, *SECONDARY_DIMENSIONS):
            change = getattr(newest, name) - getattr(oldest, name)
            deltas[name] = change
            squared += change * change

        valence_slope = (newest.valence - oldest.valence) / steps
        return EmotionalMomentum(
            delta=EmotionalDelta.from_dict(deltas),
            valence_slope=valence_slope,
            magnitude=squared ** 0.5,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot of the memory and its read-outs."""
        return {
            "window": self._window,
            "decay": self._decay,
            "states": [s.to_dict() for s in self._states],
            "trend": self.trend().to_dict(),
            "momentum": self.momentum().to_dict(),
        }
