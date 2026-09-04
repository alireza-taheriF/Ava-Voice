"""Emotional vector schema (Phase 1B).

This module defines the **deterministic contract** of the Persian Adaptive
Emotional Prosody Engine: a single, serializable :class:`EmotionalVector` that
every downstream stage (prosody planning, TTS conditioning, analytics) consumes.

The schema is intentionally *engine-agnostic* — it carries affect, not acoustic
parameters — mirroring the design of :mod:`ava_voice.linguistics` where the
linguistic layer emits *candidates and positions*, never final durations.

Two families of dimensions
--------------------------
**Primary (categorical) dimensions** — the discrete affective channels, each a
saturating activation in ``[0, 1]``:

* ``joy`` — positive, uplifted affect.
* ``sadness`` — negative, low-energy affect.
* ``anger`` — negative, high-energy, dominant affect.
* ``fear`` — negative, high-energy, submissive affect.
* ``calm`` — low-arousal, settled affect.
* ``humor`` — playful / comedic affect (distinct from plain joy).
* ``formality`` — a *register* channel (formal ↔ casual), modelled alongside
  emotion because Persian prosody is strongly register-sensitive (e.g. رسمی vs
  محاوره‌ای speech).

**Secondary (dimensional) coordinates** — a continuous affect space derived from
the primaries, following the well-established VAD (Valence–Arousal–Dominance)
model, plus a scalar intensity:

* ``valence`` — pleasantness, in ``[-1, +1]``.
* ``arousal`` — activation / energy, in ``[0, 1]``.
* ``dominance`` — sense of control, in ``[0, 1]``.
* ``intensity`` — overall expressive magnitude, in ``[0, 1]``.

Design guarantees
-----------------
* **Deterministic** — no randomness, no clock, no I/O. Identical inputs always
  produce identical vectors.
* **Bounded** — every field is clamped to its documented range at construction,
  so the vector is always valid regardless of caller arithmetic.
* **Serializable** — :meth:`EmotionalVector.to_dict` / :meth:`from_dict` provide
  a stable, JSON-friendly round-trip suitable for snapshot testing and for
  embedding into the linguistic :class:`~ava_voice.linguistics.token_graph.TokenGraph`.
* **Immutable** — the dataclass is frozen; combinators return *new* vectors,
  which keeps the fusion pipeline auditable (patent-grade traceability).

Examples
--------
>>> v = EmotionalVector(joy=0.9, valence=0.8, arousal=0.7, intensity=0.85)
>>> v.dominant_emotion
<PrimaryEmotion.JOY: 'joy'>
>>> round(EmotionalVector.neutral().valence, 3)
0.0
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Final, Mapping


class PrimaryEmotion(str, enum.Enum):
    """The closed set of primary (categorical) affective channels.

    String-valued so labels serialize cleanly into JSON and the token graph.
    The declaration order is the canonical iteration order used everywhere in
    the engine (keeps scoring, fusion and serialization deterministic).
    """

    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    CALM = "calm"
    HUMOR = "humor"
    FORMALITY = "formality"


#: Canonical, ordered tuple of primary dimension names. Any code that iterates
#: over primaries MUST use this ordering to preserve determinism.
PRIMARY_DIMENSIONS: Final[tuple[str, ...]] = tuple(e.value for e in PrimaryEmotion)

#: Canonical, ordered tuple of secondary (dimensional) coordinate names.
SECONDARY_DIMENSIONS: Final[tuple[str, ...]] = (
    "valence",
    "arousal",
    "dominance",
    "intensity",
)

# Inclusive numeric bounds per field. Centralized so clamping, validation and
# documentation never drift apart.
_UNIT_RANGE: Final[tuple[float, float]] = (0.0, 1.0)
_SIGNED_RANGE: Final[tuple[float, float]] = (-1.0, 1.0)
_FIELD_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "joy": _UNIT_RANGE,
    "sadness": _UNIT_RANGE,
    "anger": _UNIT_RANGE,
    "fear": _UNIT_RANGE,
    "calm": _UNIT_RANGE,
    "humor": _UNIT_RANGE,
    "formality": _UNIT_RANGE,
    "valence": _SIGNED_RANGE,
    "arousal": _UNIT_RANGE,
    "dominance": _UNIT_RANGE,
    "intensity": _UNIT_RANGE,
}


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive ``[low, high]`` interval.

    A tiny, dependency-free helper reused across the engine so bounding logic is
    defined in exactly one place.
    """
    if value < low:
        return low
    if value > high:
        return high
    return value


@dataclass(frozen=True)
class EmotionalVector:
    """A bounded, serializable snapshot of emotional state.

    All fields are clamped to their documented ranges at construction time (via
    :meth:`__post_init__`), so an ``EmotionalVector`` is *always* valid — callers
    may perform unbounded arithmetic and wrap the result in a new vector without
    having to clamp manually.

    Parameters
    ----------
    joy, sadness, anger, fear, calm, humor, formality:
        Primary channel activations, each clamped to ``[0, 1]``.
    valence:
        Pleasantness in ``[-1, +1]``.
    arousal, dominance, intensity:
        Activation, control and overall magnitude, each in ``[0, 1]``.
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
    dominance: float = 0.5
    intensity: float = 0.0

    # -- Validation --------------------------------------------------------- #
    def __post_init__(self) -> None:
        """Clamp every field into its declared range.

        The dataclass is frozen, so we bypass the immutability guard with
        ``object.__setattr__`` — the canonical, documented pattern for
        normalizing frozen dataclasses.
        """
        for name, (low, high) in _FIELD_BOUNDS.items():
            object.__setattr__(self, name, clamp(float(getattr(self, name)), low, high))

    # -- Accessors ---------------------------------------------------------- #
    @property
    def primary_scores(self) -> dict[str, float]:
        """Return the primary channels as an ordered ``{name: value}`` mapping."""
        return {name: getattr(self, name) for name in PRIMARY_DIMENSIONS}

    @property
    def secondary_scores(self) -> dict[str, float]:
        """Return the secondary coordinates as an ordered mapping."""
        return {name: getattr(self, name) for name in SECONDARY_DIMENSIONS}

    @property
    def dominant_emotion(self) -> PrimaryEmotion | None:
        """Return the strongest primary channel, or ``None`` if all are zero.

        Ties are broken deterministically by :data:`PRIMARY_DIMENSIONS` order
        (the first-declared channel wins) so the result never depends on dict
        or set iteration quirks.
        """
        best_name: str | None = None
        best_value = 0.0
        for name in PRIMARY_DIMENSIONS:
            value = getattr(self, name)
            if value > best_value:
                best_value = value
                best_name = name
        return PrimaryEmotion(best_name) if best_name is not None else None

    @property
    def is_neutral(self) -> bool:
        """Return ``True`` if no primary channel is active and affect is flat."""
        return (
            all(getattr(self, name) == 0.0 for name in PRIMARY_DIMENSIONS)
            and self.valence == 0.0
            and self.arousal == 0.0
            and self.intensity == 0.0
        )

    # -- Combinators (return new vectors; never mutate) --------------------- #
    def blend(self, other: "EmotionalVector", weight: float) -> "EmotionalVector":
        """Linearly interpolate towards ``other`` by ``weight``.

        ``weight = 0`` returns ``self``; ``weight = 1`` returns ``other``. The
        weight is clamped to ``[0, 1]`` so blending is always a convex mix and
        the result stays in-range.
        """
        w = clamp(weight, 0.0, 1.0)
        return self._map_with(
            other, lambda a, b: a * (1.0 - w) + b * w
        )

    def scaled(self, factor: float) -> "EmotionalVector":
        """Return a copy with every field multiplied by ``factor`` (then clamped).

        Useful for attenuating a whole vector by an intensity or decay term.
        """
        return EmotionalVector(
            **{
                name: getattr(self, name) * factor
                for name in (*PRIMARY_DIMENSIONS, *SECONDARY_DIMENSIONS)
            }
        )

    def with_values(self, **overrides: float) -> "EmotionalVector":
        """Return a copy with the given fields replaced (and re-clamped)."""
        data = {**self.to_dict(), **overrides}
        return EmotionalVector(**data)

    def distance(self, other: "EmotionalVector") -> float:
        """Return the Euclidean distance to ``other`` over *all* dimensions.

        Deterministic scalar used by the contextual analyzer to detect emotional
        shifts. It is not normalized (callers compare against fixed thresholds).
        """
        total = 0.0
        for name in (*PRIMARY_DIMENSIONS, *SECONDARY_DIMENSIONS):
            delta = getattr(self, name) - getattr(other, name)
            total += delta * delta
        return total ** 0.5

    def _map_with(
        self,
        other: "EmotionalVector",
        op: "Any",
    ) -> "EmotionalVector":
        """Apply a binary ``op(self_field, other_field)`` across every field."""
        return EmotionalVector(
            **{
                name: op(getattr(self, name), getattr(other, name))
                for name in (*PRIMARY_DIMENSIONS, *SECONDARY_DIMENSIONS)
            }
        )

    # -- Serialization ------------------------------------------------------ #
    def to_dict(self) -> dict[str, float]:
        """Return a flat, JSON-serializable ``{dimension: value}`` mapping.

        Key order follows ``PRIMARY_DIMENSIONS`` then ``SECONDARY_DIMENSIONS``,
        making the output stable and snapshot-comparable.
        """
        return {
            name: getattr(self, name)
            for name in (*PRIMARY_DIMENSIONS, *SECONDARY_DIMENSIONS)
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> "EmotionalVector":
        """Rebuild a vector from a mapping produced by :meth:`to_dict`.

        Unknown keys are ignored and missing keys fall back to field defaults,
        so the round-trip is forgiving to schema evolution.
        """
        valid = set(PRIMARY_DIMENSIONS) | set(SECONDARY_DIMENSIONS)
        return cls(**{k: float(v) for k, v in data.items() if k in valid})

    @classmethod
    def neutral(cls) -> "EmotionalVector":
        """Return the canonical neutral vector (flat affect, mid dominance)."""
        return cls()


#: Shared, immutable neutral vector for allocation-free reuse.
NEUTRAL: Final[EmotionalVector] = EmotionalVector.neutral()
