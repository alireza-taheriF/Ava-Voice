"""Prosody profile schema (Phase 1C).

This module defines the **deterministic contract** of the Persian Adaptive
Emotional Prosody Engine's output: a single, immutable, serializable
:class:`ProsodyProfile` describing *how* a sentence should be spoken, together
with the small closed vocabularies (contour shapes) it references.

Where :mod:`ava_voice.emotion` answers *what is felt*, this layer answers *how
that feeling is realized in speech*. The profile is deliberately **engine
-agnostic**: every field is a normalized or relative control value with a
documented range and unit, so it can drive any downstream acoustic model (neural
vocoder, formant synth, unit selection) without leaking model-specific
parameters into the linguistic/prosodic reasoning.

Fields
------
``pitch_base``
    Relative pitch height as a **semitone offset** from the speaker's baseline
    F0, in ``[-12, +12]`` (0 = neutral baseline; +12 = one octave up).
``pitch_variance``
    Normalized pitch variability / expressiveness in ``[0, 1]`` (0 = monotone,
    1 = highly modulated).
``speaking_rate``
    Tempo as a **multiplier** of the baseline rate in ``[0.5, 2.0]`` (1.0 =
    normal; <1 slower; >1 faster).
``energy_level``
    Normalized vocal energy / loudness in ``[0, 1]``.
``pause_density``
    Normalized measure of how *frequently* the sentence pauses, in ``[0, 1]``
    (roughly boundary-pauses per token, plus an emotional offset).
``pause_duration``
    A **multiplier** applied to nominal boundary-pause durations in
    ``[0.25, 3.0]`` (1.0 = nominal; >1 = longer, more deliberate pauses).
``emphasis_strength``
    Normalized overall emphasis intensity in ``[0, 1]``.
``contour_shape``
    The sentence-level intonation :class:`ContourShape`.
``elongation_factor``
    A **multiplier** for expressive vowel lengthening in ``[1.0, 2.5]`` (1.0 =
    no lengthening).

Design guarantees
-----------------
* **Deterministic** — no randomness, clock, or I/O anywhere in the layer.
* **Immutable & bounded** — the dataclass is frozen and every numeric field is
  clamped to its documented range at construction, so a profile is always valid.
* **Serializable** — :meth:`ProsodyProfile.to_dict` / :meth:`from_dict` provide a
  stable JSON round-trip suitable for snapshot testing and for embedding into
  the linguistic :class:`~ava_voice.linguistics.token_graph.TokenGraph`.

Examples
--------
>>> p = ProsodyProfile(pitch_base=4.0, speaking_rate=1.2, contour_shape=ContourShape.RISING)
>>> p.contour_shape is ContourShape.RISING
True
>>> ProsodyProfile.from_dict(p.to_dict()) == p
True
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Final, Mapping

from ava_voice.emotion.emotion_schema import PRIMARY_DIMENSIONS, clamp


class ContourShape(str, enum.Enum):
    """Closed vocabulary of sentence-level intonation contours.

    String-valued so contours serialize cleanly into JSON and the token graph.

    * ``RISING`` — overall upward F0 trend (e.g. yes/no questions, excitement).
    * ``FALLING`` — overall downward trend (declaratives, resolution, sadness).
    * ``OSCILLATING`` — repeated up/down movement (playfulness, wavering fear).
    * ``FLAT`` — compressed, near-level contour (calm, formal, detached).
    * ``BROKEN`` — abrupt, discontinuous jumps (anger, agitation).
    """

    RISING = "rising"
    FALLING = "falling"
    OSCILLATING = "oscillating"
    FLAT = "flat"
    BROKEN = "broken"


# Inclusive numeric bounds per profile field. Centralized so clamping,
# validation and documentation can never drift apart (patent-grade auditability).
_FIELD_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "pitch_base": (-12.0, 12.0),
    "pitch_variance": (0.0, 1.0),
    "speaking_rate": (0.5, 2.0),
    "energy_level": (0.0, 1.0),
    "pause_density": (0.0, 1.0),
    "pause_duration": (0.25, 3.0),
    "emphasis_strength": (0.0, 1.0),
    "elongation_factor": (1.0, 2.5),
}

#: Canonical, ordered tuple of profile field names (numeric + the enum).
PROFILE_FIELDS: Final[tuple[str, ...]] = (
    "pitch_base",
    "pitch_variance",
    "speaking_rate",
    "energy_level",
    "pause_density",
    "pause_duration",
    "emphasis_strength",
    "contour_shape",
    "elongation_factor",
)


@dataclass(frozen=True)
class ProsodyProfile:
    """Immutable, bounded, serializable description of spoken prosody.

    All numeric fields are clamped into their documented ranges by
    :meth:`__post_init__`; callers may therefore perform unbounded arithmetic and
    wrap the result in a new profile without manual clamping.
    """

    pitch_base: float = 0.0
    pitch_variance: float = 0.2
    speaking_rate: float = 1.0
    energy_level: float = 0.5
    pause_density: float = 0.2
    pause_duration: float = 1.0
    emphasis_strength: float = 0.0
    contour_shape: ContourShape = ContourShape.FALLING
    elongation_factor: float = 1.0

    def __post_init__(self) -> None:
        """Clamp every numeric field and coerce ``contour_shape`` to the enum."""
        for name, (low, high) in _FIELD_BOUNDS.items():
            object.__setattr__(
                self, name, clamp(float(getattr(self, name)), low, high)
            )
        # Accept a raw string for ``contour_shape`` (e.g. after deserialization).
        if not isinstance(self.contour_shape, ContourShape):
            object.__setattr__(
                self, "contour_shape", ContourShape(str(self.contour_shape))
            )

    # -- Combinators (return new profiles; never mutate) -------------------- #
    def with_values(self, **overrides: Any) -> "ProsodyProfile":
        """Return a copy with the given fields replaced (and re-clamped)."""
        data = self.to_dict()
        data.update(overrides)
        return ProsodyProfile.from_dict(data)

    # -- Serialization ------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-serializable ``{field: value}`` mapping.

        Key order follows :data:`PROFILE_FIELDS`, making the output stable and
        snapshot-comparable. ``contour_shape`` is emitted as its string value.
        """
        result: dict[str, Any] = {}
        for name in PROFILE_FIELDS:
            value = getattr(self, name)
            result[name] = value.value if isinstance(value, ContourShape) else value
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProsodyProfile":
        """Rebuild a profile from a mapping produced by :meth:`to_dict`.

        Unknown keys are ignored and missing keys fall back to field defaults, so
        the round-trip is forgiving to schema evolution.
        """
        valid = set(PROFILE_FIELDS)
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in valid:
                continue
            if key == "contour_shape":
                kwargs[key] = ContourShape(str(value))
            else:
                kwargs[key] = float(value)
        return cls(**kwargs)

    @classmethod
    def neutral(cls) -> "ProsodyProfile":
        """Return the canonical neutral profile (baseline, level speech)."""
        return cls()


#: Shared, immutable neutral profile for allocation-free reuse.
NEUTRAL_PROFILE: Final[ProsodyProfile] = ProsodyProfile.neutral()


def blend_by_emotion(
    scores: Mapping[str, float],
    anchors: Mapping[str, float],
    default: float = 0.0,
) -> float:
    """Return the score-weighted average of per-emotion ``anchors``.

    This is the shared kernel of every categorical→prosody mapper: given the
    primary-emotion ``scores`` and a table of per-emotion anchor values, it
    computes

    .. math::  \\frac{\\sum_e s_e \\cdot \\text{anchor}_e}{\\sum_e s_e}

    iterating in the canonical :data:`~ava_voice.emotion.emotion_schema.PRIMARY_DIMENSIONS`
    order so floating-point accumulation is deterministic. When no emotion is
    active (all scores ≤ 0) the ``default`` resting value is returned.

    Parameters
    ----------
    scores:
        ``{emotion: score}`` mapping; non-positive or missing entries are
        ignored. Scores need not be normalized — only relative magnitude matters.
    anchors:
        ``{emotion: anchor_value}`` mapping of canonical per-emotion contributions.
    default:
        Value returned when there is no active emotion mass.
    """
    total = 0.0
    accumulated = 0.0
    for emotion in PRIMARY_DIMENSIONS:
        score = scores.get(emotion, 0.0)
        if score <= 0.0:
            continue
        anchor = anchors.get(emotion)
        if anchor is None:
            continue
        total += score
        accumulated += score * anchor
    if total == 0.0:
        return default
    return accumulated / total
