"""Ezafe (اضافه) acoustic modulator — the Phase 0 embodiment of the invention.

A lexical final ``e`` (``خانه`` → *xane*) and the ezafe linker (``خانه بزرگ``
→ *xane-ye*, ``کتابِ علی`` → *ketâb-e*) are the same IPA vowel on paper and
radically different events in speech. This module is the only place that
**branches on the linker as a first-class phoneme** and sets its duration,
local F0 and local energy from the talker's **register and affect**.

Closed-form mapping (deterministic, no ML)
------------------------------------------
Let ``f`` be formality, ``v`` valence, ``i`` intensity, ``s`` sadness, all
already bounded by :class:`~ava_voice.emotion.emotion_schema.EmotionalVector`.

* **Duration factor**

  .. math::

     \\alpha = (1 - 0.28 f)\\,(1 + 0.30\\, i \\max(0, v))\\,(1 + 0.12 s)
              \\cdot [1.08]_{\\text{ye}}

  Formal speech *clips* the linker; warm high-valence speech *lengthens* it;
  sadness adds a small linger. The ``ye`` allomorph is slightly longer
  because of the on-glide. The result is clamped to ``[0.55, 1.65]``.

* **Local F0 offset (semitones)**

  .. math::  \\Delta F_0 = 1.10\\, i \\max(0, v) - 0.90 f - 0.25 s

  Warmth raises the linker; formality flattens it; sadness drops it a little.

* **Local energy gain** (added on the energy curve, then clamped to ``[0, 1]``)

  .. math::  g = 0.10\\, i\\,(1 - 0.65 f)

Register label (for the audit record, not a separate computation)
-----------------------------------------------------------------
* ``clipped`` — ``f ≥ 0.55``
* ``warm`` — ``v > 0.20`` and ``i ≥ 0.35`` and ``f < 0.40``
* ``neutral`` — otherwise

The linker is **never** expressively drawn out by
:mod:`~ava_voice.acoustics.elongation_controller`; grammatical length lives
here only. All outputs are pure functions of ``(units, durations, vector)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

from ava_voice.acoustics.acoustic_schema import Curve, PhonemeUnit
from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import EmotionalVector, clamp
from ava_voice.linguistics.ezafe import classify_allomorph

_logger = get_logger("acoustics.ezafe")

# Duration coefficients — see the module docstring. Named and frozen so a
# later calibration pass can replace them from a versioned JSON file without
# changing the functional form of the claim.
_K_FORMALITY_CLIP: Final[float] = 0.28
_K_WARM_STRETCH: Final[float] = 0.30
_K_SAD_LINGER: Final[float] = 0.12
_YE_EXTRA: Final[float] = 1.08
_FACTOR_MIN: Final[float] = 0.55
_FACTOR_MAX: Final[float] = 1.65

_K_PITCH_WARM: Final[float] = 1.10
_K_PITCH_FORMAL: Final[float] = 0.90
_K_PITCH_SAD: Final[float] = 0.25

_K_ENERGY: Final[float] = 0.10
_K_ENERGY_FORMAL: Final[float] = 0.65

_CLIPPED_FORMALITY: Final[float] = 0.55
_WARM_VALENCE: Final[float] = 0.20
_WARM_INTENSITY: Final[float] = 0.35
_WARM_MAX_FORMALITY: Final[float] = 0.40


@dataclass(frozen=True)
class EzafeDecision:
    """Per-linker control values before they hit the timeline.

    Attributes
    ----------
    phoneme_index:
        Global phoneme index of the tagged linker.
    token_id:
        Owning ``TOKEN`` id.
    symbol:
        Linker symbol (almost always ``e``).
    allomorph:
        ``"e"`` or ``"ye"``.
    duration_factor:
        Multiplier applied to the linker's base duration.
    pitch_offset_st:
        Local F0 offset in semitones, added only on the linker span.
    energy_gain:
        Local additive energy, added only on the linker span.
    register:
        ``clipped``, ``warm`` or ``neutral``.
    """

    phoneme_index: int
    token_id: str | None
    symbol: str
    allomorph: str
    duration_factor: float
    pitch_offset_st: float
    energy_gain: float
    register: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "phoneme_index": self.phoneme_index,
            "token_id": self.token_id,
            "symbol": self.symbol,
            "allomorph": self.allomorph,
            "duration_factor": self.duration_factor,
            "pitch_offset_st": self.pitch_offset_st,
            "energy_gain": self.energy_gain,
            "register": self.register,
        }


@dataclass(frozen=True)
class EzafePlan:
    """All linker decisions for one utterance."""

    decisions: tuple[EzafeDecision, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {"decisions": [d.to_dict() for d in self.decisions]}

    def by_index(self) -> dict[int, EzafeDecision]:
        """Return decisions keyed by global phoneme index."""
        return {d.phoneme_index: d for d in self.decisions}


class EzafeModulator:
    """Deterministic register-conditioned modulator of the ezafe linker."""

    def plan(
        self, units: Sequence[PhonemeUnit], vector: EmotionalVector
    ) -> EzafePlan:
        """Return an :class:`EzafePlan` for every ``is_ezafe`` unit."""
        symbols = [u.symbol for u in units]
        decisions: list[EzafeDecision] = []
        for position, unit in enumerate(units):
            if not unit.is_ezafe:
                continue
            allomorph = classify_allomorph(symbols, position)
            factor = self._duration_factor(vector, allomorph)
            decisions.append(
                EzafeDecision(
                    phoneme_index=unit.index,
                    token_id=unit.token_id,
                    symbol=unit.symbol,
                    allomorph=allomorph,
                    duration_factor=factor,
                    pitch_offset_st=round(self._pitch_offset(vector), 4),
                    energy_gain=round(self._energy_gain(vector), 4),
                    register=self._register(vector),
                )
            )
        return EzafePlan(decisions=tuple(decisions))

    def apply_durations(
        self,
        units: Sequence[PhonemeUnit],
        base_durations: Sequence[float],
        plan: EzafePlan,
    ) -> tuple[float, ...]:
        """Return durations with the linker factor applied (others unchanged)."""
        by_index = plan.by_index()
        adjusted: list[float] = []
        for unit, base in zip(units, base_durations):
            decision = by_index.get(unit.index)
            if decision is None:
                adjusted.append(base)
            else:
                adjusted.append(base * decision.duration_factor)
        return tuple(adjusted)

    @staticmethod
    def overlay_pitch(curve: Curve, spans: Sequence[tuple[float, float, float]]) -> Curve:
        """Add a semitone offset on each ``(start_ms, end_ms, offset)`` span."""
        return _overlay_curve(curve, spans, clamp_low=None, clamp_high=None)

    @staticmethod
    def overlay_energy(
        curve: Curve, spans: Sequence[tuple[float, float, float]]
    ) -> Curve:
        """Add energy gain on each span and clamp the result to ``[0, 1]``."""
        return _overlay_curve(curve, spans, clamp_low=0.0, clamp_high=1.0)

    @staticmethod
    def _duration_factor(vector: EmotionalVector, allomorph: str) -> float:
        """Closed-form linker duration multiplier."""
        warm = max(0.0, vector.valence)
        factor = (
            (1.0 - _K_FORMALITY_CLIP * vector.formality)
            * (1.0 + _K_WARM_STRETCH * vector.intensity * warm)
            * (1.0 + _K_SAD_LINGER * vector.sadness)
        )
        if allomorph == "ye":
            factor *= _YE_EXTRA
        return clamp(factor, _FACTOR_MIN, _FACTOR_MAX)

    @staticmethod
    def _pitch_offset(vector: EmotionalVector) -> float:
        """Local F0 offset in semitones."""
        warm = max(0.0, vector.valence)
        return (
            _K_PITCH_WARM * vector.intensity * warm
            - _K_PITCH_FORMAL * vector.formality
            - _K_PITCH_SAD * vector.sadness
        )

    @staticmethod
    def _energy_gain(vector: EmotionalVector) -> float:
        """Local additive energy gain."""
        return _K_ENERGY * vector.intensity * (1.0 - _K_ENERGY_FORMAL * vector.formality)

    @staticmethod
    def _register(vector: EmotionalVector) -> str:
        """Coarse register label for the audit record."""
        if vector.formality >= _CLIPPED_FORMALITY:
            return "clipped"
        if (
            vector.valence > _WARM_VALENCE
            and vector.intensity >= _WARM_INTENSITY
            and vector.formality < _WARM_MAX_FORMALITY
        ):
            return "warm"
        return "neutral"


def _overlay_curve(
    curve: Curve,
    spans: Sequence[tuple[float, float, float]],
    *,
    clamp_low: float | None,
    clamp_high: float | None,
) -> Curve:
    """Return a copy of ``curve`` with additive offsets on time spans."""
    if not spans or not curve.values:
        return curve
    values = list(curve.values)
    hop = curve.hop_ms
    for frame, value in enumerate(values):
        time_ms = frame * hop
        total = value
        for start_ms, end_ms, offset in spans:
            if start_ms <= time_ms < end_ms:
                total += offset
        if clamp_low is not None or clamp_high is not None:
            low = clamp_low if clamp_low is not None else total
            high = clamp_high if clamp_high is not None else total
            total = clamp(total, low, high)
        values[frame] = round(total, 4)
    return Curve(unit=curve.unit, hop_ms=curve.hop_ms, values=tuple(values))


_default_modulator: Final[EzafeModulator] = EzafeModulator()


def plan_ezafe(
    units: Sequence[PhonemeUnit], vector: EmotionalVector
) -> EzafePlan:
    """Plan ezafe modulation using the module-level default modulator."""
    return _default_modulator.plan(units, vector)
