"""Pause shaping into timing segments (Phase 2A).

The prosody layer decided *how long* pauses should be *relative* to nominal
(via the profile's ``pause_duration`` scale) and the linguistics layer decided
*where* pauses sit (the ``PAUSE`` nodes). This module turns that into concrete
**silence durations in milliseconds**, ready to be placed on the absolute
timeline by the aligner.

Inputs and rules
----------------
* **Punctuation** — each pause carries a
  :class:`~ava_voice.linguistics.punctuation_mapper.ControlSignal` whose *nominal*
  silence length reflects Persian punctuation convention: the Persian comma
  ``،`` is a brief breath-group boundary, the full stop ``.`` a clause/sentence
  boundary, the newline the longest, and ``…`` (تعلیق) a long, suspended silence.
* **Emotional state** — the profile's ``pause_duration`` multiplier (sadness /
  calm lengthen; anger / joy compress) scales every nominal duration uniformly.
* **Sentence boundaries** — terminal signals are categorized as ``"sentence"``
  so downstream stages can, e.g., add extra settling only at true sentence ends.

The output is a pure function of ``(signal, profile)`` — no randomness, no ML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import clamp
from ava_voice.linguistics.punctuation_mapper import ControlSignal
from ava_voice.prosody.prosody_schema import ProsodyProfile

_logger = get_logger("acoustics.pause")

# Nominal silence duration (ms) per control signal, at pause_duration = 1.0.
_NOMINAL_MS: Final[dict[ControlSignal, float]] = {
    ControlSignal.PAUSE_SHORT: 120.0,
    ControlSignal.PAUSE_MEDIUM: 260.0,
    ControlSignal.PAUSE_LONG: 480.0,
    ControlSignal.TRAILING_OFF: 420.0,
    ControlSignal.RISING_PITCH: 200.0,
    ControlSignal.FALLING_PITCH: 260.0,
    ControlSignal.EMPHASIS_BOOST: 180.0,
    ControlSignal.PITCH_LOWER: 90.0,
}
_DEFAULT_NOMINAL_MS: Final[float] = 150.0

# Coarse category per signal (used to tag sentence-final silences).
_CATEGORY: Final[dict[ControlSignal, str]] = {
    ControlSignal.PAUSE_SHORT: "breath",
    ControlSignal.PITCH_LOWER: "breath",
    ControlSignal.PAUSE_MEDIUM: "clause",
    ControlSignal.FALLING_PITCH: "sentence",
    ControlSignal.RISING_PITCH: "sentence",
    ControlSignal.EMPHASIS_BOOST: "sentence",
    ControlSignal.PAUSE_LONG: "sentence",
    ControlSignal.TRAILING_OFF: "sentence",
}
_DEFAULT_CATEGORY: Final[str] = "clause"


@dataclass(frozen=True)
class PauseShape:
    """The shaped duration and category for a single pause.

    Attributes
    ----------
    signal:
        The control-signal string.
    nominal_ms:
        The nominal (unscaled) silence length.
    duration_ms:
        The emotion-scaled silence length.
    category:
        ``"breath"`` / ``"clause"`` / ``"sentence"``.
    """

    signal: str
    nominal_ms: float
    duration_ms: float
    category: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "signal": self.signal,
            "nominal_ms": self.nominal_ms,
            "duration_ms": self.duration_ms,
            "category": self.category,
        }


class PauseShaper:
    """Deterministic shaper of pause durations from signal + prosody profile."""

    def shape(
        self, signal: ControlSignal | None, profile: ProsodyProfile
    ) -> PauseShape:
        """Return the :class:`PauseShape` for a pause ``signal`` under ``profile``.

        The nominal duration is looked up per signal and multiplied by the
        profile's ``pause_duration`` scale (clamped defensively).
        """
        nominal = (
            _NOMINAL_MS.get(signal, _DEFAULT_NOMINAL_MS)
            if signal is not None
            else _DEFAULT_NOMINAL_MS
        )
        scale = clamp(profile.pause_duration, 0.25, 3.0)
        duration = nominal * scale
        category = (
            _CATEGORY.get(signal, _DEFAULT_CATEGORY)
            if signal is not None
            else _DEFAULT_CATEGORY
        )
        signal_name = signal.value if signal is not None else "unknown"
        return PauseShape(
            signal=signal_name,
            nominal_ms=nominal,
            duration_ms=duration,
            category=category,
        )


_default_shaper: Final[PauseShaper] = PauseShaper()


def shape_pause(
    signal: ControlSignal | None, profile: ProsodyProfile
) -> PauseShape:
    """Shape a pause using the module-level default shaper."""
    return _default_shaper.shape(signal, profile)
