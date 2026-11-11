"""Speaker F0 baseline — resolve semitone offsets to hertz (Phase 0).

The prosody and pitch-envelope stages speak in **semitones relative to a
speaker baseline**. Without a baseline those numbers cannot drive a vocoder.
This module is the single conversion:

.. math::  f_{\\mathrm{Hz}} = f_0 \\cdot 2^{s / 12}

``f_0`` defaults to 165 Hz (a mid adult speaking pitch). Callers that know
the speaker (recorded mean F0, or a clone embedding) pass an explicit
:class:`SpeakerBaseline`. Conversion is deterministic and invertible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from ava_voice.acoustics.acoustic_schema import Curve
from ava_voice.emotion.emotion_schema import clamp

#: Default mid-adult speaking F0 (Hz). Not a claimed "typical Persian"
#: value — a documented, replaceable constant.
DEFAULT_SPEAKER_F0_HZ: Final[float] = 165.0

_F0_MIN_HZ: Final[float] = 40.0
_F0_MAX_HZ: Final[float] = 600.0


@dataclass(frozen=True)
class SpeakerBaseline:
    """A speaker's resting F0 in hertz."""

    f0_hz: float = DEFAULT_SPEAKER_F0_HZ

    def __post_init__(self) -> None:
        object.__setattr__(self, "f0_hz", clamp(float(self.f0_hz), _F0_MIN_HZ, _F0_MAX_HZ))

    def semitone_to_hz(self, semitones: float) -> float:
        """Convert a semitone offset from this baseline to hertz."""
        return self.f0_hz * (2.0 ** (semitones / 12.0))

    def hz_to_semitone(self, hz: float) -> float:
        """Convert an absolute F0 in hertz to a semitone offset from baseline."""
        if hz <= 0.0:
            return 0.0
        return 12.0 * math.log2(hz / self.f0_hz)

    def to_hz_curve(self, semitone_curve: Curve) -> Curve:
        """Lift a semitone :class:`Curve` into an absolute-hertz curve."""
        values = tuple(
            round(self.semitone_to_hz(value), 4) for value in semitone_curve.values
        )
        return Curve(unit="hertz", hop_ms=semitone_curve.hop_ms, values=values)

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable representation."""
        return {"f0_hz": self.f0_hz}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "SpeakerBaseline":
        """Rebuild from :meth:`to_dict` output."""
        return cls(f0_hz=float(data.get("f0_hz", DEFAULT_SPEAKER_F0_HZ)))


DEFAULT_SPEAKER: Final[SpeakerBaseline] = SpeakerBaseline()
