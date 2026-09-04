"""Frame-level energy (amplitude) envelope generation (Phase 2A).

This module renders the sentence's loudness/effort into a concrete, frame-level
**energy curve** in ``[0, 1]``, sampled every ``hop_ms`` milliseconds. It shapes
a baseline level (from the prosody profile's ``energy_level``) and adds
prominence **peaks** at emphasized tokens.

How emotion reaches this stage
------------------------------
The acoustic layer's only affective input is the
:class:`~ava_voice.prosody.prosody_schema.ProsodyProfile`, but that profile
already encodes the emotion via its ``contour_shape``, ``energy_level`` and
``emphasis_strength``. This module reads those to satisfy the specification's
emotional rules:

* **anger** (``BROKEN`` contour) → *sharper peaks*: narrow, tall prominence bumps
  and an agitated micro-modulation.
* **calm** (``FLAT`` contour) → *smoother envelope*: wide, low bumps over a
  steady baseline.
* **joy** (``RISING`` contour) → *brighter energy*: an elevated baseline with a
  gentle upward tilt and moderate peaks.
* **fear / humor** (``OSCILLATING``) → a periodic tremor riding the baseline.
* **sadness** (``FALLING``) → a gently declining, subdued baseline.

Peaks are computed as closed-form Gaussian bumps around each emphasis centre; the
whole curve is deterministic (no randomness, no ML) and clamped to ``[0, 1]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Sequence

from ava_voice.acoustics.acoustic_schema import DEFAULT_HOP_MS, Curve
from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import clamp
from ava_voice.prosody.prosody_schema import ContourShape, ProsodyProfile

_logger = get_logger("acoustics.energy")


@dataclass(frozen=True)
class _PeakShape:
    """Per-contour peak geometry: bump ``width_ms`` and additive ``gain``."""

    width_ms: float
    gain: float


# Contour → peak geometry. Anger (BROKEN) is narrow+tall; calm (FLAT) wide+low.
_PEAK_SHAPES: Final[dict[ContourShape, _PeakShape]] = {
    ContourShape.BROKEN: _PeakShape(width_ms=40.0, gain=0.50),
    ContourShape.FLAT: _PeakShape(width_ms=220.0, gain=0.12),
    ContourShape.OSCILLATING: _PeakShape(width_ms=70.0, gain=0.35),
    ContourShape.RISING: _PeakShape(width_ms=90.0, gain=0.35),
    ContourShape.FALLING: _PeakShape(width_ms=120.0, gain=0.25),
}
_DEFAULT_PEAK: Final[_PeakShape] = _PeakShape(width_ms=90.0, gain=0.30)

# Baseline tilt (fraction of level added/removed across the sentence) per contour.
_BASELINE_TILT: Final[dict[ContourShape, float]] = {
    ContourShape.RISING: 0.18,    # brighten toward the end (joy)
    ContourShape.FALLING: -0.18,  # subside toward the end (sadness)
}
# Micro-modulation depth (fraction of level) for agitated/tremulous contours.
_MODULATION_DEPTH: Final[dict[ContourShape, float]] = {
    ContourShape.OSCILLATING: 0.10,
    ContourShape.BROKEN: 0.08,
}
_MODULATION_CYCLES: Final[float] = 6.0


class EnergyEnvelopeGenerator:
    """Deterministic renderer of a frame-level energy curve."""

    def build(
        self,
        profile: ProsodyProfile,
        total_duration_ms: float,
        hop_ms: float = DEFAULT_HOP_MS,
        emphasis_centers: Sequence[tuple[float, float]] = (),
    ) -> Curve:
        """Render the energy :class:`Curve` for ``profile`` over the sentence.

        Parameters
        ----------
        profile:
            Supplies ``energy_level``, ``contour_shape`` and ``emphasis_strength``.
        total_duration_ms:
            Sentence duration to span.
        hop_ms:
            Frame step in milliseconds.
        emphasis_centers:
            ``(center_ms, strength)`` pairs marking emphasized tokens; each adds a
            prominence bump whose height scales with ``strength``.
        """
        n_frames = self._frame_count(total_duration_ms, hop_ms)
        if n_frames == 0:
            return Curve(unit="normalized", hop_ms=hop_ms, values=())

        level = clamp(profile.energy_level, 0.0, 1.0)
        shape = profile.contour_shape
        peak_shape = _PEAK_SHAPES.get(shape, _DEFAULT_PEAK)
        tilt = _BASELINE_TILT.get(shape, 0.0)
        modulation = _MODULATION_DEPTH.get(shape, 0.0)
        span = max(1, n_frames - 1)

        values: list[float] = []
        for frame in range(n_frames):
            t = frame / span
            time_ms = frame * hop_ms

            # Baseline with a contour-dependent tilt centered on ``level``.
            value = level * (1.0 + tilt * (t - 0.5) * 2.0)
            # Agitated/tremulous micro-modulation.
            if modulation:
                value += (
                    level
                    * modulation
                    * math.sin(2.0 * math.pi * _MODULATION_CYCLES * t)
                )
            # Additive prominence peaks at emphasis centres.
            for center_ms, strength in emphasis_centers:
                value += self._peak(
                    time_ms, center_ms, strength, peak_shape, profile.emphasis_strength
                )
            values.append(round(clamp(value, 0.0, 1.0), 4))

        return Curve(unit="normalized", hop_ms=hop_ms, values=tuple(values))

    # -- Internals ---------------------------------------------------------- #
    @staticmethod
    def _frame_count(total_duration_ms: float, hop_ms: float) -> int:
        """Return the number of frames spanning ``total_duration_ms``."""
        if total_duration_ms <= 0.0 or hop_ms <= 0.0:
            return 0
        return max(1, int(round(total_duration_ms / hop_ms)))

    @staticmethod
    def _peak(
        time_ms: float,
        center_ms: float,
        strength: float,
        peak_shape: _PeakShape,
        emphasis_strength: float,
    ) -> float:
        """Return the Gaussian bump contribution at ``time_ms``.

        Height scales with both the per-token ``strength`` and the sentence-level
        ``emphasis_strength``; width comes from the contour's peak geometry.
        """
        width = max(peak_shape.width_ms, 1.0)
        height = peak_shape.gain * clamp(strength, 0.0, 1.0)
        height *= 0.5 + 0.5 * clamp(emphasis_strength, 0.0, 1.0)
        delta = (time_ms - center_ms) / width
        return height * math.exp(-0.5 * delta * delta)


_default_generator: Final[EnergyEnvelopeGenerator] = EnergyEnvelopeGenerator()


def build_energy_curve(
    profile: ProsodyProfile,
    total_duration_ms: float,
    hop_ms: float = DEFAULT_HOP_MS,
    emphasis_centers: Sequence[tuple[float, float]] = (),
) -> Curve:
    """Build an energy curve using the module-level default generator."""
    return _default_generator.build(
        profile, total_duration_ms, hop_ms, emphasis_centers
    )
