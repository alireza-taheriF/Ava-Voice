"""Frame-level pitch (F0) envelope generation (Phase 2A).

This module renders the sentence's melodic *shape* — chosen symbolically by the
prosody layer as a :class:`~ava_voice.prosody.prosody_schema.ContourShape` — into
a concrete, frame-level **F0 curve** sampled every ``hop_ms`` milliseconds.

Inputs
------
* ``pitch_base`` — the sentence's baseline height (semitone offset), the DC level
  the whole curve is shifted onto.
* ``pitch_variance`` — the expressiveness, which sets the excursion **amplitude**
  (monotone speech ⇒ tiny swings; animated speech ⇒ wide swings).
* ``contour_shape`` — the qualitative trajectory to realize.

Contour realizations (deterministic, closed-form)
--------------------------------------------------
Time is normalized to ``t ∈ [0, 1]`` across the sentence; ``A`` is the amplitude
derived from variance:

* ``RISING`` — linear ramp ``A·(2t − 1)`` (low → high).
* ``FALLING`` — linear ramp ``A·(1 − 2t)`` (high → low).
* ``OSCILLATING`` — sinusoid ``A·sin(2π·k·t)`` (``k`` cycles; wavering line).
* ``BROKEN`` — a piecewise-constant step pattern with abrupt jumps (agitation).
* ``FLAT`` — a nearly level line with only a vanishing ripple.

Every value is offset by ``pitch_base`` and returned in **semitones** relative to
the speaker baseline. The generator is fully deterministic: no randomness, no
ML. It uses only closed-form arithmetic (`sin` is deterministic), so identical
inputs yield byte-identical curves.
"""

from __future__ import annotations

import math
from typing import Final

from ava_voice.acoustics.acoustic_schema import DEFAULT_HOP_MS, Curve
from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import clamp
from ava_voice.prosody.prosody_schema import ContourShape, ProsodyProfile

_logger = get_logger("acoustics.pitch")

# Maximum half-excursion (semitones) at full variance.
_MAX_EXCURSION_ST: Final[float] = 6.0
# Number of oscillation cycles across the sentence for the OSCILLATING contour.
_OSCILLATION_CYCLES: Final[float] = 3.0
# Step pattern for the BROKEN contour (normalized, cycled across segments).
_BROKEN_PATTERN: Final[tuple[float, ...]] = (0.2, -0.8, 0.7, -0.6, 0.4)
_BROKEN_SEGMENTS: Final[int] = 5
# Vanishing ripple amplitude (semitones) for the FLAT contour.
_FLAT_RIPPLE_ST: Final[float] = 0.15


class PitchEnvelopeGenerator:
    """Deterministic renderer of a frame-level F0 curve from a prosody profile."""

    def build(
        self,
        profile: ProsodyProfile,
        total_duration_ms: float,
        hop_ms: float = DEFAULT_HOP_MS,
    ) -> Curve:
        """Render the pitch :class:`Curve` for ``profile`` over the sentence span.

        Parameters
        ----------
        profile:
            The prosody profile supplying ``pitch_base``, ``pitch_variance`` and
            ``contour_shape``.
        total_duration_ms:
            The sentence duration to span (from the assembled timeline).
        hop_ms:
            The frame step in milliseconds.
        """
        n_frames = self._frame_count(total_duration_ms, hop_ms)
        amplitude = clamp(profile.pitch_variance, 0.0, 1.0) * _MAX_EXCURSION_ST
        base = profile.pitch_base
        shape = profile.contour_shape

        values: list[float] = []
        for frame in range(n_frames):
            # Normalized position in [0, 1]; single-frame sentences sit at 0.
            t = frame / (n_frames - 1) if n_frames > 1 else 0.0
            values.append(round(base + amplitude * self._shape_value(shape, t), 4))

        return Curve(unit="semitone", hop_ms=hop_ms, values=tuple(values))

    # -- Internals ---------------------------------------------------------- #
    @staticmethod
    def _frame_count(total_duration_ms: float, hop_ms: float) -> int:
        """Return the number of frames spanning ``total_duration_ms``."""
        if total_duration_ms <= 0.0 or hop_ms <= 0.0:
            return 0
        return max(1, int(round(total_duration_ms / hop_ms)))

    @staticmethod
    def _shape_value(shape: ContourShape, t: float) -> float:
        """Return the normalized contour value in ``[-1, 1]`` at position ``t``."""
        if shape is ContourShape.RISING:
            return 2.0 * t - 1.0
        if shape is ContourShape.FALLING:
            return 1.0 - 2.0 * t
        if shape is ContourShape.OSCILLATING:
            return math.sin(2.0 * math.pi * _OSCILLATION_CYCLES * t)
        if shape is ContourShape.BROKEN:
            # Piecewise-constant abrupt steps.
            segment = min(_BROKEN_SEGMENTS - 1, int(t * _BROKEN_SEGMENTS))
            return _BROKEN_PATTERN[segment % len(_BROKEN_PATTERN)]
        # FLAT: a vanishing ripple so the curve is not perfectly dead.
        return (_FLAT_RIPPLE_ST / _MAX_EXCURSION_ST) * math.sin(2.0 * math.pi * t)


_default_generator: Final[PitchEnvelopeGenerator] = PitchEnvelopeGenerator()


def build_pitch_curve(
    profile: ProsodyProfile,
    total_duration_ms: float,
    hop_ms: float = DEFAULT_HOP_MS,
) -> Curve:
    """Build a pitch curve using the module-level default generator."""
    return _default_generator.build(profile, total_duration_ms, hop_ms)
