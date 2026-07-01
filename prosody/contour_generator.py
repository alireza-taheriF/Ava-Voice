"""Sentence-level intonation contour generation (Phase 1C).

The contour is the *melodic shape* of the whole sentence — the gestalt an
utterance's F0 traces from first syllable to last. This module deterministically
selects one of five :class:`~ava_voice.prosody.prosody_schema.ContourShape`
categories and emits a small, normalized trajectory sketch realizing it.

Inputs
------
* **Emotional vector** — the dominant affect and its valence/arousal.
* **Pitch profile** — the emotion-derived pitch tendency and variance (from
  :mod:`~ava_voice.prosody.pitch_mapper`); variance sets the contour amplitude.
* **Terminal signal** — the sentence-final punctuation
  :class:`~ava_voice.linguistics.punctuation_mapper.ControlSignal` (e.g. a rising
  ``؟`` or a trailing ``…``), which anchors the boundary movement.

Decision rules (evaluated in priority order, deterministic)
-----------------------------------------------------------
1. **anger** dominant → ``BROKEN`` (abrupt, discontinuous jumps).
2. **fear** dominant → ``OSCILLATING`` (wavering, tremulous line).
3. **rising terminal** (``؟`` question) → ``RISING`` (Persian yes/no questions
   carry a strong final rise).
4. **calm / low-energy** neutral → ``FLAT`` (compressed, level line).
5. **playful humor** → ``OSCILLATING``; other **positive** affect → ``RISING``.
6. **negative** affect / **trailing-off** terminal → ``FALLING``.
7. default → ``FALLING`` (the unmarked Persian declarative cadence).

The output is a pure function of the inputs — no randomness, no ML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import EmotionalVector, clamp
from ava_voice.linguistics.punctuation_mapper import ControlSignal
from ava_voice.prosody.pitch_mapper import PitchProfile, PitchTendency
from ava_voice.prosody.prosody_schema import ContourShape

_logger = get_logger("prosody.contour")

# Activation at/above which a category is considered "dominant" for the contour.
_DOMINANT_THRESHOLD: Final[float] = 0.40
# |valence| needed to commit to a directional (rising/falling) contour.
_DIRECTIONAL_VALENCE: Final[float] = 0.20
# Below these, an unmarked neutral utterance is treated as flat.
_FLAT_AROUSAL: Final[float] = 0.25
_FLAT_INTENSITY: Final[float] = 0.25

# Normalized 5-point trajectory templates (relative F0 around the pitch base).
# Amplitude is applied on top; ``FLAT`` stays intentionally tiny.
_TEMPLATES: Final[dict[ContourShape, tuple[float, ...]]] = {
    ContourShape.RISING: (-0.6, -0.3, 0.0, 0.4, 0.9),
    ContourShape.FALLING: (0.6, 0.3, 0.0, -0.3, -0.7),
    ContourShape.OSCILLATING: (0.0, 0.7, -0.5, 0.6, -0.3),
    ContourShape.BROKEN: (0.2, -0.8, 0.7, -0.6, 0.4),
    ContourShape.FLAT: (0.04, 0.0, 0.02, -0.02, 0.0),
}


@dataclass(frozen=True)
class ContourResult:
    """The generated sentence contour.

    Attributes
    ----------
    shape:
        The selected :class:`ContourShape`.
    points:
        A normalized trajectory sketch (relative F0 in ``[-1, 1]``), sampled at
        five evenly-spaced positions across the sentence.
    amplitude:
        The excursion scale applied to the template, in ``[0, 1]`` (derived from
        pitch variance).
    terminal:
        The terminal control-signal string that anchored the contour, or ``None``.
    """

    shape: ContourShape
    points: tuple[float, ...] = field(default_factory=tuple)
    amplitude: float = 0.0
    terminal: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "shape": self.shape.value,
            "points": list(self.points),
            "amplitude": self.amplitude,
            "terminal": self.terminal,
        }


class ContourGenerator:
    """Deterministic sentence-level intonation contour selector."""

    def generate(
        self,
        vector: EmotionalVector,
        pitch: PitchProfile | None = None,
        terminal_signal: ControlSignal | None = None,
    ) -> ContourResult:
        """Select a contour shape and realize its trajectory.

        Parameters
        ----------
        vector:
            The sentence emotional vector.
        pitch:
            Optional pitch profile; its ``variance`` sets the contour amplitude
            and its ``tendency`` reinforces the shape choice. Defaults to a
            mid amplitude when omitted.
        terminal_signal:
            The sentence-final punctuation control signal, if any.
        """
        shape = self._select_shape(vector, pitch, terminal_signal)
        variance = pitch.variance if pitch is not None else 0.3
        amplitude = clamp(0.3 + 0.7 * variance, 0.0, 1.0)

        template = _TEMPLATES[shape]
        points = tuple(round(value * amplitude, 4) for value in template)

        return ContourResult(
            shape=shape,
            points=points,
            amplitude=amplitude,
            terminal=terminal_signal.value if terminal_signal is not None else None,
        )

    # -- Shape selection ---------------------------------------------------- #
    @staticmethod
    def _select_shape(
        vector: EmotionalVector,
        pitch: PitchProfile | None,
        terminal_signal: ControlSignal | None,
    ) -> ContourShape:
        """Apply the priority-ordered contour decision rules."""
        # 1. Anger → broken; reinforced by an unstable pitch tendency.
        if vector.anger >= _DOMINANT_THRESHOLD:
            return ContourShape.BROKEN
        # 2. Fear → oscillating (tremulous wavering).
        if vector.fear >= _DOMINANT_THRESHOLD:
            return ContourShape.OSCILLATING
        # 3. Question terminal → rising (Persian yes/no rise).
        if terminal_signal is ControlSignal.RISING_PITCH:
            return ContourShape.RISING
        # 4. Calm / unmarked low-energy neutral → flat.
        if (
            vector.arousal <= _FLAT_AROUSAL
            and vector.intensity <= _FLAT_INTENSITY
            and vector.valence >= -_DIRECTIONAL_VALENCE
        ):
            return ContourShape.FLAT
        # 5. Playful humor oscillates; other positive affect rises.
        if vector.humor >= _DOMINANT_THRESHOLD:
            return ContourShape.OSCILLATING
        if vector.valence >= _DIRECTIONAL_VALENCE:
            return ContourShape.RISING
        # 6. Trailing-off terminal fades; negative affect falls.
        if terminal_signal is ControlSignal.TRAILING_OFF:
            return ContourShape.FALLING
        if vector.valence <= -_DIRECTIONAL_VALENCE:
            return ContourShape.FALLING
        # 7. Reinforce with pitch tendency, else default declarative fall.
        if pitch is not None and pitch.tendency is PitchTendency.RISING:
            return ContourShape.RISING
        return ContourShape.FALLING


_default_generator: Final[ContourGenerator] = ContourGenerator()


def generate_contour(
    vector: EmotionalVector,
    pitch: PitchProfile | None = None,
    terminal_signal: ControlSignal | None = None,
) -> ContourResult:
    """Generate a contour using the module-level default generator."""
    return _default_generator.generate(vector, pitch, terminal_signal)
