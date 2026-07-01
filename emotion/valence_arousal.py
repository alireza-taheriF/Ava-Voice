"""Categorical → dimensional affect mapping (Phase 1B).

The classifier speaks in *categories* (joy, anger, …); most prosody and TTS
conditioning models prefer the continuous **VAD** space (Valence, Arousal,
Dominance). This module is the deterministic bridge between the two.

Method
------
Each primary emotion is assigned a fixed **VAD anchor** — a canonical point in
affect space grounded in the affective-psychology literature (Russell's
circumplex extended with a dominance axis). A sentence's dimensional coordinate
is the *score-weighted centroid* of the anchors of its active emotions:

.. math::

    (V, A, D) = \\frac{\\sum_e s_e \\cdot (V_e, A_e, D_e)}{\\sum_e s_e}

When no emotion is active the mapping returns a neutral resting point
(``valence = 0``, low ``arousal``, mid ``dominance``). The computation is a pure
function of the input scores: no randomness, no state.

Examples
--------
>>> mapper = ValenceArousalMapper()
>>> vad = mapper.map({"joy": 0.9})
>>> vad.valence > 0.5 and vad.arousal > 0.5
True
>>> mapper.map({"sadness": 0.8}).valence < 0
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import PRIMARY_DIMENSIONS, clamp

_logger = get_logger("emotion.valence_arousal")


@dataclass(frozen=True)
class VADCoordinate:
    """A point in Valence–Arousal–Dominance space.

    Attributes
    ----------
    valence:
        Pleasantness in ``[-1, +1]``.
    arousal:
        Activation / energy in ``[0, 1]``.
    dominance:
        Sense of control in ``[0, 1]``.
    """

    valence: float
    arousal: float
    dominance: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable ``{axis: value}`` mapping."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
        }


# Canonical VAD anchors per primary emotion. These constants are the crux of the
# mapping and are deliberately explicit and documented for patent-grade clarity.
#            emotion        valence  arousal  dominance
_ANCHORS: Final[dict[str, VADCoordinate]] = {
    "joy":       VADCoordinate(valence=0.85, arousal=0.70, dominance=0.65),
    "sadness":   VADCoordinate(valence=-0.75, arousal=0.28, dominance=0.30),
    "anger":     VADCoordinate(valence=-0.65, arousal=0.85, dominance=0.80),
    "fear":      VADCoordinate(valence=-0.70, arousal=0.82, dominance=0.20),
    "calm":      VADCoordinate(valence=0.35, arousal=0.15, dominance=0.55),
    "humor":     VADCoordinate(valence=0.65, arousal=0.62, dominance=0.62),
    "formality": VADCoordinate(valence=0.05, arousal=0.35, dominance=0.58),
}

# Neutral resting coordinate when no emotion is active. Arousal is slightly
# above zero to model natural baseline animation of speech; dominance is mid.
_NEUTRAL: Final[VADCoordinate] = VADCoordinate(
    valence=0.0, arousal=0.10, dominance=0.50
)


class ValenceArousalMapper:
    """Deterministic mapper from categorical scores to a :class:`VADCoordinate`.

    Anchors may be overridden through the constructor (e.g. to localize the
    model for a specific speaker persona) without changing the algorithm.
    """

    def __init__(
        self, anchors: Mapping[str, VADCoordinate] | None = None
    ) -> None:
        self._anchors: dict[str, VADCoordinate] = dict(_ANCHORS)
        if anchors:
            self._anchors.update(anchors)

    def map(self, scores: Mapping[str, float]) -> VADCoordinate:
        """Map emotion ``scores`` to a VAD coordinate via the weighted centroid.

        Parameters
        ----------
        scores:
            ``{emotion: score}`` mapping. Missing or non-positive entries are
            ignored. Scores need not be normalized — only relative magnitude
            matters for the centroid.
        """
        total = 0.0
        valence = 0.0
        arousal = 0.0
        dominance = 0.0
        # Iterate in canonical order so floating-point accumulation is stable.
        for emotion in PRIMARY_DIMENSIONS:
            score = scores.get(emotion, 0.0)
            if score <= 0.0:
                continue
            anchor = self._anchors.get(emotion)
            if anchor is None:
                continue
            total += score
            valence += score * anchor.valence
            arousal += score * anchor.arousal
            dominance += score * anchor.dominance

        if total == 0.0:
            return _NEUTRAL

        return VADCoordinate(
            valence=clamp(valence / total, -1.0, 1.0),
            arousal=clamp(arousal / total, 0.0, 1.0),
            dominance=clamp(dominance / total, 0.0, 1.0),
        )


_default_mapper: Final[ValenceArousalMapper] = ValenceArousalMapper()


def map_to_vad(scores: Mapping[str, float]) -> VADCoordinate:
    """Map ``scores`` to VAD using the module-level default mapper."""
    return _default_mapper.map(scores)
