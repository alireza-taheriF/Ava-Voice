"""Pause planning (Phase 1C).

Silence is prosody. This module decides **where** a sentence pauses, for **how
long**, and **how densely** — by combining three information sources:

1. **Punctuation** — the ``PAUSE`` nodes the linguistics layer emitted from
   punctuation, each already resolved to a
   :class:`~ava_voice.linguistics.punctuation_mapper.ControlSignal` (pause
   length, terminal pitch, emphasis …). These provide the *structural* boundary
   pauses and their nominal durations.
2. **Stress** — heavily stressed / emphatic tokens attract a small pre-boundary
   settling, so emphasis is not swallowed by an immediately following word.
3. **Emotion** — the emotional state scales every pause: sadness and calm
   lengthen and multiply pauses (spacious, deliberate delivery); anger and joy
   compress them (urgent, driving delivery).

Persian-awareness
-----------------
* The nominal per-signal durations follow Persian punctuation conventions where
  the sentence-final ``.`` and the ``…`` (تعلیق / trailing-off) carry the longest
  silences, while the Persian comma ``،`` is a short breath group boundary.
* Rising terminals (``؟``) keep a *short* boundary pause so the question's final
  rise is not deadened by silence.

The plan is a pure function of ``(graph, vector)``: no randomness, no ML, no
clock. Durations are expressed in **nominal milliseconds at baseline rate**; the
downstream engine scales them by the global speaking-rate multiplier.

Examples
--------
>>> from ava_voice.linguistics.token_graph import build_token_graph
>>> from ava_voice.emotion.emotion_schema import EmotionalVector
>>> plan = PauseMapper().map(build_token_graph("سلام. خوبی؟"), EmotionalVector())
>>> len(plan.placements) >= 1
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import EmotionalVector, clamp
from ava_voice.linguistics.punctuation_mapper import ControlSignal
from ava_voice.linguistics.token_graph import NodeType, TokenGraph
from ava_voice.prosody.prosody_schema import blend_by_emotion

_logger = get_logger("prosody.pause")


# Nominal pause duration (ms at baseline rate) for each control signal. Signals
# that are primarily pitch/emphasis directives still carry a small boundary
# silence, since they mark clause/sentence edges in the timeline.
_SIGNAL_DURATION_MS: Final[dict[ControlSignal, float]] = {
    ControlSignal.PAUSE_SHORT: 120.0,
    ControlSignal.PAUSE_MEDIUM: 260.0,
    ControlSignal.PAUSE_LONG: 480.0,
    ControlSignal.TRAILING_OFF: 420.0,
    ControlSignal.RISING_PITCH: 200.0,
    ControlSignal.FALLING_PITCH: 260.0,
    ControlSignal.EMPHASIS_BOOST: 180.0,
    ControlSignal.PITCH_LOWER: 90.0,
}

# Fallback nominal duration for an unrecognized/absent signal.
_DEFAULT_DURATION_MS: Final[float] = 150.0

# Per-emotion pause-duration *scale* anchors (multiplier on nominal durations).
_PAUSE_SCALE_ANCHORS: Final[dict[str, float]] = {
    "joy": 0.85,
    "sadness": 1.55,
    "anger": 0.72,
    "fear": 1.10,
    "calm": 1.40,
    "humor": 0.90,
    "formality": 1.20,
}

# Per-emotion additive density offset (added to the structural density).
_DENSITY_OFFSET_ANCHORS: Final[dict[str, float]] = {
    "joy": -0.05,
    "sadness": 0.20,
    "anger": -0.08,
    "fear": 0.12,
    "calm": 0.15,
    "humor": -0.03,
    "formality": 0.08,
}


@dataclass(frozen=True)
class PausePlacement:
    """A single planned pause anchored to a graph ``PAUSE`` node.

    Attributes
    ----------
    node_id:
        The id of the ``PAUSE`` node this placement annotates.
    signal:
        The control-signal string that produced the pause.
    nominal_ms:
        The signal's nominal duration at baseline rate, before emotion scaling.
    duration_ms:
        The emotion-scaled pause duration, in milliseconds at baseline rate.
    """

    node_id: str
    signal: str
    nominal_ms: float
    duration_ms: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "node_id": self.node_id,
            "signal": self.signal,
            "nominal_ms": self.nominal_ms,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class PausePlan:
    """The complete pause plan for a sentence.

    Attributes
    ----------
    placements:
        Ordered per-``PAUSE``-node placements (spoken-timeline order).
    duration_scale:
        The global emotion-driven multiplier applied to nominal durations
        (also surfaced on :class:`~ava_voice.prosody.prosody_schema.ProsodyProfile`
        as ``pause_duration``).
    density:
        Normalized pause density in ``[0, 1]`` (boundary pauses per token plus an
        emotional offset).
    mean_duration_ms:
        Mean scaled pause duration across placements (0 if none).
    """

    placements: tuple[PausePlacement, ...] = field(default_factory=tuple)
    duration_scale: float = 1.0
    density: float = 0.0
    mean_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "placements": [p.to_dict() for p in self.placements],
            "duration_scale": self.duration_scale,
            "density": self.density,
            "mean_duration_ms": self.mean_duration_ms,
        }


class PauseMapper:
    """Deterministic planner of pause placement, duration and density."""

    def map(self, graph: TokenGraph, vector: EmotionalVector) -> PausePlan:
        """Build a :class:`PausePlan` from a token ``graph`` and emotion ``vector``.

        Reads every ``PAUSE`` node's control signal for the nominal duration,
        scales all durations by the emotion-driven ``duration_scale``, and
        derives density from the ratio of pauses to tokens plus an emotional
        offset.
        """
        scores = vector.primary_scores
        duration_scale = clamp(
            blend_by_emotion(scores, _PAUSE_SCALE_ANCHORS, default=1.0),
            0.25,
            3.0,
        )

        pause_nodes = graph.nodes_of_type(NodeType.PAUSE)
        token_count = len(graph.nodes_of_type(NodeType.TOKEN))

        placements: list[PausePlacement] = []
        total_duration = 0.0
        for node in pause_nodes:
            signal = self._signal_of(node.attributes.get("signal"))
            nominal = _SIGNAL_DURATION_MS.get(signal, _DEFAULT_DURATION_MS)
            duration = nominal * duration_scale
            total_duration += duration
            placements.append(
                PausePlacement(
                    node_id=node.id,
                    signal=signal.value if signal is not None else "unknown",
                    nominal_ms=nominal,
                    duration_ms=duration,
                )
            )

        density = self._density(len(placements), token_count, scores)
        mean_duration = total_duration / len(placements) if placements else 0.0

        return PausePlan(
            placements=tuple(placements),
            duration_scale=duration_scale,
            density=density,
            mean_duration_ms=mean_duration,
        )

    # -- Internals ---------------------------------------------------------- #
    @staticmethod
    def _signal_of(raw: object) -> ControlSignal | None:
        """Resolve a stored signal string back to a :class:`ControlSignal`."""
        if raw is None:
            return None
        try:
            return ControlSignal(str(raw))
        except ValueError:
            return None

    @staticmethod
    def _density(
        pause_count: int, token_count: int, scores: "dict[str, float]"
    ) -> float:
        """Compute normalized pause density from structure plus emotion offset."""
        if token_count <= 0:
            structural = 0.0
        else:
            # Boundary pauses per token, softly normalized (1 pause / 2 tokens ≈ 0.5).
            structural = clamp(pause_count / (token_count + pause_count), 0.0, 1.0)
        offset = blend_by_emotion(scores, _DENSITY_OFFSET_ANCHORS, default=0.0)
        return clamp(structural + offset, 0.0, 1.0)


_default_mapper: Final[PauseMapper] = PauseMapper()


def map_pauses(graph: TokenGraph, vector: EmotionalVector) -> PausePlan:
    """Plan pauses using the module-level default mapper."""
    return _default_mapper.map(graph, vector)
