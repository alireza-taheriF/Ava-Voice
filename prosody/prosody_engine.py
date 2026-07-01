"""Prosody engine orchestration (Phase 1C).

This is the top of the Prosody Intelligence Layer and the culmination of the
Persian Adaptive Emotional Prosody Engine: it converts *emotional meaning* into
*speech behavior*. It is the prosodic analogue of
:class:`~ava_voice.linguistics.token_graph.TokenGraphBuilder` and
:class:`~ava_voice.emotion.emotion_fusion.EmotionEngine`.

Input / output
--------------
* **Input** — a Phase 1A :class:`~ava_voice.linguistics.token_graph.TokenGraph`
  (structure, stress, punctuation) *and* a Phase 1B
  :class:`~ava_voice.emotion.emotion_schema.EmotionalVector` (affect).
* **Output** — a single immutable
  :class:`~ava_voice.prosody.prosody_schema.ProsodyProfile`, wrapped in a
  :class:`ProsodyResult` that also exposes every intermediate sub-profile/plan
  for auditability.

Pipeline (each stage is a pure, single-responsibility module)
-------------------------------------------------------------
``pitch → tempo → energy → pause → emphasis → contour → assemble profile
→ annotate graph``

Graph annotation
----------------
When ``annotate`` is enabled the engine writes prosodic metadata onto the
existing graph nodes (no new node types are introduced):

* the ``SENTENCE`` node receives the full profile, the contour, and the detailed
  sub-profiles under ``prosody`` / ``contour`` / ``prosody_detail``;
* each ``TOKEN`` node receives its emphasis/elongation decision under ``prosody``;
* each ``PAUSE`` node receives its scaled duration under ``prosody``.

Everything is deterministic: identical ``(graph, vector)`` inputs always yield
identical profiles and annotations — no randomness, no ML, no clock.

Examples
--------
>>> from ava_voice.linguistics.token_graph import build_token_graph
>>> from ava_voice.emotion.emotion_fusion import analyze_once
>>> graph = build_token_graph("خیلی خوشحالم!")
>>> vector = analyze_once("خیلی خوشحالم!").vector
>>> profile = ProsodyEngine().generate(graph, vector).profile
>>> profile.speaking_rate > 1.0  # joy speeds speech up
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import EmotionalVector
from ava_voice.linguistics.punctuation_mapper import ControlSignal
from ava_voice.linguistics.token_graph import NodeType, TokenGraph
from ava_voice.prosody.contour_generator import ContourGenerator, ContourResult
from ava_voice.prosody.emphasis_mapper import EmphasisMapper, EmphasisPlan
from ava_voice.prosody.energy_mapper import EnergyMapper, EnergyProfile
from ava_voice.prosody.pause_mapper import PauseMapper, PausePlan
from ava_voice.prosody.pitch_mapper import PitchMapper, PitchProfile
from ava_voice.prosody.prosody_schema import ProsodyProfile
from ava_voice.prosody.tempo_mapper import TempoMapper, TempoProfile

_logger = get_logger("prosody.engine")


@dataclass(frozen=True)
class ProsodyResult:
    """The complete, auditable result of prosody generation.

    Attributes
    ----------
    profile:
        The final :class:`ProsodyProfile` (the deliverable).
    pitch, tempo, energy:
        The per-dimension emotion→prosody sub-profiles.
    pause:
        The :class:`PausePlan` (placement, duration, density).
    emphasis:
        The :class:`EmphasisPlan` (per-token prominence and elongation).
    contour:
        The :class:`ContourResult` (sentence-level intonation).
    """

    profile: ProsodyProfile
    pitch: PitchProfile
    tempo: TempoProfile
    energy: EnergyProfile
    pause: PausePlan
    emphasis: EmphasisPlan
    contour: ContourResult

    def to_dict(self) -> dict[str, object]:
        """Return a fully JSON-serializable representation of the result."""
        return {
            "profile": self.profile.to_dict(),
            "pitch": self.pitch.to_dict(),
            "tempo": self.tempo.to_dict(),
            "energy": self.energy.to_dict(),
            "pause": self.pause.to_dict(),
            "emphasis": self.emphasis.to_dict(),
            "contour": self.contour.to_dict(),
        }


class ProsodyEngine:
    """Deterministic orchestration of the full Prosody Intelligence Layer.

    All collaborators are injected (and default to their module singletons),
    keeping the orchestration thin and each stage independently testable.
    """

    def __init__(
        self,
        pitch_mapper: PitchMapper | None = None,
        tempo_mapper: TempoMapper | None = None,
        energy_mapper: EnergyMapper | None = None,
        pause_mapper: PauseMapper | None = None,
        emphasis_mapper: EmphasisMapper | None = None,
        contour_generator: ContourGenerator | None = None,
    ) -> None:
        self._pitch = pitch_mapper or PitchMapper()
        self._tempo = tempo_mapper or TempoMapper()
        self._energy = energy_mapper or EnergyMapper()
        self._pause = pause_mapper or PauseMapper()
        self._emphasis = emphasis_mapper or EmphasisMapper()
        self._contour = contour_generator or ContourGenerator()

    # -- Public API --------------------------------------------------------- #
    def generate(
        self, graph: TokenGraph, vector: EmotionalVector, *, annotate: bool = True
    ) -> ProsodyResult:
        """Generate a :class:`ProsodyResult` from a token ``graph`` and ``vector``.

        When ``annotate`` is ``True`` (default) prosodic metadata is written back
        onto the graph nodes in place.
        """
        pitch = self._pitch.map(vector)
        tempo = self._tempo.map(vector)
        energy = self._energy.map(vector)
        pause = self._pause.map(graph, vector)
        emphasis = self._emphasis.map(graph, vector)
        terminal = self._terminal_signal(graph)
        contour = self._contour.generate(vector, pitch, terminal)

        profile = ProsodyProfile(
            pitch_base=pitch.base,
            pitch_variance=pitch.variance,
            speaking_rate=tempo.rate,
            energy_level=energy.level,
            pause_density=pause.density,
            pause_duration=pause.duration_scale,
            emphasis_strength=emphasis.emphasis_strength,
            contour_shape=contour.shape,
            elongation_factor=emphasis.elongation_factor,
        )

        result = ProsodyResult(
            profile=profile,
            pitch=pitch,
            tempo=tempo,
            energy=energy,
            pause=pause,
            emphasis=emphasis,
            contour=contour,
        )

        if annotate:
            self._annotate_graph(graph, result)
        return result

    # -- Terminal signal ---------------------------------------------------- #
    @staticmethod
    def _terminal_signal(graph: TokenGraph) -> ControlSignal | None:
        """Return the sentence-final punctuation control signal, if any.

        The last ``PAUSE`` node in spoken-timeline order carries the terminal
        boundary (question rise, statement fall, trailing off, …).
        """
        pause_nodes = graph.nodes_of_type(NodeType.PAUSE)
        if not pause_nodes:
            return None
        raw = pause_nodes[-1].attributes.get("signal")
        try:
            return ControlSignal(str(raw))
        except ValueError:
            return None

    # -- Graph annotation --------------------------------------------------- #
    def _annotate_graph(self, graph: TokenGraph, result: ProsodyResult) -> None:
        """Write prosodic metadata onto SENTENCE, TOKEN and PAUSE nodes."""
        for sentence in graph.nodes_of_type(NodeType.SENTENCE):
            sentence.attributes["prosody"] = result.profile.to_dict()
            sentence.attributes["contour"] = result.contour.to_dict()
            sentence.attributes["prosody_detail"] = {
                "pitch": result.pitch.to_dict(),
                "tempo": result.tempo.to_dict(),
                "energy": result.energy.to_dict(),
                "pause": result.pause.to_dict(),
                "emphasis": result.emphasis.to_dict(),
            }

        emphasis_by_id = {t.node_id: t for t in result.emphasis.tokens}
        for token in graph.nodes_of_type(NodeType.TOKEN):
            decision = emphasis_by_id.get(token.id)
            if decision is None:
                continue
            token.attributes["prosody"] = {
                "emphasized": decision.emphasized,
                "boost": decision.boost,
                "elongate": decision.elongate,
                "elongation_factor": (
                    result.emphasis.elongation_factor if decision.elongate else 1.0
                ),
            }

        pause_by_id = {p.node_id: p for p in result.pause.placements}
        for node in graph.nodes_of_type(NodeType.PAUSE):
            placement = pause_by_id.get(node.id)
            if placement is None:
                continue
            node.attributes["prosody"] = {
                "nominal_ms": placement.nominal_ms,
                "duration_ms": placement.duration_ms,
                "duration_scale": result.pause.duration_scale,
            }


_default_engine: Final[ProsodyEngine] = ProsodyEngine()


def generate_prosody(
    graph: TokenGraph, vector: EmotionalVector, *, annotate: bool = True
) -> ProsodyResult:
    """Generate prosody using the module-level default engine."""
    return _default_engine.generate(graph, vector, annotate=annotate)
