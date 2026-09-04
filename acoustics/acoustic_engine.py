"""Acoustic engine orchestration (Phase 2A).

This is the top of the Adaptive Persian Acoustic Layer and the final symbolic
stage of the Persian Adaptive Emotional Prosody Engine: it converts *speech
behavior* (a :class:`~ava_voice.prosody.prosody_schema.ProsodyProfile`) plus
linguistic structure (a
:class:`~ava_voice.linguistics.token_graph.TokenGraph`) into concrete
**synthesis parameters** — an :class:`~ava_voice.acoustics.acoustic_schema.AcousticProfile`.
No waveform is produced; this layer stops at the frame/phoneme parameter level.

Input / output
--------------
* **Input** — a Phase 1A token graph (phonemes, stress, punctuation; ideally
  already prosody-annotated by Phase 1C) *and* a Phase 1C prosody profile.
* **Output** — an immutable, serializable ``AcousticProfile`` (phoneme
  durations, F0 & energy curves, pause segments, elongation segments, stress
  boosts, master timeline).

Pipeline (each stage is a pure, single-responsibility module)
-------------------------------------------------------------
``extract & align (phonemes+stress) → base durations → elongation
→ pause shaping → assemble timeline → pitch curve → energy curve
→ assemble AcousticProfile → annotate graph``

Graph annotation
----------------
When ``annotate`` is enabled the engine writes acoustic metadata onto existing
nodes (no new node types): each ``PHONEME`` node gets its absolute timing, each
``TOKEN`` and ``PAUSE`` node its span, and the ``SENTENCE`` node the full
serialized profile under ``acoustic``.

Everything is deterministic: identical ``(graph, profile)`` inputs always yield
identical parameters and annotations — no randomness, no ML, no clock.

Examples
--------
>>> from ava_voice.linguistics.token_graph import build_token_graph
>>> from ava_voice.emotion.emotion_fusion import analyze_once
>>> from ava_voice.prosody.prosody_engine import generate_prosody
>>> graph = build_token_graph("سلام!")
>>> profile = generate_prosody(graph, analyze_once("سلام!").vector).profile
>>> acoustic = AcousticEngine().generate(graph, profile)
>>> acoustic.total_duration_ms > 0
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ava_voice.acoustics.acoustic_schema import (
    DEFAULT_HOP_MS,
    AcousticProfile,
    ElongationSegment,
    StressBoost,
)
from ava_voice.acoustics.duration_mapper import DurationMapper
from ava_voice.acoustics.elongation_controller import ElongationController
from ava_voice.acoustics.energy_envelope import EnergyEnvelopeGenerator
from ava_voice.acoustics.phoneme_aligner import (
    AlignmentResult,
    PauseSlot,
    PhonemeAligner,
    TokenSlot,
)
from ava_voice.acoustics.pause_shaper import PauseShape, PauseShaper
from ava_voice.acoustics.pitch_envelope import PitchEnvelopeGenerator
from ava_voice.core.logger import get_logger
from ava_voice.linguistics.punctuation_mapper import ControlSignal
from ava_voice.linguistics.token_graph import NodeType, TokenGraph
from ava_voice.prosody.prosody_schema import ProsodyProfile

_logger = get_logger("acoustics.engine")


@dataclass(frozen=True)
class _WorkingTiming:
    """Internal accumulator bundling per-phoneme working state."""

    durations_by_index: dict[int, float]
    elongated_indices: set[int]
    elongation_segments: list[ElongationSegment]
    index_to_node: dict[int, str | None]


class AcousticEngine:
    """Deterministic orchestration of the Adaptive Persian Acoustic Layer."""

    def __init__(
        self,
        aligner: PhonemeAligner | None = None,
        duration_mapper: DurationMapper | None = None,
        elongation_controller: ElongationController | None = None,
        pause_shaper: PauseShaper | None = None,
        pitch_generator: PitchEnvelopeGenerator | None = None,
        energy_generator: EnergyEnvelopeGenerator | None = None,
        hop_ms: float = DEFAULT_HOP_MS,
    ) -> None:
        self._aligner = aligner or PhonemeAligner()
        self._duration = duration_mapper or DurationMapper()
        self._elongation = elongation_controller or ElongationController()
        self._pause = pause_shaper or PauseShaper()
        self._pitch = pitch_generator or PitchEnvelopeGenerator()
        self._energy = energy_generator or EnergyEnvelopeGenerator()
        self._hop_ms = hop_ms

    # -- Public API --------------------------------------------------------- #
    def generate(
        self, graph: TokenGraph, profile: ProsodyProfile, *, annotate: bool = True
    ) -> AcousticProfile:
        """Generate an :class:`AcousticProfile` from ``graph`` and ``profile``.

        When ``annotate`` is ``True`` (default) acoustic metadata is written back
        onto the graph nodes in place.
        """
        slots = self._aligner.extract_sequence(graph)

        timing = self._compute_durations(slots, profile)
        pause_shapes = self._shape_pauses(slots, profile)

        alignment = self._aligner.assemble_timeline(
            slots,
            timing.durations_by_index,
            timing.elongated_indices,
            pause_shapes,
        )

        emphasis_centers = self._emphasis_centers(slots, alignment, profile)
        pitch_curve = self._pitch.build(
            profile, alignment.total_duration_ms, self._hop_ms
        )
        energy_curve = self._energy.build(
            profile, alignment.total_duration_ms, self._hop_ms, emphasis_centers
        )
        stress_boosts = self._stress_boosts(slots, profile)

        acoustic = AcousticProfile(
            phoneme_durations=alignment.phoneme_durations,
            pitch_curve=pitch_curve,
            energy_curve=energy_curve,
            pause_segments=alignment.pause_segments,
            elongation_segments=tuple(timing.elongation_segments),
            stress_boosts=stress_boosts,
            timing_map=alignment.timing_map,
            total_duration_ms=alignment.total_duration_ms,
            frame_hop_ms=self._hop_ms,
        )

        if annotate:
            self._annotate_graph(graph, acoustic, timing.index_to_node)
        return acoustic

    # -- Durations + elongation -------------------------------------------- #
    def _compute_durations(
        self, slots: list[TokenSlot | PauseSlot], profile: ProsodyProfile
    ) -> _WorkingTiming:
        """Compute final per-phoneme durations (base + elongation) per token."""
        durations_by_index: dict[int, float] = {}
        elongated_indices: set[int] = set()
        elongation_segments: list[ElongationSegment] = []
        index_to_node: dict[int, str | None] = {}

        for slot in slots:
            if not isinstance(slot, TokenSlot):
                continue
            base = self._duration.map_units(slot.units, profile.speaking_rate)
            result = self._elongation.apply(
                slot.units, base, profile.elongation_factor, slot.elongate
            )
            for unit, duration in zip(slot.units, result.durations):
                durations_by_index[unit.index] = duration
                index_to_node[unit.index] = unit.node_id
            for segment in result.segments:
                elongated_indices.add(segment.phoneme_index)
                elongation_segments.append(segment)

        return _WorkingTiming(
            durations_by_index=durations_by_index,
            elongated_indices=elongated_indices,
            elongation_segments=elongation_segments,
            index_to_node=index_to_node,
        )

    # -- Pauses ------------------------------------------------------------- #
    def _shape_pauses(
        self, slots: list[TokenSlot | PauseSlot], profile: ProsodyProfile
    ) -> dict[str, PauseShape]:
        """Shape every pause slot's silence duration."""
        shapes: dict[str, PauseShape] = {}
        for slot in slots:
            if isinstance(slot, PauseSlot):
                signal = self._to_signal(slot.signal)
                shapes[slot.node_id] = self._pause.shape(signal, profile)
        return shapes

    @staticmethod
    def _to_signal(raw: str | None) -> ControlSignal | None:
        """Resolve a stored signal string back to a :class:`ControlSignal`."""
        if raw is None:
            return None
        try:
            return ControlSignal(str(raw))
        except ValueError:
            return None

    # -- Emphasis / stress -------------------------------------------------- #
    @staticmethod
    def _emphasis_centers(
        slots: list[TokenSlot | PauseSlot],
        alignment: AlignmentResult,
        profile: ProsodyProfile,
    ) -> list[tuple[float, float]]:
        """Return ``(center_ms, strength)`` pairs for emphasized tokens."""
        span_by_token: dict[str, tuple[float, float]] = {}
        for entry in alignment.timing_map:
            if entry.kind == "token":
                span_by_token[entry.ref_id] = (entry.start_ms, entry.end_ms)

        centers: list[tuple[float, float]] = []
        for slot in slots:
            if not isinstance(slot, TokenSlot) or not slot.emphasized:
                continue
            span = span_by_token.get(slot.token_id)
            if span is None:
                continue
            center = (span[0] + span[1]) / 2.0
            strength = slot.boost if slot.boost > 0.0 else profile.emphasis_strength
            centers.append((center, strength))
        return centers

    @staticmethod
    def _stress_boosts(
        slots: list[TokenSlot | PauseSlot], profile: ProsodyProfile
    ) -> tuple[StressBoost, ...]:
        """Build per-token stress-boost records for emphasized tokens."""
        boosts: list[StressBoost] = []
        for slot in slots:
            if not isinstance(slot, TokenSlot):
                continue
            if not slot.emphasized and slot.boost <= 0.0:
                continue
            boost = slot.boost if slot.boost > 0.0 else profile.emphasis_strength
            boosts.append(
                StressBoost(
                    token_id=slot.token_id,
                    surface=slot.surface,
                    boost=round(boost, 4),
                    emphasized=slot.emphasized,
                )
            )
        return tuple(boosts)

    # -- Graph annotation --------------------------------------------------- #
    def _annotate_graph(
        self,
        graph: TokenGraph,
        acoustic: AcousticProfile,
        index_to_node: dict[int, str | None],
    ) -> None:
        """Write acoustic metadata onto PHONEME, TOKEN, PAUSE and SENTENCE nodes."""
        node_by_id = {n.id: n for n in graph.nodes}

        # Phoneme-level timing.
        for phoneme in acoustic.phoneme_durations:
            node_id = index_to_node.get(phoneme.index)
            node = node_by_id.get(node_id) if node_id is not None else None
            if node is not None:
                node.attributes["acoustic"] = {
                    "duration_ms": phoneme.duration_ms,
                    "start_ms": phoneme.start_ms,
                    "end_ms": phoneme.end_ms,
                    "elongated": phoneme.elongated,
                }

        # Token / pause spans from the master timeline.
        for entry in acoustic.timing_map:
            node = node_by_id.get(entry.ref_id)
            if node is None:
                continue
            node.attributes["acoustic"] = {
                "start_ms": entry.start_ms,
                "end_ms": entry.end_ms,
                "duration_ms": entry.duration_ms,
            }

        # Sentence-level full profile.
        for sentence in graph.nodes_of_type(NodeType.SENTENCE):
            sentence.attributes["acoustic"] = acoustic.to_dict()


_default_engine: Final[AcousticEngine] = AcousticEngine()


def generate_acoustics(
    graph: TokenGraph, profile: ProsodyProfile, *, annotate: bool = True
) -> AcousticProfile:
    """Generate acoustics using the module-level default engine."""
    return _default_engine.generate(graph, profile, annotate=annotate)
