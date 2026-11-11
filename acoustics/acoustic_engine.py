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
    EzafeSegment,
    StressBoost,
)
from ava_voice.acoustics.duration_mapper import DurationMapper
from ava_voice.acoustics.elongation_controller import ElongationController
from ava_voice.acoustics.energy_envelope import EnergyEnvelopeGenerator
from ava_voice.acoustics.ezafe_modulator import EzafeModulator, EzafePlan
from ava_voice.acoustics.phoneme_aligner import (
    AlignmentResult,
    PauseSlot,
    PhonemeAligner,
    TokenSlot,
)
from ava_voice.acoustics.pause_shaper import PauseShape, PauseShaper
from ava_voice.acoustics.pitch_envelope import PitchEnvelopeGenerator
from ava_voice.acoustics.speaker_baseline import DEFAULT_SPEAKER, SpeakerBaseline
from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import EmotionalVector
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
    ezafe_plan: EzafePlan
    ezafe_base_ms: dict[int, float]


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
        ezafe_modulator: EzafeModulator | None = None,
        hop_ms: float = DEFAULT_HOP_MS,
    ) -> None:
        self._aligner = aligner or PhonemeAligner()
        self._duration = duration_mapper or DurationMapper()
        self._elongation = elongation_controller or ElongationController()
        self._pause = pause_shaper or PauseShaper()
        self._pitch = pitch_generator or PitchEnvelopeGenerator()
        self._energy = energy_generator or EnergyEnvelopeGenerator()
        self._ezafe = ezafe_modulator or EzafeModulator()
        self._hop_ms = hop_ms

    # -- Public API --------------------------------------------------------- #
    def generate(
        self,
        graph: TokenGraph,
        profile: ProsodyProfile,
        *,
        vector: EmotionalVector | None = None,
        speaker: SpeakerBaseline | None = None,
        annotate: bool = True,
    ) -> AcousticProfile:
        """Generate an :class:`AcousticProfile` from ``graph`` and ``profile``.

        ``vector`` drives ezafe-linker modulation. When omitted, the engine
        reads ``SENTENCE.attributes['emotion']`` (filled by the emotion
        layer) and otherwise uses the neutral vector.

        ``speaker`` supplies the F0 baseline used to lift the semitone curve
        into hertz. Defaults to :data:`~ava_voice.acoustics.speaker_baseline.DEFAULT_SPEAKER`.

        When ``annotate`` is ``True`` (default) acoustic metadata is written back
        onto the graph nodes in place.
        """
        speaker = speaker or DEFAULT_SPEAKER
        vector = vector or self._vector_from_graph(graph)
        slots = self._aligner.extract_sequence(graph)

        timing = self._compute_durations(slots, profile, vector)
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
        pitch_curve, energy_curve = self._apply_ezafe_overlays(
            pitch_curve, energy_curve, alignment, timing.ezafe_plan
        )
        stress_boosts = self._stress_boosts(slots, profile)
        ezafe_segments = self._ezafe_segments(alignment, timing)

        acoustic = AcousticProfile(
            phoneme_durations=alignment.phoneme_durations,
            pitch_curve=pitch_curve,
            pitch_curve_hz=speaker.to_hz_curve(pitch_curve),
            energy_curve=energy_curve,
            pause_segments=alignment.pause_segments,
            elongation_segments=tuple(timing.elongation_segments),
            ezafe_segments=ezafe_segments,
            stress_boosts=stress_boosts,
            timing_map=alignment.timing_map,
            total_duration_ms=alignment.total_duration_ms,
            frame_hop_ms=self._hop_ms,
            speaker_f0_hz=speaker.f0_hz,
        )

        if annotate:
            self._annotate_graph(graph, acoustic, timing.index_to_node)
        return acoustic

    # -- Durations + elongation -------------------------------------------- #
    def _compute_durations(
        self,
        slots: list[TokenSlot | PauseSlot],
        profile: ProsodyProfile,
        vector: EmotionalVector,
    ) -> _WorkingTiming:
        """Compute final per-phoneme durations (base + ezafe + elongation)."""
        durations_by_index: dict[int, float] = {}
        elongated_indices: set[int] = set()
        elongation_segments: list[ElongationSegment] = []
        index_to_node: dict[int, str | None] = {}
        ezafe_base_ms: dict[int, float] = {}
        ezafe_decisions = []

        for slot in slots:
            if not isinstance(slot, TokenSlot):
                continue
            base = self._duration.map_units(slot.units, profile.speaking_rate)
            plan = self._ezafe.plan(slot.units, vector)
            ezafe_decisions.extend(plan.decisions)
            modulated = self._ezafe.apply_durations(slot.units, base, plan)
            for unit, before, after in zip(slot.units, base, modulated):
                if unit.is_ezafe:
                    ezafe_base_ms[unit.index] = before
            result = self._elongation.apply(
                slot.units, modulated, profile.elongation_factor, slot.elongate
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
            ezafe_plan=EzafePlan(decisions=tuple(ezafe_decisions)),
            ezafe_base_ms=ezafe_base_ms,
        )

    @staticmethod
    def _vector_from_graph(graph: TokenGraph) -> EmotionalVector:
        """Read a sentence-level vector from graph annotation, or neutral."""
        for sentence in graph.nodes_of_type(NodeType.SENTENCE):
            data = sentence.attributes.get("emotion")
            if isinstance(data, dict):
                return EmotionalVector.from_dict(data)
        return EmotionalVector.neutral()

    def _apply_ezafe_overlays(
        self,
        pitch_curve: object,
        energy_curve: object,
        alignment: AlignmentResult,
        plan: EzafePlan,
    ):
        """Add local F0/energy offsets on each linker span."""
        by_index = plan.by_index()
        pitch_spans: list[tuple[float, float, float]] = []
        energy_spans: list[tuple[float, float, float]] = []
        for phoneme in alignment.phoneme_durations:
            decision = by_index.get(phoneme.index)
            if decision is None:
                continue
            pitch_spans.append(
                (phoneme.start_ms, phoneme.end_ms, decision.pitch_offset_st)
            )
            energy_spans.append(
                (phoneme.start_ms, phoneme.end_ms, decision.energy_gain)
            )
        return (
            self._ezafe.overlay_pitch(pitch_curve, pitch_spans),
            self._ezafe.overlay_energy(energy_curve, energy_spans),
        )

    @staticmethod
    def _ezafe_segments(
        alignment: AlignmentResult, timing: _WorkingTiming
    ) -> tuple[EzafeSegment, ...]:
        """Build auditable linker records from the assembled timeline."""
        by_index = timing.ezafe_plan.by_index()
        segments: list[EzafeSegment] = []
        for phoneme in alignment.phoneme_durations:
            decision = by_index.get(phoneme.index)
            if decision is None:
                continue
            base_ms = timing.ezafe_base_ms.get(phoneme.index, phoneme.duration_ms)
            segments.append(
                EzafeSegment(
                    token_id=phoneme.token_id,
                    phoneme_index=phoneme.index,
                    symbol=phoneme.symbol,
                    allomorph=decision.allomorph,
                    register=decision.register,
                    duration_factor=round(decision.duration_factor, 4),
                    base_ms=round(base_ms, 4),
                    duration_ms=phoneme.duration_ms,
                    start_ms=phoneme.start_ms,
                    end_ms=phoneme.end_ms,
                    pitch_offset_st=decision.pitch_offset_st,
                    energy_gain=decision.energy_gain,
                )
            )
        return tuple(segments)

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
    graph: TokenGraph,
    profile: ProsodyProfile,
    *,
    vector: EmotionalVector | None = None,
    speaker: SpeakerBaseline | None = None,
    annotate: bool = True,
) -> AcousticProfile:
    """Generate acoustics using the module-level default engine."""
    return _default_engine.generate(
        graph, profile, vector=vector, speaker=speaker, annotate=annotate
    )
