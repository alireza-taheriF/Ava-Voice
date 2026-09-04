"""Phoneme / stress / timing alignment (Phase 2A).

This module is the structural heart of the acoustic layer. It performs two
deterministic jobs:

1. **Extraction / alignment** (:meth:`PhonemeAligner.extract_sequence`) — walk
   the token graph in spoken order and, for every token, align its ordered
   ``PHONEME`` symbols with its ``STRESS`` syllable map, producing a flat
   sequence of :class:`~ava_voice.acoustics.acoustic_schema.PhonemeUnit` objects
   that carry each phoneme's syllable role, stress level and elongation
   candidacy. Pauses are emitted as their own slots so the timeline stays
   complete.
2. **Timeline assembly** (:meth:`PhonemeAligner.assemble_timeline`) — given the
   final per-phoneme durations and per-pause silence lengths, lay everything out
   on a single absolute millisecond timeline, yielding the synthesis-ready
   :class:`~ava_voice.acoustics.acoustic_schema.PhonemeDuration` list, the
   :class:`~ava_voice.acoustics.acoustic_schema.PauseSegment` list and the master
   :class:`~ava_voice.acoustics.acoustic_schema.TimingEntry` map.

The alignment relies on the invariant that a token's phoneme sequence equals the
concatenation of its syllables' ``onset + nucleus + coda`` (guaranteed by
:mod:`ava_voice.linguistics.stress_detector`); a defensive fallback handles any
mismatch so extraction never raises.

Everything is a pure function of the graph and the supplied durations — no
randomness, no ML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Mapping, Sequence

from ava_voice.acoustics.acoustic_schema import (
    PauseSegment,
    PhonemeDuration,
    PhonemeUnit,
    TimingEntry,
)
from ava_voice.acoustics.pause_shaper import PauseShape
from ava_voice.core.logger import get_logger
from ava_voice.linguistics.phonemizer import is_vowel
from ava_voice.linguistics.stress_detector import StressLevel
from ava_voice.linguistics.token_graph import NodeType, RelationType, TokenGraph

_logger = get_logger("acoustics.aligner")


@dataclass(frozen=True)
class TokenSlot:
    """A token on the timeline with its aligned phoneme units.

    Attributes
    ----------
    token_id:
        Id of the ``TOKEN`` node.
    surface:
        The token's surface text.
    units:
        Ordered aligned :class:`PhonemeUnit` objects.
    emphasized, boost, elongate:
        Prosodic decisions read from the token's ``prosody`` annotation (Phase
        1C), defaulting to neutral when the graph was not prosody-annotated.
    """

    token_id: str
    surface: str | None
    units: tuple[PhonemeUnit, ...]
    emphasized: bool
    boost: float
    elongate: bool


@dataclass(frozen=True)
class PauseSlot:
    """A pause on the timeline.

    Attributes
    ----------
    node_id:
        Id of the ``PAUSE`` node.
    signal:
        The control-signal string.
    """

    node_id: str
    signal: str | None


@dataclass(frozen=True)
class AlignmentResult:
    """The assembled absolute timeline.

    Attributes
    ----------
    phoneme_durations:
        Ordered per-phoneme timings with absolute start/end.
    pause_segments:
        Ordered silence spans with absolute start/end.
    timing_map:
        The master timeline of tokens and pauses.
    total_duration_ms:
        The full sentence duration.
    """

    phoneme_durations: tuple[PhonemeDuration, ...] = field(default_factory=tuple)
    pause_segments: tuple[PauseSegment, ...] = field(default_factory=tuple)
    timing_map: tuple[TimingEntry, ...] = field(default_factory=tuple)
    total_duration_ms: float = 0.0


class PhonemeAligner:
    """Deterministic extractor and timeline assembler."""

    # -- Extraction --------------------------------------------------------- #
    def extract_sequence(
        self, graph: TokenGraph
    ) -> list[TokenSlot | PauseSlot]:
        """Return the ordered token/pause slots with aligned phoneme units.

        Tokens and pauses are taken in graph insertion order, which the Phase 1A
        builder guarantees to equal spoken-timeline order.
        """
        phonemes_by_token = self._grouped_targets(graph, RelationType.HAS_PHONEME)
        stress_by_token = self._single_target(graph, RelationType.HAS_STRESS)
        node_by_id = {n.id: n for n in graph.nodes}

        slots: list[TokenSlot | PauseSlot] = []
        global_index = 0
        for node in graph.nodes:
            if node.type is NodeType.TOKEN:
                phon_nodes = sorted(
                    (node_by_id[pid] for pid in phonemes_by_token.get(node.id, [])),
                    key=lambda n: n.attributes.get("order", 0),
                )
                stress_node = node_by_id.get(stress_by_token.get(node.id, ""))
                syllables = (
                    stress_node.attributes.get("syllables", [])
                    if stress_node is not None
                    else []
                )
                units, global_index = self._align_token(
                    node.id, phon_nodes, syllables, global_index
                )
                prosody = node.attributes.get("prosody", {})
                slots.append(
                    TokenSlot(
                        token_id=node.id,
                        surface=node.value,
                        units=units,
                        emphasized=bool(prosody.get("emphasized", False)),
                        boost=float(prosody.get("boost", 0.0)),
                        elongate=bool(prosody.get("elongate", False)),
                    )
                )
            elif node.type is NodeType.PAUSE:
                slots.append(
                    PauseSlot(
                        node_id=node.id,
                        signal=node.attributes.get("signal"),
                    )
                )
        return slots

    def _align_token(
        self,
        token_id: str,
        phon_nodes: Sequence[object],
        syllables: Sequence[Mapping[str, object]],
        global_index: int,
    ) -> tuple[tuple[PhonemeUnit, ...], int]:
        """Align one token's phonemes to its syllable map into units."""
        symbols = [getattr(n, "value", None) for n in phon_nodes]
        node_ids = [getattr(n, "id", None) for n in phon_nodes]
        metadata = self._syllable_metadata(syllables)

        # Use the syllable alignment only if it matches the phoneme sequence.
        if [m["symbol"] for m in metadata] != symbols:
            metadata = self._fallback_metadata(symbols)

        units: list[PhonemeUnit] = []
        for position, symbol in enumerate(symbols):
            meta = metadata[position]
            units.append(
                PhonemeUnit(
                    index=global_index,
                    node_id=node_ids[position],
                    token_id=token_id,
                    symbol=str(symbol),
                    is_vowel=is_vowel(str(symbol)),
                    role=str(meta["role"]),
                    syllable_index=int(meta["syllable_index"]),
                    stress_level=str(meta["stress_level"]),
                    is_elongation_candidate=bool(meta["is_elongation_candidate"]),
                )
            )
            global_index += 1
        return tuple(units), global_index

    @staticmethod
    def _syllable_metadata(
        syllables: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """Flatten syllables into ordered per-phoneme role/stress metadata."""
        metadata: list[dict[str, object]] = []
        for syllable in syllables:
            index = syllable.get("index", -1)
            stress_level = syllable.get("stress_level", StressLevel.NONE.value)
            elong = syllable.get("is_elongation_candidate", False)
            for symbol in syllable.get("onset", []):
                metadata.append(
                    {
                        "symbol": symbol,
                        "role": "onset",
                        "syllable_index": index,
                        "stress_level": stress_level,
                        "is_elongation_candidate": False,
                    }
                )
            metadata.append(
                {
                    "symbol": syllable.get("nucleus"),
                    "role": "nucleus",
                    "syllable_index": index,
                    "stress_level": stress_level,
                    "is_elongation_candidate": bool(elong),
                }
            )
            for symbol in syllable.get("coda", []):
                metadata.append(
                    {
                        "symbol": symbol,
                        "role": "coda",
                        "syllable_index": index,
                        "stress_level": stress_level,
                        "is_elongation_candidate": False,
                    }
                )
        return metadata

    @staticmethod
    def _fallback_metadata(symbols: Sequence[object]) -> list[dict[str, object]]:
        """Build neutral metadata when syllable alignment is unavailable."""
        metadata: list[dict[str, object]] = []
        for symbol in symbols:
            vowel = is_vowel(str(symbol))
            metadata.append(
                {
                    "symbol": symbol,
                    "role": "nucleus" if vowel else "onset",
                    "syllable_index": -1,
                    "stress_level": StressLevel.NONE.value,
                    "is_elongation_candidate": False,
                }
            )
        return metadata

    # -- Assembly ----------------------------------------------------------- #
    def assemble_timeline(
        self,
        slots: Sequence[TokenSlot | PauseSlot],
        durations_by_index: Mapping[int, float],
        elongated_indices: set[int],
        pause_shapes: Mapping[str, PauseShape],
    ) -> AlignmentResult:
        """Lay slots out on an absolute timeline into an :class:`AlignmentResult`.

        Parameters
        ----------
        slots:
            The token/pause slots from :meth:`extract_sequence`.
        durations_by_index:
            Final per-phoneme durations keyed by global phoneme index.
        elongated_indices:
            Global indices of phonemes that were elongated (for the ``elongated``
            flag on each :class:`PhonemeDuration`).
        pause_shapes:
            Shaped pause durations keyed by ``PAUSE`` node id.
        """
        cursor = 0.0
        phoneme_durations: list[PhonemeDuration] = []
        pause_segments: list[PauseSegment] = []
        timing_map: list[TimingEntry] = []

        for slot in slots:
            if isinstance(slot, TokenSlot):
                start = cursor
                for unit in slot.units:
                    duration = float(durations_by_index.get(unit.index, 0.0))
                    phoneme_durations.append(
                        PhonemeDuration(
                            index=unit.index,
                            symbol=unit.symbol,
                            token_id=unit.token_id,
                            duration_ms=round(duration, 4),
                            start_ms=round(cursor, 4),
                            end_ms=round(cursor + duration, 4),
                            is_vowel=unit.is_vowel,
                            stress_level=unit.stress_level,
                            elongated=unit.index in elongated_indices,
                        )
                    )
                    cursor += duration
                timing_map.append(
                    TimingEntry(
                        kind="token",
                        ref_id=slot.token_id,
                        label=slot.surface,
                        start_ms=round(start, 4),
                        end_ms=round(cursor, 4),
                    )
                )
            else:  # PauseSlot
                shape = pause_shapes.get(slot.node_id)
                duration = shape.duration_ms if shape is not None else 0.0
                start = cursor
                cursor += duration
                signal = shape.signal if shape is not None else (slot.signal or "unknown")
                category = shape.category if shape is not None else "clause"
                pause_segments.append(
                    PauseSegment(
                        node_id=slot.node_id,
                        signal=signal,
                        start_ms=round(start, 4),
                        end_ms=round(cursor, 4),
                        duration_ms=round(duration, 4),
                        category=category,
                    )
                )
                timing_map.append(
                    TimingEntry(
                        kind="pause",
                        ref_id=slot.node_id,
                        label=signal,
                        start_ms=round(start, 4),
                        end_ms=round(cursor, 4),
                    )
                )

        return AlignmentResult(
            phoneme_durations=tuple(phoneme_durations),
            pause_segments=tuple(pause_segments),
            timing_map=tuple(timing_map),
            total_duration_ms=round(cursor, 4),
        )

    # -- Graph traversal helpers ------------------------------------------- #
    @staticmethod
    def _grouped_targets(
        graph: TokenGraph, relation: RelationType
    ) -> dict[str, list[str]]:
        """Map ``source_id -> [target_id, …]`` for edges of ``relation``."""
        mapping: dict[str, list[str]] = {}
        for edge in graph.edges:
            if edge.relation is relation:
                mapping.setdefault(edge.source, []).append(edge.target)
        return mapping

    @staticmethod
    def _single_target(
        graph: TokenGraph, relation: RelationType
    ) -> dict[str, str]:
        """Map ``source_id -> target_id`` for single-target edges of ``relation``."""
        mapping: dict[str, str] = {}
        for edge in graph.edges:
            if edge.relation is relation:
                mapping[edge.source] = edge.target
        return mapping


_default_aligner: Final[PhonemeAligner] = PhonemeAligner()


def extract_sequence(graph: TokenGraph) -> list[TokenSlot | PauseSlot]:
    """Extract the aligned slot sequence using the default aligner."""
    return _default_aligner.extract_sequence(graph)
