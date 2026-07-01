"""Sentence token graph — orchestration of the linguistics layer (Phase 1A).

The :class:`TokenGraph` is the integration point of the Persian Linguistic
Intelligence Layer. It represents a sentence as a typed, directed graph that
binds every linguistic annotation to its token:

* **TOKEN** nodes — words with their canonical spoken form and ambiguity data.
* **PHONEME** nodes — the phoneme sequence of each token, order-preserving.
* **STRESS** nodes — the syllable-level stress map of each token.
* **PAUSE** nodes — prosodic control signals derived from punctuation.
* **EMOTION** nodes — *placeholders* reserved for the future Persian Emotional
  Prosody Engine. They carry no affect yet, only the attachment point.

Node timeline (spoken order) is expressed with ``NEXT`` edges over TOKEN and
PAUSE nodes; per-token annotations hang off the token via typed relation edges.

The builder is deterministic: it runs the fixed pipeline
``normalize → resolve → phonemize → stress`` per token and
``punctuation → control signal`` per mark, then assembles the graph. The result
is fully serializable via :meth:`TokenGraph.to_dict`, making the whole layer
snapshot-testable.

Example
-------
>>> graph = build_token_graph("سلام!")
>>> [n.value for n in graph.nodes_of_type(NodeType.TOKEN)]
['سلام']
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Any, Final

from ava_voice.core.logger import get_logger
from ava_voice.linguistics import ambiguity_resolver, phonemizer
from ava_voice.linguistics.persian_normalizer import PersianNormalizer
from ava_voice.linguistics.punctuation_mapper import (
    ControlSignal,
    is_punctuation,
    map_punctuation,
)
from ava_voice.linguistics.stress_detector import StressDetector

_logger = get_logger("linguistics.token_graph")

# Tokenizer: a run of word characters (Persian, ZWNJ, latin, digits) OR a single
# punctuation/other non-space character. Whitespace is a separator only.
# The Persian letter range starts at U+0621 to deliberately exclude the Arabic
# punctuation block (U+060C ،, U+061B ؛, U+061F ؟), which must tokenize as marks.
_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[0-9A-Za-z\u0621-\u06ff\u200c]+")

# Signals that convey emphasis to the *preceding* token's stressed syllable.
_EMPHASIS_SIGNALS: Final[frozenset[ControlSignal]] = frozenset(
    {ControlSignal.EMPHASIS_BOOST}
)


class NodeType(str, enum.Enum):
    """Types of nodes in a :class:`TokenGraph`."""

    SENTENCE = "sentence"
    TOKEN = "token"
    PHONEME = "phoneme"
    STRESS = "stress"
    PAUSE = "pause"
    EMOTION = "emotion"


class RelationType(str, enum.Enum):
    """Types of directed edges in a :class:`TokenGraph`."""

    NEXT = "next"                 # spoken-timeline ordering (token/pause chain)
    HAS_PHONEME = "has_phoneme"   # token -> phoneme
    HAS_STRESS = "has_stress"     # token -> stress map
    HAS_EMOTION = "has_emotion"   # token -> emotion placeholder
    CONTAINS = "contains"         # sentence -> token/pause


@dataclass
class Node:
    """A typed graph node with an opaque attribute bag.

    Attributes
    ----------
    id:
        Stable, deterministic identifier (``"<type><counter>"``).
    type:
        The :class:`NodeType`.
    value:
        Primary payload (word text, phoneme symbol, signal name, …).
    attributes:
        Structured, JSON-serializable metadata for the node.
    """

    id: str
    type: NodeType
    value: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """A directed, typed edge between two node ids."""

    source: str
    target: str
    relation: RelationType


@dataclass
class TokenGraph:
    """A directed graph of tokens and their linguistic annotations."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> Node:
        """Append ``node`` and return it for convenient chaining."""
        self.nodes.append(node)
        return node

    def add_edge(self, source: str, target: str, relation: RelationType) -> None:
        """Append a directed edge ``source -> target`` with ``relation``."""
        self.edges.append(Edge(source=source, target=target, relation=relation))

    def nodes_of_type(self, node_type: NodeType) -> list[Node]:
        """Return all nodes of the given :class:`NodeType`, in insertion order."""
        return [n for n in self.nodes if n.type is node_type]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the graph."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type.value,
                    "value": n.value,
                    "attributes": n.attributes,
                }
                for n in self.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target, "relation": e.relation.value}
                for e in self.edges
            ],
        }


@dataclass(frozen=True)
class _Token:
    """Internal tokenizer output: a text span classified as word or punctuation."""

    text: str
    is_word: bool


class TokenGraphBuilder:
    """Deterministic builder that assembles a :class:`TokenGraph` from text.

    The builder wires together the individual linguistics stages. Each stage is
    injected so it can be swapped or mocked in tests, keeping the orchestration
    itself thin and auditable.
    """

    def __init__(
        self,
        normalizer: PersianNormalizer | None = None,
        resolver: ambiguity_resolver.AmbiguityResolver | None = None,
        g2p: phonemizer.PersianPhonemizer | None = None,
        stress: StressDetector | None = None,
    ) -> None:
        self._normalizer = normalizer or PersianNormalizer()
        self._resolver = resolver or ambiguity_resolver.AmbiguityResolver()
        self._g2p = g2p or phonemizer.PersianPhonemizer()
        self._stress = stress or StressDetector()
        self._counter: dict[NodeType, int] = {}

    # -- Public API --------------------------------------------------------- #
    def build(self, text: str) -> TokenGraph:
        """Normalize ``text`` and build its :class:`TokenGraph`."""
        self._counter = {}
        graph = TokenGraph()
        sentence = graph.add_node(self._new_node(NodeType.SENTENCE, value=text))

        normalized = self._normalizer.normalize(text)
        tokens = self._tokenize(normalized)

        previous_timeline_id: str | None = None
        for position, token in enumerate(tokens):
            if token.is_word:
                emphasis = self._has_following_emphasis(tokens, position)
                token_id = self._add_word(graph, sentence.id, token.text, emphasis)
            else:
                token_id = self._add_punctuation(graph, sentence.id, token.text)
                if token_id is None:
                    continue  # unrecognized punctuation is dropped

            if previous_timeline_id is not None:
                graph.add_edge(previous_timeline_id, token_id, RelationType.NEXT)
            previous_timeline_id = token_id

        return graph

    # -- Tokenization ------------------------------------------------------- #
    @staticmethod
    def _tokenize(text: str) -> list[_Token]:
        """Split ``text`` into ordered word / punctuation tokens.

        Whitespace acts purely as a separator and produces no tokens.
        """
        tokens: list[_Token] = []
        index = 0
        length = len(text)
        while index < length:
            char = text[index]
            if char.isspace():
                index += 1
                continue
            match = _WORD_RE.match(text, index)
            if match:
                tokens.append(_Token(text=match.group(), is_word=True))
                index = match.end()
            else:
                tokens.append(_Token(text=char, is_word=False))
                index += 1
        return tokens

    @staticmethod
    def _has_following_emphasis(tokens: list[_Token], position: int) -> bool:
        """Return ``True`` if the next punctuation token conveys emphasis."""
        for follower in tokens[position + 1 :]:
            if follower.is_word:
                return False
            signal = map_punctuation(follower.text)
            if signal is None:
                continue
            return signal in _EMPHASIS_SIGNALS
        return False

    # -- Node assembly ------------------------------------------------------ #
    def _add_word(
        self, graph: TokenGraph, sentence_id: str, surface: str, emphasis: bool
    ) -> str:
        """Add a TOKEN node with phoneme, stress and emotion annotations."""
        resolution = self._resolver.resolve_token(surface)
        canonical = resolution.canonical
        phoneme_tokens = self._g2p.analyze_word(canonical)
        phoneme_symbols = [pt.symbol for pt in phoneme_tokens]
        stress_map = self._stress.detect(
            phoneme_symbols, word=canonical, emphasis=emphasis
        )

        token_node = graph.add_node(
            self._new_node(
                NodeType.TOKEN,
                value=surface,
                attributes={
                    "canonical": canonical,
                    "resolution_kind": resolution.resolution_kind,
                    "is_ambiguous": resolution.is_ambiguous,
                    "alternatives": list(resolution.alternatives),
                },
            )
        )
        graph.add_edge(sentence_id, token_node.id, RelationType.CONTAINS)

        for order, ptok in enumerate(phoneme_tokens):
            phoneme_node = graph.add_node(
                self._new_node(
                    NodeType.PHONEME,
                    value=ptok.symbol,
                    attributes={
                        "order": order,
                        "source_grapheme": ptok.source_grapheme,
                        "is_ambiguous": ptok.is_ambiguous,
                        "alternatives": list(ptok.alternatives),
                        "is_placeholder": ptok.is_placeholder,
                    },
                )
            )
            graph.add_edge(token_node.id, phoneme_node.id, RelationType.HAS_PHONEME)

        stress_node = graph.add_node(
            self._new_node(
                NodeType.STRESS,
                value=str(stress_map.stressed_index),
                attributes={
                    "stressed_index": stress_map.stressed_index,
                    "emphasis_positions": list(stress_map.emphasis_positions),
                    "elongation_positions": list(stress_map.elongation_positions),
                    "syllables": [
                        {
                            "index": syl.index,
                            "onset": list(syl.onset),
                            "nucleus": syl.nucleus,
                            "coda": list(syl.coda),
                            "stress_level": syl.stress_level.value,
                            "is_elongation_candidate": syl.is_elongation_candidate,
                        }
                        for syl in stress_map.syllables
                    ],
                },
            )
        )
        graph.add_edge(token_node.id, stress_node.id, RelationType.HAS_STRESS)

        # Emotion placeholder — reserved attachment point, intentionally empty.
        emotion_node = graph.add_node(
            self._new_node(
                NodeType.EMOTION,
                value=None,
                attributes={"placeholder": True, "emotion": None, "intensity": None},
            )
        )
        graph.add_edge(token_node.id, emotion_node.id, RelationType.HAS_EMOTION)

        return token_node.id

    def _add_punctuation(
        self, graph: TokenGraph, sentence_id: str, mark: str
    ) -> str | None:
        """Add a PAUSE node for a recognized punctuation ``mark``.

        Returns the node id, or ``None`` if the mark is unrecognized.
        """
        if not is_punctuation(mark):
            return None
        signal = map_punctuation(mark)
        assert signal is not None  # guaranteed by is_punctuation
        pause_node = graph.add_node(
            self._new_node(
                NodeType.PAUSE,
                value=signal.value,
                attributes={"mark": mark, "signal": signal.value},
            )
        )
        graph.add_edge(sentence_id, pause_node.id, RelationType.CONTAINS)
        return pause_node.id

    # -- Helpers ------------------------------------------------------------ #
    def _new_node(
        self,
        node_type: NodeType,
        *,
        value: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Node:
        """Create a node with a deterministic, type-scoped identifier."""
        count = self._counter.get(node_type, 0)
        self._counter[node_type] = count + 1
        node_id = f"{node_type.value}_{count}"
        return Node(
            id=node_id,
            type=node_type,
            value=value,
            attributes=attributes or {},
        )


_default_builder: Final[TokenGraphBuilder] = TokenGraphBuilder()


def build_token_graph(text: str) -> TokenGraph:
    """Build a :class:`TokenGraph` using the module-level default builder."""
    return _default_builder.build(text)
