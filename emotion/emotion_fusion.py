"""Emotional vector fusion & engine orchestration (Phase 1B).

This is the top of the Emotional Vector Engine — the analogue of
:class:`~ava_voice.linguistics.token_graph.TokenGraphBuilder` for affect. It
wires the individual stages into one deterministic pipeline and emits the final
:class:`~ava_voice.emotion.emotion_schema.EmotionalVector` that downstream
prosody/TTS conditioning consumes.

Pipeline (each stage is a pure, single-responsibility module)
-------------------------------------------------------------
``classify (lexical) → map (valence/arousal/dominance) → estimate (intensity)
→ assemble base vector → contextual adjust (memory) → commit to memory``

Fusion recipe (deterministic, closed-form)
------------------------------------------
1. **Primary activations** — each raw lexical score ``s`` is passed through a
   saturating transfer ``a = s / (s + K)`` into ``[0, 1)``. This gives smooth,
   bounded, monotonic activations without any learned parameters.
2. **Secondary coordinates** — valence and dominance come straight from the VAD
   mapper; arousal is a fixed blend of the VAD arousal and the estimated
   intensity (loud, elongated, exclaimed speech is physiologically higher
   arousal). Intensity is carried through verbatim.
3. **Contextual shaping** — the assembled vector is smoothed toward / away from
   the recent trend by the contextual analyzer (continuity vs shift).
4. **Memory commit** — the adjusted vector is pushed into the rolling memory so
   the *next* sentence sees it.

Integration with the linguistic token graph
--------------------------------------------
:meth:`EmotionEngine.analyze_graph` consumes a
:class:`~ava_voice.linguistics.token_graph.TokenGraph`: it reads the raw
sentence text from the ``SENTENCE`` node and the canonical words from the
``TOKEN`` nodes, computes the sentence emotion, then *fills the reserved
``EMOTION`` placeholder nodes* (created empty by Phase 1A) with per-token
affect and records the sentence-level vector on the ``SENTENCE`` node. No new
node types are introduced — the engine simply populates the attachment points
the linguistics layer left for it.

Examples
--------
>>> engine = EmotionEngine()
>>> analysis = engine.analyze("خیلی خوشحالم!")
>>> analysis.vector.dominant_emotion.value
'joy'
>>> analysis.vector.intensity > 0.4
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ava_voice.core.logger import get_logger
from ava_voice.emotion.classifier import EmotionClassifier, LexicalEmotion
from ava_voice.emotion.contextual_analyzer import ContextAdjustment, ContextualAnalyzer
from ava_voice.emotion.emotion_memory import EmotionMemory
from ava_voice.emotion.emotion_schema import (
    PRIMARY_DIMENSIONS,
    EmotionalVector,
    clamp,
)
from ava_voice.emotion.intensity_estimator import IntensityEstimate, IntensityEstimator
from ava_voice.emotion.valence_arousal import ValenceArousalMapper, VADCoordinate
from ava_voice.linguistics.token_graph import (
    NodeType,
    RelationType,
    TokenGraph,
)

_logger = get_logger("emotion.fusion")

# Saturation constant for the primary-activation transfer ``s / (s + K)``.
# Smaller K -> activations rise faster with lexical score.
_ACTIVATION_K: Final[float] = 0.5

# Blend of VAD-derived arousal vs estimated intensity when forming final arousal.
_AROUSAL_FROM_VAD: Final[float] = 0.6
_AROUSAL_FROM_INTENSITY: Final[float] = 0.4


@dataclass(frozen=True)
class EmotionAnalysis:
    """The complete, auditable result of analyzing one sentence.

    Attributes
    ----------
    vector:
        The final, context-adjusted :class:`EmotionalVector` (the deliverable).
    base_vector:
        The vector *before* contextual shaping — useful for debugging and for
        callers that want the raw per-sentence read without history.
    lexical:
        The :class:`LexicalEmotion` from the classifier.
    vad:
        The :class:`VADCoordinate` from the valence/arousal mapper.
    intensity:
        The :class:`IntensityEstimate` from the intensity stage.
    context:
        The :class:`ContextAdjustment` describing the temporal shaping applied.
    """

    vector: EmotionalVector
    base_vector: EmotionalVector
    lexical: LexicalEmotion
    vad: VADCoordinate
    intensity: IntensityEstimate
    context: ContextAdjustment

    def to_dict(self) -> dict[str, object]:
        """Return a fully JSON-serializable representation of the analysis."""
        return {
            "vector": self.vector.to_dict(),
            "base_vector": self.base_vector.to_dict(),
            "lexical": self.lexical.to_dict(),
            "vad": self.vad.to_dict(),
            "intensity": self.intensity.to_dict(),
            "context": self.context.to_dict(),
        }


class EmotionEngine:
    """Deterministic orchestration of the full Emotional Vector Engine.

    The engine owns a short-term :class:`EmotionMemory`, so a single instance
    models one *conversation* / *utterance stream*: successive
    :meth:`analyze` calls influence one another through context. For a stateless,
    single-shot read use :func:`analyze_once` or construct a fresh engine.

    All collaborators are injected (and default to their module singletons),
    keeping the orchestration thin and each stage independently testable.
    """

    def __init__(
        self,
        classifier: EmotionClassifier | None = None,
        mapper: ValenceArousalMapper | None = None,
        estimator: IntensityEstimator | None = None,
        analyzer: ContextualAnalyzer | None = None,
        memory: EmotionMemory | None = None,
    ) -> None:
        self._classifier = classifier or EmotionClassifier()
        self._mapper = mapper or ValenceArousalMapper()
        self._estimator = estimator or IntensityEstimator()
        self._analyzer = analyzer or ContextualAnalyzer()
        self._memory = memory or EmotionMemory()

    # -- Introspection ------------------------------------------------------ #
    @property
    def memory(self) -> EmotionMemory:
        """Return the engine's rolling emotional memory (read/inspect)."""
        return self._memory

    def reset(self) -> None:
        """Clear conversational memory (start a fresh utterance stream)."""
        self._memory.reset()

    # -- Core analysis ------------------------------------------------------ #
    def analyze(
        self, text: str, *, tokens: list[str] | None = None, update_memory: bool = True
    ) -> EmotionAnalysis:
        """Analyze ``text`` and return its :class:`EmotionAnalysis`.

        Parameters
        ----------
        text:
            The *raw* sentence text. Passed to the intensity estimator verbatim
            (before normalization) so punctuation repetition is preserved.
        tokens:
            Optional pre-tokenized canonical words (e.g. from a token graph). When
            omitted, the classifier tokenizes ``text`` itself.
        update_memory:
            When ``True`` (default) the adjusted vector is committed to memory so
            it influences subsequent sentences.
        """
        lexical = (
            self._classifier.classify_tokens(tokens)
            if tokens is not None
            else self._classifier.classify(text)
        )
        vad = self._mapper.map(lexical.scores)
        intensity = self._estimator.estimate(
            text, lexical_strength=lexical.matched_lexical_strength
        )

        base_vector = self._assemble(lexical, vad, intensity)
        adjustment = self._analyzer.adjust(base_vector, self._memory)

        if update_memory:
            self._memory.push(adjustment.vector)

        return EmotionAnalysis(
            vector=adjustment.vector,
            base_vector=base_vector,
            lexical=lexical,
            vad=vad,
            intensity=intensity,
            context=adjustment,
        )

    # -- Token-graph integration ------------------------------------------- #
    def analyze_graph(
        self, graph: TokenGraph, *, update_memory: bool = True, annotate: bool = True
    ) -> EmotionAnalysis:
        """Analyze a :class:`TokenGraph` and (optionally) annotate it in place.

        The raw sentence text is taken from the ``SENTENCE`` node value and the
        canonical words from the ``TOKEN`` nodes. When ``annotate`` is ``True``
        the reserved ``EMOTION`` placeholder nodes are populated with per-token
        affect and the ``SENTENCE`` node receives the sentence-level vector.
        """
        sentence_nodes = graph.nodes_of_type(NodeType.SENTENCE)
        raw_text = sentence_nodes[0].value or "" if sentence_nodes else ""
        token_nodes = graph.nodes_of_type(NodeType.TOKEN)
        tokens = [str(node.attributes.get("canonical", node.value)) for node in token_nodes]

        analysis = self.analyze(raw_text, tokens=tokens, update_memory=update_memory)

        if annotate:
            self._annotate_graph(graph, analysis)
        return analysis

    def _annotate_graph(self, graph: TokenGraph, analysis: EmotionAnalysis) -> None:
        """Fill EMOTION placeholder nodes and record the sentence vector."""
        # Sentence-level vector onto the SENTENCE node.
        for sentence in graph.nodes_of_type(NodeType.SENTENCE):
            sentence.attributes["emotion"] = analysis.vector.to_dict()
            dominant = analysis.vector.dominant_emotion
            sentence.attributes["dominant_emotion"] = (
                dominant.value if dominant is not None else None
            )

        sentence_intensity = analysis.vector.intensity
        emotion_by_token = self._emotion_nodes_by_token(graph)
        for token_node, emotion_node in emotion_by_token:
            canonical = str(token_node.attributes.get("canonical", token_node.value))
            local = self._classifier.classify_tokens([canonical])
            dominant = local.dominant
            emotion_node.value = dominant
            emotion_node.attributes.update(
                {
                    "placeholder": False,
                    "emotion": dominant,
                    "score": (
                        clamp(local.scores.get(dominant, 0.0), 0.0, 1.0)
                        if dominant is not None
                        else 0.0
                    ),
                    "intensity": sentence_intensity,
                }
            )

    @staticmethod
    def _emotion_nodes_by_token(graph: TokenGraph):
        """Yield ``(token_node, emotion_node)`` pairs following HAS_EMOTION edges.

        Using the edges (rather than positional zipping) keeps the association
        correct even if node ordering changes in a future builder revision.
        """
        node_by_id = {node.id: node for node in graph.nodes}
        pairs = []
        for edge in graph.edges:
            if edge.relation is RelationType.HAS_EMOTION:
                token_node = node_by_id.get(edge.source)
                emotion_node = node_by_id.get(edge.target)
                if token_node is not None and emotion_node is not None:
                    pairs.append((token_node, emotion_node))
        return pairs

    # -- Fusion ------------------------------------------------------------- #
    def _assemble(
        self,
        lexical: LexicalEmotion,
        vad: VADCoordinate,
        intensity: IntensityEstimate,
    ) -> EmotionalVector:
        """Assemble the base :class:`EmotionalVector` from the three stages."""
        primaries = {
            emotion: self._activation(lexical.scores.get(emotion, 0.0))
            for emotion in PRIMARY_DIMENSIONS
        }
        arousal = clamp(
            _AROUSAL_FROM_VAD * vad.arousal
            + _AROUSAL_FROM_INTENSITY * intensity.intensity,
            0.0,
            1.0,
        )
        return EmotionalVector(
            **primaries,
            valence=vad.valence,
            arousal=arousal,
            dominance=vad.dominance,
            intensity=intensity.intensity,
        )

    @staticmethod
    def _activation(score: float) -> float:
        """Saturating transfer ``s / (s + K)`` mapping a raw score into ``[0, 1)``."""
        if score <= 0.0:
            return 0.0
        return score / (score + _ACTIVATION_K)


def analyze_once(text: str) -> EmotionAnalysis:
    """Analyze ``text`` with a fresh, memory-less engine (stateless convenience).

    Each call constructs a new :class:`EmotionEngine`, so there is no
    cross-call context. Use an explicit :class:`EmotionEngine` instance when you
    want conversational continuity.
    """
    return EmotionEngine().analyze(text)
