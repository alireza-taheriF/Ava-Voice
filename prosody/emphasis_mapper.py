"""Emphasis & elongation planning (Phase 1C).

Emphasis is *selective prominence*: not every word is spoken with equal force.
This module decides, per token, whether it should be **emphasized** (extra
prominence / pitch accent), how strong that boost is, and whether its stressed
vowel should be **elongated** for expressive effect (کِشش, e.g. ``خیییلی``).

Inputs
------
* **Token graph** — the linguistics layer already computed per-token stress maps
  (:class:`~ava_voice.linguistics.stress_detector.StressMap`) exposing
  ``emphasis_positions`` (syllables boosted because a ``!`` followed) and
  ``elongation_positions`` (long vowels / repeated-letter candidates). These are
  read via the ``HAS_STRESS`` edges. When the Phase 1B emotion engine has
  annotated the graph, per-token ``EMOTION`` nodes carry an emotion ``score``
  read via the ``HAS_EMOTION`` edges.
* **Emotional intensity** — a globally intense, aroused, dominant delivery
  raises overall emphasis strength and the elongation factor.

Decision rules (deterministic)
------------------------------
A token is emphasized when any of the following hold:

1. its stress map already flags an emphatic syllable (punctuation ``!`` cue),
2. it carries emotional weight (annotated ``EMOTION`` score ≥ threshold), or
3. the utterance is highly intense and the token is the final content word
   (Persian frequently lands nuclear emphasis on the sentence-final content
   word).

Elongation is applied to tokens whose stress map lists elongation candidates, or
to emphasized tokens under high intensity. Everything is a pure function of
``(graph, vector)`` — no randomness, no ML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import EmotionalVector, clamp
from ava_voice.linguistics.token_graph import NodeType, RelationType, TokenGraph

_logger = get_logger("prosody.emphasis")

# Annotated emotion score at/above which a token is treated as emotion-bearing.
_EMOTION_EMPHASIS_THRESHOLD: Final[float] = 0.50

# Global intensity at/above which the sentence-final content word is emphasized
# and emphasized tokens become elongation candidates.
_HIGH_INTENSITY: Final[float] = 0.65

# Expressive intensity below which no vowel is drawn out at all. Elongation
# (کِشش) is an *affective* device, so a lengthenable syllable is not sufficient
# reason to lengthen it — most Persian words contain a long vowel, which made
# elongation the default state for nearly every token.
_ELONGATION_INTENSITY_FLOOR: Final[float] = 0.35

# Weights forming the overall emphasis strength from the emotional drives.
_W_INTENSITY: Final[float] = 0.60
_W_AROUSAL: Final[float] = 0.20
_W_DOMINANCE: Final[float] = 0.20

# Extra boost fraction granted to emotion-bearing tokens over baseline emphasis.
_EMOTION_BOOST_BONUS: Final[float] = 0.15


@dataclass(frozen=True)
class TokenEmphasis:
    """Per-token emphasis / elongation decision.

    Attributes
    ----------
    node_id:
        Id of the ``TOKEN`` node.
    surface:
        The token's surface text (for readability / debugging).
    emphasized:
        Whether the token receives extra prominence.
    boost:
        The prominence boost applied, in ``[0, 1]`` (0 when not emphasized).
    elongate:
        Whether expressive vowel elongation applies to this token.
    """

    node_id: str
    surface: str | None
    emphasized: bool
    boost: float
    elongate: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "node_id": self.node_id,
            "surface": self.surface,
            "emphasized": self.emphasized,
            "boost": self.boost,
            "elongate": self.elongate,
        }


@dataclass(frozen=True)
class EmphasisPlan:
    """The complete emphasis plan for a sentence.

    Attributes
    ----------
    tokens:
        Ordered per-token emphasis decisions (spoken-timeline order).
    emphasis_strength:
        Global emphasis intensity in ``[0, 1]`` (surfaced on the profile).
    elongation_factor:
        Global expressive elongation multiplier in ``[1.0, 2.5]``.
    """

    tokens: tuple[TokenEmphasis, ...] = field(default_factory=tuple)
    emphasis_strength: float = 0.0
    elongation_factor: float = 1.0

    @property
    def emphasized_ids(self) -> tuple[str, ...]:
        """Return the ids of emphasized tokens."""
        return tuple(t.node_id for t in self.tokens if t.emphasized)

    @property
    def elongation_ids(self) -> tuple[str, ...]:
        """Return the ids of tokens marked for elongation."""
        return tuple(t.node_id for t in self.tokens if t.elongate)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "tokens": [t.to_dict() for t in self.tokens],
            "emphasis_strength": self.emphasis_strength,
            "elongation_factor": self.elongation_factor,
        }


class EmphasisMapper:
    """Deterministic planner of token emphasis and elongation."""

    def map(self, graph: TokenGraph, vector: EmotionalVector) -> EmphasisPlan:
        """Build an :class:`EmphasisPlan` from a token ``graph`` and emotion ``vector``."""
        emphasis_strength = clamp(
            _W_INTENSITY * vector.intensity
            + _W_AROUSAL * vector.arousal
            + _W_DOMINANCE * vector.dominance,
            0.0,
            1.0,
        )
        # Sadness draws vowels out (mournful lengthening); intensity widens all.
        elongation_factor = clamp(
            1.0 + 0.6 * vector.intensity + 0.2 * vector.sadness,
            1.0,
            2.5,
        )

        token_nodes = graph.nodes_of_type(NodeType.TOKEN)
        stress_by_token = self._targets_by_source(graph, RelationType.HAS_STRESS)
        emotion_by_token = self._targets_by_source(graph, RelationType.HAS_EMOTION)
        node_by_id = {n.id: n for n in graph.nodes}
        last_index = len(token_nodes) - 1
        high_intensity = vector.intensity >= _HIGH_INTENSITY
        # Affect gate: below the floor nothing is drawn out, however many
        # lengthenable syllables the sentence happens to contain.
        expressive = vector.intensity >= _ELONGATION_INTENSITY_FLOOR

        decisions: list[TokenEmphasis] = []
        for index, token in enumerate(token_nodes):
            stress_node = node_by_id.get(stress_by_token.get(token.id, ""))
            has_stress_emphasis = self._has_stress_emphasis(stress_node)
            has_elongation_candidate = self._has_elongation_candidate(stress_node)
            has_written_drawl = self._has_written_drawl(stress_node)
            emotion_score = self._emotion_score(
                node_by_id.get(emotion_by_token.get(token.id, ""))
            )

            emphasized = (
                has_stress_emphasis
                or emotion_score >= _EMOTION_EMPHASIS_THRESHOLD
                or (high_intensity and index == last_index)
            )
            boost = 0.0
            if emphasized:
                bonus = (
                    _EMOTION_BOOST_BONUS
                    if emotion_score >= _EMOTION_EMPHASIS_THRESHOLD
                    else 0.0
                )
                boost = clamp(emphasis_strength + bonus, 0.0, 1.0)

            # A drawl the writer spelled out (سلاااام) is an explicit request and
            # is always honoured. Otherwise elongation needs both a lengthenable
            # syllable and genuine expressive intensity.
            elongate = has_written_drawl or (
                expressive
                and (has_elongation_candidate or (emphasized and high_intensity))
            )

            decisions.append(
                TokenEmphasis(
                    node_id=token.id,
                    surface=token.value,
                    emphasized=emphasized,
                    boost=boost,
                    elongate=elongate,
                )
            )

        return EmphasisPlan(
            tokens=tuple(decisions),
            emphasis_strength=emphasis_strength,
            elongation_factor=elongation_factor,
        )

    # -- Graph traversal helpers ------------------------------------------- #
    @staticmethod
    def _targets_by_source(
        graph: TokenGraph, relation: RelationType
    ) -> dict[str, str]:
        """Map ``source_id -> target_id`` for edges of the given relation.

        Each token has at most one stress node and one emotion node, so a plain
        dict is sufficient and keeps the association robust to node ordering.
        """
        mapping: dict[str, str] = {}
        for edge in graph.edges:
            if edge.relation is relation:
                mapping[edge.source] = edge.target
        return mapping

    @staticmethod
    def _has_stress_emphasis(stress_node: object) -> bool:
        """Return ``True`` if the stress node lists emphatic syllable positions."""
        if stress_node is None:
            return False
        positions = getattr(stress_node, "attributes", {}).get(
            "emphasis_positions", []
        )
        return bool(positions)

    @staticmethod
    def _has_elongation_candidate(stress_node: object) -> bool:
        """Return ``True`` if the stress node lists elongation candidates."""
        if stress_node is None:
            return False
        positions = getattr(stress_node, "attributes", {}).get(
            "elongation_positions", []
        )
        return bool(positions)

    @staticmethod
    def _has_written_drawl(stress_node: object) -> bool:
        """Return ``True`` if the writer spelled the drawl out (``سلاااام``)."""
        if stress_node is None:
            return False
        return bool(
            getattr(stress_node, "attributes", {}).get("has_written_drawl", False)
        )

    @staticmethod
    def _emotion_score(emotion_node: object) -> float:
        """Return the annotated emotion score for a token (0 if none/placeholder)."""
        if emotion_node is None:
            return 0.0
        attributes = getattr(emotion_node, "attributes", {})
        if attributes.get("placeholder", True):
            return 0.0
        score = attributes.get("score", 0.0)
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0


_default_mapper: Final[EmphasisMapper] = EmphasisMapper()


def map_emphasis(graph: TokenGraph, vector: EmotionalVector) -> EmphasisPlan:
    """Plan emphasis using the module-level default mapper."""
    return _default_mapper.map(graph, vector)
