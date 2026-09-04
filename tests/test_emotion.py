"""Deterministic unit tests for the Emotional Vector Engine (Phase 1B).

The engine is required to be fully deterministic and free of machine learning,
so tests assert exact or thresholded behaviour that must hold on every run.
"""

from __future__ import annotations

import pytest

from ava_voice.emotion.classifier import EmotionClassifier, classify
from ava_voice.emotion.contextual_analyzer import ContextualAnalyzer
from ava_voice.emotion.emotion_fusion import EmotionEngine, analyze_once
from ava_voice.emotion.emotion_memory import EmotionMemory
from ava_voice.emotion.emotion_schema import (
    NEUTRAL,
    PRIMARY_DIMENSIONS,
    SECONDARY_DIMENSIONS,
    EmotionalVector,
    PrimaryEmotion,
)
from ava_voice.emotion.intensity_estimator import IntensityEstimator, estimate_intensity
from ava_voice.emotion.valence_arousal import ValenceArousalMapper
from ava_voice.linguistics.token_graph import NodeType, build_token_graph


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def test_vector_clamps_out_of_range_fields() -> None:
    v = EmotionalVector(joy=5.0, valence=-9.0, arousal=2.0, dominance=-1.0)
    assert v.joy == 1.0
    assert v.valence == -1.0
    assert v.arousal == 1.0
    assert v.dominance == 0.0


def test_vector_dominant_emotion_and_ties() -> None:
    assert EmotionalVector(joy=0.9).dominant_emotion is PrimaryEmotion.JOY
    # Tie resolves to the first-declared dimension (joy before sadness).
    tie = EmotionalVector(joy=0.5, sadness=0.5)
    assert tie.dominant_emotion is PrimaryEmotion.JOY
    assert NEUTRAL.dominant_emotion is None


def test_vector_serialization_roundtrip_is_stable() -> None:
    v = EmotionalVector(joy=0.4, fear=0.2, valence=0.3, arousal=0.5, intensity=0.6)
    data = v.to_dict()
    assert list(data.keys()) == [*PRIMARY_DIMENSIONS, *SECONDARY_DIMENSIONS]
    assert EmotionalVector.from_dict(data).to_dict() == data


def test_vector_blend_is_convex() -> None:
    a = EmotionalVector(joy=1.0, valence=1.0)
    b = EmotionalVector(joy=0.0, valence=-1.0)
    mid = a.blend(b, 0.5)
    assert mid.joy == 0.5
    assert mid.valence == 0.0


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
def test_classifier_joy_high() -> None:
    result = classify("خیلی خوشحالم")
    assert result.dominant == "joy"
    assert result.scores["joy"] > 0.8  # intensifier boosts the joy weight
    assert result.polarity == 1.0


def test_classifier_fear_high_zwnj_insensitive() -> None:
    spaced = classify("می\u200cترسم").scores["fear"]
    joined = classify("میترسم").scores["fear"]
    assert spaced == joined > 0.8


def test_classifier_negation_suppresses_and_flips() -> None:
    result = classify("خوشحال نیستم")
    assert result.scores["joy"] < 0.1  # joy suppressed by negation
    assert result.scores["sadness"] > 0.0  # energy redirected to counter channel
    assert result.dominant == "sadness"


def test_classifier_phrase_beats_unigram() -> None:
    result = classify("با کمال احترام")
    assert result.dominant == "formality"
    assert "با کمال احترام" in result.matched_terms


def test_classifier_is_deterministic() -> None:
    text = "خیلی عصبانیم و ناراحت"
    assert classify(text).to_dict() == classify(text).to_dict()


@pytest.mark.parametrize(
    ("arabic", "persian"),
    [
        ("خيلي خوشحالم", "خیلی خوشحالم"),
        ("مي‌ترسم", "می‌ترسم"),
        ("خوشحال نيستم", "خوشحال نیستم"),
        ("واقعا عالي بود", "واقعا عالی بود"),
    ],
)
def test_classifier_is_insensitive_to_arabic_keyboard(
    arabic: str, persian: str
) -> None:
    """Arabic ي/ك must score identically to Persian ی/ک.

    32% of the lexicon contains ی or ک, and classify() previously did not
    normalize, so all of it was unreachable for anyone typing on an Arabic
    keyboard layout.
    """
    assert classify(arabic).scores == classify(persian).scores
    assert classify(arabic).dominant == classify(persian).dominant


def test_classifier_is_insensitive_to_optional_harakat() -> None:
    # Harakat are optional in Persian writing; both spellings are the same word.
    assert classify("لطفاً کمکم کن").scores == classify("لطفا کمکم کن").scores
    assert classify("لطفا کمکم کن").dominant == "formality"
    assert classify("احتراما").dominant == "formality"


def test_classifier_binds_mi_prefix_written_with_a_space() -> None:
    # The normalizer binds "می" to its verb with a ZWNJ, reaching "میترسم".
    assert classify("خیلی می ترسم").dominant == "fear"
    assert classify("خيلي مي ترسم").dominant == "fear"


def test_classifier_injected_lexicon_matches_any_spelling() -> None:
    clf = EmotionClassifier(lexicon={"مي‌خندم": {"humor": 0.9}})
    assert clf.classify("مي‌خندم").dominant == "humor"
    assert clf.classify("می‌خندم").dominant == "humor"
    assert clf.classify("میخندم").dominant == "humor"


# --------------------------------------------------------------------------- #
# Valence / arousal / dominance
# --------------------------------------------------------------------------- #
def test_vad_joy_is_positive_and_activated() -> None:
    vad = ValenceArousalMapper().map({"joy": 0.9})
    assert vad.valence > 0.5
    assert vad.arousal > 0.5


def test_vad_sadness_is_negative_low_dominance() -> None:
    vad = ValenceArousalMapper().map({"sadness": 0.8})
    assert vad.valence < 0.0
    assert vad.dominance < 0.5


def test_vad_empty_scores_returns_neutral_resting() -> None:
    vad = ValenceArousalMapper().map({})
    assert vad.valence == 0.0
    assert 0.0 <= vad.arousal <= 0.2
    assert vad.dominance == 0.5


# --------------------------------------------------------------------------- #
# Intensity estimator
# --------------------------------------------------------------------------- #
def test_intensity_elongation_and_punctuation_high() -> None:
    assert estimate_intensity("عاااالی!!!").intensity > 0.7


def test_intensity_plain_word_low() -> None:
    assert estimate_intensity("خوبه").intensity < 0.2


def test_intensity_word_repetition() -> None:
    assert estimate_intensity("برو برو برو").intensity > 0.5


def test_intensity_lexical_strength_contributes() -> None:
    weak = estimate_intensity("سلام", lexical_strength=0.0).intensity
    strong = estimate_intensity("سلام", lexical_strength=0.9).intensity
    assert strong > weak


def test_intensity_decomposition_is_serializable() -> None:
    data = IntensityEstimator().estimate("عاااالی!!!").to_dict()
    assert set(data) == {
        "intensity",
        "punctuation",
        "repetition",
        "elongation",
        "emphasis",
        "lexical",
    }


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
def test_memory_window_evicts_oldest() -> None:
    mem = EmotionMemory(window=2)
    mem.push(EmotionalVector(joy=0.1))
    mem.push(EmotionalVector(joy=0.2))
    mem.push(EmotionalVector(joy=0.3))
    assert len(mem.states()) == 2
    assert mem.states()[0].vector.joy == 0.2


def test_memory_trend_weighted_toward_recent() -> None:
    mem = EmotionMemory(window=3, decay=0.5)
    mem.push(EmotionalVector(joy=0.0))
    mem.push(EmotionalVector(joy=1.0))
    # Recency-weighted mean must lean toward the most recent (1.0) state.
    assert mem.trend().joy > 0.5


def test_memory_momentum_direction() -> None:
    mem = EmotionMemory()
    mem.push(EmotionalVector(valence=-0.5))
    mem.push(EmotionalVector(valence=0.5))
    assert mem.momentum().valence_slope > 0.0
    assert mem.momentum().magnitude > 0.0


def test_memory_empty_defaults() -> None:
    mem = EmotionMemory()
    assert mem.is_empty
    assert mem.trend().is_neutral
    assert mem.momentum().magnitude == 0.0


# --------------------------------------------------------------------------- #
# Contextual analyzer
# --------------------------------------------------------------------------- #
def test_context_no_history_is_passthrough() -> None:
    current = EmotionalVector(joy=0.7, valence=0.7)
    out = ContextualAnalyzer().adjust(current, EmotionMemory())
    assert out.vector == current
    assert out.shift_detected is False
    assert out.continuity is False


def test_context_detects_shift_on_valence_flip() -> None:
    mem = EmotionMemory()
    mem.push(EmotionalVector(joy=0.8, valence=0.8, arousal=0.6))
    out = ContextualAnalyzer().adjust(
        EmotionalVector(anger=0.8, valence=-0.6, arousal=0.8), mem
    )
    assert out.shift_detected is True
    # A shift smooths only lightly, so the new (negative) valence survives.
    assert out.vector.valence < 0.0


def test_context_continuity_smooths_toward_trend() -> None:
    mem = EmotionMemory()
    mem.push(EmotionalVector(joy=0.6, valence=0.6, arousal=0.5))
    out = ContextualAnalyzer().adjust(
        EmotionalVector(joy=0.7, valence=0.65, arousal=0.5), mem
    )
    assert out.continuity is True
    assert out.shift_detected is False


# --------------------------------------------------------------------------- #
# Fusion engine
# --------------------------------------------------------------------------- #
def test_engine_joy_sentence() -> None:
    analysis = analyze_once("خیلی خوشحالم!")
    assert analysis.vector.dominant_emotion is PrimaryEmotion.JOY
    assert analysis.vector.valence > 0.5
    assert analysis.vector.intensity > 0.4


def test_engine_fear_sentence() -> None:
    analysis = analyze_once("می‌ترسم")
    assert analysis.vector.dominant_emotion is PrimaryEmotion.FEAR
    assert analysis.vector.valence < 0.0


def test_engine_is_deterministic() -> None:
    assert analyze_once("عصبانیم!!!").to_dict() == analyze_once("عصبانیم!!!").to_dict()


def test_engine_memory_accumulates_across_calls() -> None:
    engine = EmotionEngine()
    engine.analyze("خوشحالم")
    engine.analyze("خیلی خوشحالم!")
    assert len(engine.memory.states()) == 2
    engine.reset()
    assert engine.memory.is_empty


# --------------------------------------------------------------------------- #
# Token-graph integration
# --------------------------------------------------------------------------- #
def test_engine_annotates_token_graph() -> None:
    graph = build_token_graph("خیلی خوشحالم!")
    EmotionEngine().analyze_graph(graph)

    sentence = graph.nodes_of_type(NodeType.SENTENCE)[0]
    assert sentence.attributes["dominant_emotion"] == "joy"
    assert set(sentence.attributes["emotion"]) == set(
        [*PRIMARY_DIMENSIONS, *SECONDARY_DIMENSIONS]
    )

    # The reserved EMOTION placeholders are now populated (no longer empty).
    emotions = graph.nodes_of_type(NodeType.EMOTION)
    assert any(n.attributes["placeholder"] is False for n in emotions)
    joy_node = [n for n in emotions if n.value == "joy"]
    assert joy_node and joy_node[0].attributes["score"] > 0.0


def test_engine_graph_matches_text_analysis() -> None:
    # Analyzing via the graph must agree with analyzing the raw text directly.
    text = "می‌ترسم!"
    graph = build_token_graph(text)
    from_graph = EmotionEngine().analyze_graph(graph).vector.to_dict()
    from_text = EmotionEngine().analyze(text).vector.to_dict()
    assert from_graph == from_text
