"""Deterministic unit tests for the Prosody Intelligence Layer (Phase 1C).

The layer must be fully deterministic and free of machine learning, so tests
assert exact or thresholded behaviour that must hold on every run.
"""

from __future__ import annotations

from ava_voice.emotion.emotion_schema import EmotionalVector
from ava_voice.linguistics.token_graph import NodeType, build_token_graph
from ava_voice.prosody.contour_generator import ContourGenerator, generate_contour
from ava_voice.prosody.emphasis_mapper import EmphasisMapper
from ava_voice.prosody.energy_mapper import map_energy
from ava_voice.prosody.pause_mapper import map_pauses
from ava_voice.prosody.pitch_mapper import PitchTendency, map_pitch
from ava_voice.prosody.prosody_engine import ProsodyEngine, generate_prosody
from ava_voice.prosody.prosody_schema import ContourShape, ProsodyProfile
from ava_voice.prosody.tempo_mapper import TempoRegularity, map_tempo


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def test_profile_clamps_out_of_range() -> None:
    p = ProsodyProfile(pitch_base=99.0, speaking_rate=9.0, elongation_factor=0.1)
    assert p.pitch_base == 12.0
    assert p.speaking_rate == 2.0
    assert p.elongation_factor == 1.0


def test_profile_roundtrip_is_stable() -> None:
    p = ProsodyProfile(
        pitch_base=3.0, speaking_rate=1.3, contour_shape=ContourShape.RISING
    )
    data = p.to_dict()
    assert data["contour_shape"] == "rising"
    assert ProsodyProfile.from_dict(data) == p


def test_profile_from_dict_accepts_string_contour() -> None:
    p = ProsodyProfile.from_dict({"contour_shape": "broken", "energy_level": 0.9})
    assert p.contour_shape is ContourShape.BROKEN
    assert p.energy_level == 0.9


# --------------------------------------------------------------------------- #
# Pitch mapper
# --------------------------------------------------------------------------- #
def test_pitch_joy_higher_than_sadness() -> None:
    joy = map_pitch(EmotionalVector(joy=0.9, valence=0.8, arousal=0.7, intensity=0.8))
    sad = map_pitch(
        EmotionalVector(sadness=0.9, valence=-0.7, arousal=0.3, intensity=0.6)
    )
    assert joy.base > 0.0
    assert sad.base < 0.0
    assert joy.base > sad.base


def test_pitch_anger_and_fear_are_unstable_and_variable() -> None:
    anger = map_pitch(EmotionalVector(anger=0.9, arousal=0.85, intensity=0.8))
    fear = map_pitch(EmotionalVector(fear=0.9, arousal=0.8, intensity=0.7))
    assert anger.tendency is PitchTendency.UNSTABLE
    assert fear.tendency is PitchTendency.UNSTABLE
    assert anger.variance > 0.5
    assert fear.variance > 0.5


def test_pitch_formality_compresses_variance() -> None:
    plain = map_pitch(EmotionalVector(joy=0.6, arousal=0.6, intensity=0.5))
    formal = map_pitch(
        EmotionalVector(joy=0.6, formality=0.9, arousal=0.6, intensity=0.5)
    )
    assert formal.variance < plain.variance


# --------------------------------------------------------------------------- #
# Tempo mapper
# --------------------------------------------------------------------------- #
def test_tempo_joy_faster_sadness_slower() -> None:
    joy = map_tempo(EmotionalVector(joy=0.9, arousal=0.7, intensity=0.7))
    sad = map_tempo(EmotionalVector(sadness=0.9, arousal=0.3, intensity=0.6))
    assert joy.rate > 1.0
    assert sad.rate < 1.0


def test_tempo_fear_is_irregular_calm_is_stable() -> None:
    fear = map_tempo(EmotionalVector(fear=0.9, arousal=0.8, intensity=0.7))
    calm = map_tempo(EmotionalVector(calm=0.9, arousal=0.1, intensity=0.2))
    assert fear.regularity is TempoRegularity.IRREGULAR
    assert calm.regularity is TempoRegularity.STABLE


# --------------------------------------------------------------------------- #
# Energy mapper
# --------------------------------------------------------------------------- #
def test_energy_anger_high_calm_low() -> None:
    anger = map_energy(EmotionalVector(anger=0.9, arousal=0.85, intensity=0.8))
    calm = map_energy(EmotionalVector(calm=0.9, arousal=0.1, intensity=0.2))
    joy = map_energy(EmotionalVector(joy=0.8, arousal=0.6, intensity=0.6))
    assert anger.level > 0.7
    assert calm.level < 0.4
    assert 0.5 < joy.level < anger.level  # joy is medium-high


# --------------------------------------------------------------------------- #
# Pause mapper
# --------------------------------------------------------------------------- #
def test_pause_reads_punctuation_nodes() -> None:
    graph = build_token_graph("سلام. خوبی؟")
    plan = map_pauses(graph, EmotionalVector())
    assert len(plan.placements) == 2
    signals = {p.signal for p in plan.placements}
    assert "pause_medium" in signals
    assert "rising_pitch" in signals


def test_pause_sadness_lengthens_vs_anger() -> None:
    graph = build_token_graph("رفت. آمد.")
    sad = map_pauses(graph, EmotionalVector(sadness=0.9)).duration_scale
    angry = map_pauses(graph, EmotionalVector(anger=0.9)).duration_scale
    assert sad > 1.0
    assert angry < 1.0
    assert sad > angry


def test_pause_density_bounded() -> None:
    graph = build_token_graph("سلام دنیا.")
    plan = map_pauses(graph, EmotionalVector(calm=0.8))
    assert 0.0 <= plan.density <= 1.0


# --------------------------------------------------------------------------- #
# Emphasis mapper
# --------------------------------------------------------------------------- #
def test_emphasis_bang_marks_token() -> None:
    # A "!" upgrades the preceding token's stress to emphatic in Phase 1A.
    graph = build_token_graph("عالی!")
    plan = EmphasisMapper().map(graph, EmotionalVector(joy=0.8, intensity=0.8))
    assert plan.emphasized_ids  # at least one emphasized token
    assert plan.emphasis_strength > 0.0


def test_emphasis_high_intensity_raises_elongation_factor() -> None:
    graph = build_token_graph("خوبه")
    low = EmphasisMapper().map(graph, EmotionalVector(intensity=0.1))
    high = EmphasisMapper().map(graph, EmotionalVector(intensity=0.9))
    assert high.elongation_factor > low.elongation_factor


# --------------------------------------------------------------------------- #
# Contour generator
# --------------------------------------------------------------------------- #
def test_contour_question_rises() -> None:
    from ava_voice.linguistics.punctuation_mapper import ControlSignal

    result = generate_contour(
        EmotionalVector(calm=0.3), terminal_signal=ControlSignal.RISING_PITCH
    )
    assert result.shape is ContourShape.RISING


def test_contour_anger_broken_fear_oscillating() -> None:
    assert (
        ContourGenerator().generate(EmotionalVector(anger=0.8)).shape
        is ContourShape.BROKEN
    )
    assert (
        ContourGenerator().generate(EmotionalVector(fear=0.8)).shape
        is ContourShape.OSCILLATING
    )


def test_contour_neutral_is_flat() -> None:
    result = generate_contour(EmotionalVector())
    assert result.shape is ContourShape.FLAT


# --------------------------------------------------------------------------- #
# Engine + graph annotation
# --------------------------------------------------------------------------- #
def test_engine_produces_profile() -> None:
    graph = build_token_graph("خیلی خوشحالم!")
    vector = EmotionalVector(joy=0.8, valence=0.8, arousal=0.7, intensity=0.85)
    result = generate_prosody(graph, vector)
    assert isinstance(result.profile, ProsodyProfile)
    assert result.profile.speaking_rate > 1.0  # joy speeds up
    assert result.profile.pitch_base > 0.0     # joy raises pitch


def test_engine_annotates_graph_nodes() -> None:
    graph = build_token_graph("سلام. خوبی؟")
    vector = EmotionalVector(joy=0.6, valence=0.6, arousal=0.6, intensity=0.7)
    ProsodyEngine().generate(graph, vector)

    sentence = graph.nodes_of_type(NodeType.SENTENCE)[0]
    assert "prosody" in sentence.attributes
    assert "contour" in sentence.attributes
    assert set(sentence.attributes["prosody"]) >= {"pitch_base", "speaking_rate"}

    tokens = graph.nodes_of_type(NodeType.TOKEN)
    assert all("prosody" in t.attributes for t in tokens)

    pauses = graph.nodes_of_type(NodeType.PAUSE)
    assert all("duration_ms" in p.attributes["prosody"] for p in pauses)


def test_engine_is_deterministic() -> None:
    graph1 = build_token_graph("عصبانیم!!!")
    graph2 = build_token_graph("عصبانیم!!!")
    vector = EmotionalVector(anger=0.9, valence=-0.6, arousal=0.85, intensity=0.9)
    assert (
        generate_prosody(graph1, vector).to_dict()
        == generate_prosody(graph2, vector).to_dict()
    )


def test_engine_integrates_with_emotion_engine() -> None:
    # End-to-end: text -> emotion -> prosody, using the Phase 1B annotations.
    from ava_voice.emotion.emotion_fusion import EmotionEngine

    graph = build_token_graph("خیلی خوشحالم!")
    vector = EmotionEngine().analyze_graph(graph).vector
    result = generate_prosody(graph, vector)
    assert result.profile.contour_shape in set(ContourShape)
    # The emotion annotations let emphasis pick up the joyful token.
    assert result.emphasis.emphasis_strength > 0.0
