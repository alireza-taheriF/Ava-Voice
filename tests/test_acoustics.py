"""Deterministic unit tests for the Adaptive Persian Acoustic Layer (Phase 2A).

The layer must be fully deterministic and free of machine learning / waveform
generation, so tests assert exact or thresholded behaviour that must hold on
every run.
"""

from __future__ import annotations

from ava_voice.acoustics.acoustic_engine import AcousticEngine, generate_acoustics
from ava_voice.acoustics.acoustic_schema import (
    AcousticProfile,
    PhonemeUnit,
)
from ava_voice.acoustics.duration_mapper import DurationMapper
from ava_voice.acoustics.elongation_controller import ElongationController
from ava_voice.acoustics.energy_envelope import build_energy_curve
from ava_voice.acoustics.pause_shaper import PauseShaper
from ava_voice.acoustics.phoneme_aligner import PhonemeAligner, TokenSlot
from ava_voice.acoustics.pitch_envelope import build_pitch_curve
from ava_voice.emotion.emotion_fusion import analyze_once
from ava_voice.linguistics.phonemizer import VOWEL_PLACEHOLDER
from ava_voice.linguistics.punctuation_mapper import ControlSignal
from ava_voice.linguistics.token_graph import NodeType, build_token_graph
from ava_voice.prosody.prosody_engine import generate_prosody
from ava_voice.prosody.prosody_schema import ContourShape, ProsodyProfile


def _unit(symbol, *, is_vowel=False, role="onset", stress="none", elong=False, index=0):
    return PhonemeUnit(
        index=index,
        node_id=f"phoneme_{index}",
        token_id="token_0",
        symbol=symbol,
        is_vowel=is_vowel,
        role=role,
        syllable_index=0,
        stress_level=stress,
        is_elongation_candidate=elong,
    )


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def test_acoustic_profile_roundtrip() -> None:
    graph = build_token_graph("سلام!")
    profile = generate_prosody(graph, analyze_once("سلام!").vector).profile
    acoustic = generate_acoustics(graph, profile, annotate=False)
    restored = AcousticProfile.from_dict(acoustic.to_dict())
    assert restored.to_dict() == acoustic.to_dict()


# --------------------------------------------------------------------------- #
# Duration mapper
# --------------------------------------------------------------------------- #
def test_duration_fast_compresses_slow_stretches() -> None:
    mapper = DurationMapper()
    vowel = _unit("a", is_vowel=True, role="nucleus")
    fast = mapper.base_duration(vowel, 1.6)
    slow = mapper.base_duration(vowel, 0.7)
    normal = mapper.base_duration(vowel, 1.0)
    assert fast < normal < slow


def test_duration_vowel_longer_than_consonant() -> None:
    mapper = DurationMapper()
    vowel = mapper.base_duration(_unit("a", is_vowel=True, role="nucleus"), 1.0)
    cons = mapper.base_duration(_unit("s", role="onset"), 1.0)
    assert vowel > cons


def test_duration_stress_lengthens_nucleus() -> None:
    mapper = DurationMapper()
    plain = mapper.base_duration(_unit("a", is_vowel=True, role="nucleus"), 1.0)
    stressed = mapper.base_duration(
        _unit("a", is_vowel=True, role="nucleus", stress="primary"), 1.0
    )
    assert stressed > plain


def test_duration_floor_preserves_syllable() -> None:
    mapper = DurationMapper()
    # Even at maximum rate, the vowel nucleus stays above the integrity floor.
    tiny = mapper.base_duration(_unit("e", is_vowel=True, role="nucleus"), 2.0)
    assert tiny >= 55.0


# --------------------------------------------------------------------------- #
# Elongation controller
# --------------------------------------------------------------------------- #
def test_elongation_stretches_candidate_vowel() -> None:
    units = [_unit("a", is_vowel=True, role="nucleus", elong=True, index=0)]
    result = ElongationController().apply(units, [100.0], 2.0, False)
    assert result.durations[0] == 200.0
    assert len(result.segments) == 1
    assert result.segments[0].added_ms == 100.0


def test_elongation_skips_consonants() -> None:
    units = [_unit("s", role="onset", index=0)]
    result = ElongationController().apply(units, [80.0], 2.0, True)
    assert result.durations[0] == 80.0
    assert not result.segments


def test_elongation_salaaam_multiplies_repeated_vowels() -> None:
    # "سلاااام" — the repeated ا letters surface as several vowel nuclei that are
    # all flagged as elongation candidates, so total lengthening accumulates.
    graph = build_token_graph("سلاااام")
    profile = ProsodyProfile(elongation_factor=2.0)
    acoustic = generate_acoustics(graph, profile, annotate=False)
    assert len(acoustic.elongation_segments) >= 2


def test_elongation_never_stretches_the_vowel_placeholder() -> None:
    """``V`` marks a short vowel the phonemizer could not identify.

    Its quality and length are unknown, so drawing it out would render an
    invented vowel expressively. It used to be eligible because is_vowel("V") is
    True.
    """
    units = [
        _unit(VOWEL_PLACEHOLDER, is_vowel=True, role="nucleus", elong=True, index=0)
    ]
    result = ElongationController().apply(units, [70.0], 2.0, True)
    assert result.durations[0] == 70.0
    assert not result.segments


def test_elongation_placeholder_excluded_end_to_end() -> None:
    # "خوبه" phonemizes to x-v-V-b-V-h: every nucleus is a placeholder.
    graph = build_token_graph("خوبه")
    profile = ProsodyProfile(elongation_factor=2.5)
    acoustic = generate_acoustics(graph, profile, annotate=False)
    assert not [s for s in acoustic.elongation_segments if s.symbol == VOWEL_PLACEHOLDER]


# --------------------------------------------------------------------------- #
# Pause shaper
# --------------------------------------------------------------------------- #
def test_pause_scales_with_profile() -> None:
    shaper = PauseShaper()
    short = shaper.shape(ControlSignal.PAUSE_MEDIUM, ProsodyProfile(pause_duration=0.5))
    long = shaper.shape(ControlSignal.PAUSE_MEDIUM, ProsodyProfile(pause_duration=2.0))
    assert long.duration_ms > short.duration_ms
    assert long.category == "clause"


def test_pause_terminal_category_is_sentence() -> None:
    shaper = PauseShaper()
    assert shaper.shape(ControlSignal.RISING_PITCH, ProsodyProfile()).category == "sentence"


# --------------------------------------------------------------------------- #
# Pitch envelope
# --------------------------------------------------------------------------- #
def test_pitch_curve_rising_ends_above_start() -> None:
    profile = ProsodyProfile(
        pitch_base=0.0, pitch_variance=0.8, contour_shape=ContourShape.RISING
    )
    curve = build_pitch_curve(profile, 500.0)
    assert curve.values[-1] > curve.values[0]


def test_pitch_curve_falling_ends_below_start() -> None:
    profile = ProsodyProfile(
        pitch_base=0.0, pitch_variance=0.8, contour_shape=ContourShape.FALLING
    )
    curve = build_pitch_curve(profile, 500.0)
    assert curve.values[-1] < curve.values[0]


def test_pitch_curve_frame_count_matches_duration() -> None:
    curve = build_pitch_curve(ProsodyProfile(), 200.0, 10.0)
    assert curve.n_frames == 20


def test_pitch_curve_deterministic() -> None:
    profile = ProsodyProfile(pitch_variance=0.6, contour_shape=ContourShape.OSCILLATING)
    assert build_pitch_curve(profile, 300.0).values == build_pitch_curve(
        profile, 300.0
    ).values


# --------------------------------------------------------------------------- #
# Energy envelope
# --------------------------------------------------------------------------- #
def test_energy_peaks_raise_curve() -> None:
    profile = ProsodyProfile(
        energy_level=0.4, emphasis_strength=0.9, contour_shape=ContourShape.BROKEN
    )
    flat = build_energy_curve(profile, 300.0, 10.0, ())
    peaked = build_energy_curve(profile, 300.0, 10.0, [(150.0, 0.9)])
    assert max(peaked.values) > max(flat.values)


def test_energy_curve_bounded() -> None:
    profile = ProsodyProfile(energy_level=0.9, emphasis_strength=1.0)
    curve = build_energy_curve(profile, 300.0, 10.0, [(150.0, 1.0)])
    assert all(0.0 <= v <= 1.0 for v in curve.values)


# --------------------------------------------------------------------------- #
# Phoneme aligner
# --------------------------------------------------------------------------- #
def test_aligner_extracts_units_with_roles() -> None:
    graph = build_token_graph("سلام")
    slots = PhonemeAligner().extract_sequence(graph)
    token_slots = [s for s in slots if isinstance(s, TokenSlot)]
    assert token_slots
    units = token_slots[0].units
    assert [u.symbol for u in units] == ["s", "a", "l", "a", "m"]
    # Nuclei are the vowels; at least one nucleus role present.
    assert any(u.role == "nucleus" and u.is_vowel for u in units)


def test_aligner_timeline_is_contiguous() -> None:
    graph = build_token_graph("سلام. خوبی؟")
    profile = generate_prosody(graph, analyze_once("سلام. خوبی؟").vector).profile
    acoustic = generate_acoustics(graph, profile, annotate=False)
    # Phoneme + pause segments should tile the timeline without gaps/overlaps.
    events = sorted(
        [(p.start_ms, p.end_ms) for p in acoustic.phoneme_durations]
        + [(p.start_ms, p.end_ms) for p in acoustic.pause_segments]
    )
    for (s1, e1), (s2, e2) in zip(events, events[1:]):
        assert abs(e1 - s2) < 1e-6  # contiguous


# --------------------------------------------------------------------------- #
# Engine + graph annotation
# --------------------------------------------------------------------------- #
def test_engine_produces_profile_with_curves() -> None:
    graph = build_token_graph("خیلی خوشحالم!")
    profile = generate_prosody(graph, analyze_once("خیلی خوشحالم!").vector).profile
    acoustic = generate_acoustics(graph, profile)
    assert acoustic.total_duration_ms > 0
    assert acoustic.pitch_curve.n_frames > 0
    assert acoustic.energy_curve.n_frames == acoustic.pitch_curve.n_frames
    assert acoustic.phoneme_durations


def test_engine_annotates_graph() -> None:
    graph = build_token_graph("سلام!")
    profile = generate_prosody(graph, analyze_once("سلام!").vector).profile
    generate_acoustics(graph, profile)

    sentence = graph.nodes_of_type(NodeType.SENTENCE)[0]
    assert "acoustic" in sentence.attributes
    assert sentence.attributes["acoustic"]["total_duration_ms"] > 0

    phonemes = graph.nodes_of_type(NodeType.PHONEME)
    assert all("acoustic" in p.attributes for p in phonemes)
    assert all("duration_ms" in p.attributes["acoustic"] for p in phonemes)

    tokens = graph.nodes_of_type(NodeType.TOKEN)
    assert all("acoustic" in t.attributes for t in tokens)


def test_engine_is_deterministic() -> None:
    text = "خیلی عصبانیم!!!"
    g1 = build_token_graph(text)
    g2 = build_token_graph(text)
    vector = analyze_once(text).vector
    p1 = generate_prosody(g1, vector).profile
    p2 = generate_prosody(g2, vector).profile
    a1 = AcousticEngine().generate(g1, p1, annotate=False)
    a2 = AcousticEngine().generate(g2, p2, annotate=False)
    assert a1.to_dict() == a2.to_dict()


def test_slow_sadness_longer_than_fast_joy() -> None:
    # Full stack: sad speech should take longer than joyful speech for similar text.
    sad_text = "خیلی ناراحتم."
    joy_text = "خیلی خوشحالم."
    sad_graph = build_token_graph(sad_text)
    joy_graph = build_token_graph(joy_text)
    sad_profile = generate_prosody(sad_graph, analyze_once(sad_text).vector).profile
    joy_profile = generate_prosody(joy_graph, analyze_once(joy_text).vector).profile
    sad_total = generate_acoustics(sad_graph, sad_profile, annotate=False).total_duration_ms
    joy_total = generate_acoustics(joy_graph, joy_profile, annotate=False).total_duration_ms
    assert sad_profile.speaking_rate < joy_profile.speaking_rate
    assert sad_total > joy_total
