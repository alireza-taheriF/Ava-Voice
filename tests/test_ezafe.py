"""Ezafe linker: detection, allomorph, and register-conditioned acoustics."""

from __future__ import annotations

from ava_voice.acoustics.acoustic_engine import AcousticEngine
from ava_voice.acoustics.acoustic_schema import PhonemeUnit
from ava_voice.acoustics.ezafe_modulator import EzafeModulator
from ava_voice.acoustics.speaker_baseline import SpeakerBaseline
from ava_voice.emotion.emotion_schema import EmotionalVector
from ava_voice.linguistics.ezafe import (
    apply_explicit_ezafe,
    classify_allomorph,
    has_explicit_ezafe,
)
from ava_voice.linguistics.phonemizer import PersianPhonemizer, WordPhonemes
from ava_voice.linguistics.token_graph import NodeType, build_token_graph
from ava_voice.prosody.prosody_schema import ProsodyProfile


def _unit(symbol: str, *, index: int = 0, is_ezafe: bool = False, is_vowel: bool = True):
    return PhonemeUnit(
        index=index,
        node_id=f"p{index}",
        token_id="t0",
        symbol=symbol,
        is_vowel=is_vowel,
        role="nucleus" if is_vowel else "coda",
        syllable_index=0,
        stress_level="none",
        is_elongation_candidate=False,
        is_ezafe=is_ezafe,
    )


def test_explicit_kasra_is_ezafe() -> None:
    assert has_explicit_ezafe("کتابِ") is True
    assert has_explicit_ezafe("کتاب") is False
    assert has_explicit_ezafe("خانه") is False


def test_legacy_g2p_marks_kasra_linker() -> None:
    words = PersianPhonemizer().analyze_sentence(["کتابِ", "علی"])
    assert words[0].ezafe is True
    assert words[0].tokens[-1].is_ezafe is True
    assert words[0].tokens[-1].symbol == "e"
    assert words[1].ezafe is False


def test_lexical_final_e_is_not_ezafe() -> None:
    words = PersianPhonemizer().analyze_sentence(["خانه"])
    assert words[0].ezafe is False
    assert all(not tok.is_ezafe for tok in words[0].tokens)


def test_vowel_final_kasra_selects_ye() -> None:
    words = PersianPhonemizer().analyze_sentence(["خانهِ"])
    assert words[0].ezafe is True
    symbols = words[0].symbols
    assert symbols[-2:] == ["y", "e"]
    assert words[0].tokens[-1].is_ezafe is True
    assert classify_allomorph(symbols, len(symbols) - 1) == "ye"


def test_consonant_final_allomorph_is_e() -> None:
    assert classify_allomorph(["k", "e", "t", "a", "b", "e"], 5) == "e"
    assert classify_allomorph(["x", "a", "n", "e", "y", "e"], 5) == "ye"


def test_apply_explicit_ezafe_on_empty() -> None:
    out = apply_explicit_ezafe(WordPhonemes())
    assert out.ezafe is True
    assert out.tokens[0].is_ezafe is True


def test_graph_carries_ezafe_flag() -> None:
    graph = build_token_graph("کتابِ علی")
    tokens = graph.nodes_of_type(NodeType.TOKEN)
    assert any(t.attributes.get("ezafe") for t in tokens)
    phonemes = graph.nodes_of_type(NodeType.PHONEME)
    assert any(p.attributes.get("is_ezafe") for p in phonemes)


def test_formal_ezafe_shorter_than_warm() -> None:
    units = [
        _unit("b", index=0, is_vowel=False),
        _unit("e", index=1, is_ezafe=True),
    ]
    base = (70.0, 85.0)
    modulator = EzafeModulator()
    formal = EmotionalVector(formality=0.9, intensity=0.2, valence=0.0)
    warm = EmotionalVector(formality=0.0, intensity=0.85, valence=0.8, joy=0.8)
    formal_plan = modulator.plan(units, formal)
    warm_plan = modulator.plan(units, warm)
    formal_ms = modulator.apply_durations(units, base, formal_plan)
    warm_ms = modulator.apply_durations(units, base, warm_plan)
    assert formal_plan.decisions[0].register == "clipped"
    assert warm_plan.decisions[0].register == "warm"
    assert warm_ms[1] > formal_ms[1]
    assert formal_ms[0] == warm_ms[0] == 70.0


def test_ezafe_not_confused_with_lexical_e_in_engine() -> None:
    engine = AcousticEngine()
    lexical = build_token_graph("خانه")
    linked = build_token_graph("خانهِ بزرگ")
    profile = ProsodyProfile()
    lexical_ac = engine.generate(lexical, profile, annotate=False)
    linked_ac = engine.generate(linked, profile, annotate=False)
    assert lexical_ac.ezafe_segments == ()
    assert linked_ac.ezafe_segments
    assert linked_ac.ezafe_segments[0].allomorph == "ye"


def test_engine_writes_ezafe_segment_and_hz_curve() -> None:
    graph = build_token_graph("کتابِ علی")
    vector = EmotionalVector(formality=0.8, intensity=0.4)
    acoustic = AcousticEngine().generate(
        graph, ProsodyProfile(), vector=vector, annotate=False
    )
    assert acoustic.ezafe_segments
    segment = acoustic.ezafe_segments[0]
    assert segment.register == "clipped"
    assert segment.duration_factor < 1.0
    assert acoustic.pitch_curve_hz.n_frames == acoustic.pitch_curve.n_frames
    assert acoustic.pitch_curve_hz.unit == "hertz"
    assert any(p.is_ezafe for p in acoustic.phoneme_durations)


def test_speaker_baseline_converts_semitone_to_hz() -> None:
    speaker = SpeakerBaseline(f0_hz=200.0)
    assert abs(speaker.semitone_to_hz(0.0) - 200.0) < 1e-6
    assert speaker.semitone_to_hz(12.0) == 400.0
    recovered = speaker.hz_to_semitone(400.0)
    assert abs(recovered - 12.0) < 1e-6
