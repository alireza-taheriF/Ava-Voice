"""Deterministic unit tests for the Persian Linguistic Intelligence Layer.

Every test asserts exact, repeatable output — the layer is required to be fully
deterministic, so snapshot-style assertions are appropriate.
"""

from __future__ import annotations

from ava_voice.linguistics.ambiguity_resolver import AmbiguityResolver
from ava_voice.linguistics.persian_normalizer import (
    ZWNJ,
    NormalizerConfig,
    PersianNormalizer,
)
from ava_voice.linguistics.phonemizer import (
    VOWEL_PLACEHOLDER,
    PersianPhonemizer,
    phonemize_word,
)
from ava_voice.linguistics.punctuation_mapper import (
    ControlSignal,
    extract_signals,
    map_punctuation,
)
from ava_voice.linguistics.stress_detector import StressDetector, StressLevel
from ava_voice.linguistics.token_graph import NodeType, build_token_graph


# --------------------------------------------------------------------------- #
# Normalizer
# --------------------------------------------------------------------------- #
def test_normalizer_char_unify_and_zwnj() -> None:
    normalizer = PersianNormalizer()
    # Arabic yeh -> Persian yeh, and "می رم" prefix bound with ZWNJ.
    assert normalizer.normalize("مي رم") == f"می{ZWNJ}رم"


def test_normalizer_removes_tatweel() -> None:
    assert PersianNormalizer().normalize("ســلام") == "سلام"


def test_normalizer_unifies_digits_to_persian() -> None:
    # Arabic-indic 234 + Latin 5 -> Persian digits.
    assert PersianNormalizer().normalize("٢٣٤5") == "۲۳۴۵"


def test_normalizer_strips_emoji() -> None:
    assert PersianNormalizer().normalize("سلام 😀🎉") == "سلام"


def test_normalizer_punctuation_unification() -> None:
    assert PersianNormalizer().normalize("چطوری?") == "چطوری؟"


def test_normalizer_is_deterministic() -> None:
    normalizer = PersianNormalizer()
    text = "مي   رم!!! 123"
    assert normalizer.normalize(text) == normalizer.normalize(text)


def test_normalizer_config_can_disable_steps() -> None:
    cfg = NormalizerConfig(unify_digits=False)
    assert PersianNormalizer(cfg).normalize("۱2٣") == "۱2٣"


# --------------------------------------------------------------------------- #
# Phonemizer
# --------------------------------------------------------------------------- #
def test_phonemize_seed_word() -> None:
    assert phonemize_word("سلام") == ["s", "a", "l", "a", "m"]


def test_phonemize_fallback_inserts_vowel_placeholder() -> None:
    # A word not in the lexicon: consonant clusters get placeholder vowels.
    result = PersianPhonemizer().phonemize_word("بلبل")
    assert VOWEL_PLACEHOLDER in result
    assert result[0] == "b"


def test_phonemize_marks_ambiguous_grapheme() -> None:
    tokens = PersianPhonemizer().analyze_word("دو")  # contains WAW
    waw = [t for t in tokens if t.source_grapheme == "و"][0]
    assert waw.is_ambiguous is True
    assert "u" in waw.alternatives


# --------------------------------------------------------------------------- #
# Stress detector
# --------------------------------------------------------------------------- #
def test_stress_final_syllable_default() -> None:
    detector = StressDetector()
    stress_map = detector.detect(["k", "e", "t", "a", "b"], word="کتاب")
    # Two syllables (ke-tab); Persian nouns are final-stressed.
    assert len(stress_map.syllables) == 2
    assert stress_map.stressed_index == 1
    assert stress_map.syllables[1].stress_level is StressLevel.PRIMARY


def test_stress_prefix_mi_is_initial() -> None:
    detector = StressDetector()
    stress_map = detector.detect(
        ["m", "i", "r", "a", "v", "a", "m"], word=f"می{ZWNJ}روم"
    )
    assert stress_map.stressed_index == 0


def test_stress_emphasis_upgrades_level() -> None:
    detector = StressDetector()
    stress_map = detector.detect(["k", "e", "t", "a", "b"], word="کتاب", emphasis=True)
    assert stress_map.syllables[stress_map.stressed_index].stress_level is (
        StressLevel.EMPHATIC
    )


# --------------------------------------------------------------------------- #
# Punctuation mapper
# --------------------------------------------------------------------------- #
def test_punctuation_core_mappings() -> None:
    assert map_punctuation(".") is ControlSignal.PAUSE_MEDIUM
    assert map_punctuation("!") is ControlSignal.EMPHASIS_BOOST
    assert map_punctuation("؟") is ControlSignal.RISING_PITCH


def test_extract_signals_preserves_position() -> None:
    signals = extract_signals("سلام. خوبی؟")
    assert [s.signal for s in signals] == [
        ControlSignal.PAUSE_MEDIUM,
        ControlSignal.RISING_PITCH,
    ]
    assert signals[0].position == 4


# --------------------------------------------------------------------------- #
# Ambiguity resolver
# --------------------------------------------------------------------------- #
def test_resolver_colloquial_to_canonical() -> None:
    resolver = AmbiguityResolver()
    assert resolver.resolve_token("میرم").canonical == f"می{ZWNJ}روم"
    # ZWNJ-insensitive: the spaced/half-spaced variant resolves identically.
    assert resolver.resolve_token(f"می{ZWNJ}رم").canonical == f"می{ZWNJ}روم"


def test_resolver_marks_homograph_ambiguity() -> None:
    resolution = AmbiguityResolver().resolve_token("کرم")
    assert resolution.is_ambiguous is True
    assert resolution.resolution_kind == "homograph"
    assert "kerm" in resolution.alternatives


def test_end_to_end_normalize_then_resolve() -> None:
    # The documented "مي رم" -> "می‌روم" is achieved by chaining stages.
    normalized = PersianNormalizer().normalize("مي رم")
    canonical = AmbiguityResolver().resolve_token(normalized).canonical
    assert canonical == f"می{ZWNJ}روم"


# --------------------------------------------------------------------------- #
# Token graph
# --------------------------------------------------------------------------- #
def test_token_graph_structure() -> None:
    graph = build_token_graph("سلام!")
    tokens = graph.nodes_of_type(NodeType.TOKEN)
    assert [t.value for t in tokens] == ["سلام"]
    # Emphasis from "!" should upgrade the token's stress to emphatic.
    stress_nodes = graph.nodes_of_type(NodeType.STRESS)
    assert stress_nodes[0].attributes["emphasis_positions"]
    # Every token gets exactly one emotion placeholder.
    emotions = graph.nodes_of_type(NodeType.EMOTION)
    assert len(emotions) == len(tokens)
    assert emotions[0].attributes["placeholder"] is True


def test_token_graph_has_pause_from_punctuation() -> None:
    graph = build_token_graph("سلام. خوبی؟")
    pauses = graph.nodes_of_type(NodeType.PAUSE)
    signals = {p.attributes["signal"] for p in pauses}
    assert ControlSignal.PAUSE_MEDIUM.value in signals
    assert ControlSignal.RISING_PITCH.value in signals


def test_token_graph_is_serializable_and_deterministic() -> None:
    first = build_token_graph("کتاب خوب است.").to_dict()
    second = build_token_graph("کتاب خوب است.").to_dict()
    assert first == second
    assert set(first.keys()) == {"nodes", "edges"}
