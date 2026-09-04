"""Integration tests that load the real Homo-GE2PE checkpoint.

Excluded from the default run (``addopts = -m 'not requires_model'``) because
they need the ~229 MB checkpoint on disk and the ``g2p-neural`` extra. To run:

    pip install -e ".[g2p-neural]"
    git clone https://huggingface.co/MahtaFetrat/Homo-GE2PE-Persian
    unzip -q Homo-GE2PE-Persian/model-weights/homo-ge2pe.zip -d homo-ge2pe
    export AVA_G2P_MODEL_PATH=$PWD/homo-ge2pe
    pytest -m requires_model

The headline assertion is :func:`test_no_vowel_placeholders_on_common_vocabulary`:
the rule-based backend leaves an unresolved ``V`` in roughly three of four common
words, and the neural backend must leave none.
"""

from __future__ import annotations

import pytest

from ava_voice.linguistics.neural_phonemizer import (
    G2PModelUnavailableError,
    NeuralPhonemizer,
    resolve_model_path,
)
from ava_voice.linguistics.phonemizer import (
    SHORT_A,
    VOWEL_PLACEHOLDER,
    PersianPhonemizer,
)
from ava_voice.linguistics.token_graph import NodeType, TokenGraphBuilder

pytestmark = pytest.mark.requires_model


# The vocabulary from the project audit that exposed the placeholder problem.
COMMON_WORDS: tuple[str, ...] = (
    # words named explicitly in the audit
    "مدرسه", "بزرگ", "کشور", "دو", "تو", "او", "نان", "خوب", "کتاب", "سلام",
    "خانه", "آب", "ایران", "دوست", "روز", "متشکرم", "استقلال", "نامه", "کیلو",
    "صبح", "میز", "درخت", "شهر",
    # kinship
    "مادر", "پدر", "برادر", "خواهر", "دختر", "پسر", "مرد", "زن", "بچه",
    # body
    "دست", "پا", "سر", "چشم", "گوش", "دهان", "قلب", "خون",
    # nature
    "آسمان", "زمین", "خورشید", "ماه", "ستاره", "باران", "برف", "باد", "آتش",
    "سنگ",
    # food
    "غذا", "گوشت", "مرغ", "ماهی", "سبزی", "میوه", "سیب", "پرتقال", "انگور",
    # commerce
    "کار", "پول", "بازار", "مغازه", "خرید", "فروش", "قیمت", "گران", "ارزان",
    # travel
    "راه", "ماشین", "قطار", "هواپیما", "کشتی", "سفر", "روستا",
    # abstract
    "عشق", "دل", "جان", "زندگی", "مرگ", "خواب", "بیدار",
)


@pytest.fixture(scope="module")
def neural() -> NeuralPhonemizer:
    """Load the checkpoint once for the whole module, or skip."""
    try:
        resolve_model_path()
    except G2PModelUnavailableError as exc:
        pytest.skip(str(exc))
    phonemizer = NeuralPhonemizer()
    phonemizer.warmup()
    return phonemizer


def test_no_vowel_placeholders_on_common_vocabulary(
    neural: NeuralPhonemizer,
) -> None:
    """Zero unresolved vowels where the rule backend leaves many.

    Words are phonemized one per call so a single bad word cannot fail the batch
    via the word-count guard — this measures the model, not the plumbing.
    """
    legacy = PersianPhonemizer()

    legacy_hits = [
        word
        for word in COMMON_WORDS
        if VOWEL_PLACEHOLDER in legacy.phonemize_word(word)
    ]
    neural_hits: list[tuple[str, list[str]]] = []
    for word in COMMON_WORDS:
        symbols = neural.analyze_sentence([word])[0].symbols
        if VOWEL_PLACEHOLDER in symbols:
            neural_hits.append((word, symbols))

    # Guard the premise: if the rule backend were already clean there would be
    # nothing to fix and this test would prove nothing.
    assert len(legacy_hits) > len(COMMON_WORDS) // 2, (
        f"expected the rule backend to leave placeholders in most words, "
        f"got {len(legacy_hits)}/{len(COMMON_WORDS)}"
    )
    assert not neural_hits, f"neural backend left placeholders: {neural_hits}"


def test_specific_words_the_audit_called_out(neural: NeuralPhonemizer) -> None:
    """Exact pronunciations for the words quoted in the audit."""
    expected = {
        "مدرسه": ["m", SHORT_A, "d", "r", "e", "s", "e"],        # madrese
        "بزرگ": ["b", "o", "z", "o", "r", "g"],                  # bozorg
        "کشور": ["k", "e", "sh", "v", SHORT_A, "r"],             # keshvar
        "دو": ["d", "o"],                                         # do
        "تو": ["t", "o"],                                         # to
        "او": ["'", "u"],                                         # u
        "کتاب": ["k", "e", "t", "a", "b"],                       # ketab
        "صبح": ["s", "o", "b", "h"],                             # sobh
    }
    actual = {
        word: neural.analyze_sentence([word])[0].symbols for word in expected
    }
    assert actual == expected


def test_final_he_is_a_vowel_not_a_consonant(neural: NeuralPhonemizer) -> None:
    """The rule backend renders word-final ه as /h/; it is usually /e/."""
    for word in ("مدرسه", "خانه", "نامه", "ستاره"):
        symbols = neural.analyze_sentence([word])[0].symbols
        assert symbols[-1] == "e", f"{word} -> {symbols}"


def test_waw_is_not_always_a_consonant(neural: NeuralPhonemizer) -> None:
    """The rule backend renders و as /v/ unconditionally: دو -> 'dv'."""
    assert neural.analyze_sentence(["دو"])[0].symbols == ["d", "o"]
    assert neural.analyze_sentence(["تو"])[0].symbols == ["t", "o"]
    assert neural.analyze_sentence(["خوب"])[0].symbols == ["x", "u", "b"]


def test_ezafe_is_detected_from_real_inference(neural: NeuralPhonemizer) -> None:
    """کتابِ علی → ketab-e ali, with the flag set on the head word."""
    words = neural.analyze_sentence(["کتاب", "علی"])
    assert words[0].ezafe is True
    assert words[0].symbols[-1] == "e"
    assert words[0].tokens[-1].is_ezafe is True
    assert words[1].ezafe is False


def test_ezafe_ye_allomorph_from_real_inference(neural: NeuralPhonemizer) -> None:
    """After a vowel the linker surfaces as -ye: خانه بزرگ → xane-ye bozorg."""
    words = neural.analyze_sentence(["خانه", "بزرگ"])
    assert words[0].ezafe is True
    assert words[0].symbols[-2:] == ["y", "e"]


def test_lexical_final_e_is_not_mistaken_for_ezafe(
    neural: NeuralPhonemizer,
) -> None:
    """خانه alone ends in /e/ but carries no ezafe.

    This is why the flag is needed: the phoneme string cannot distinguish them.
    """
    alone = neural.analyze_sentence(["خانه"])[0]
    assert alone.symbols[-1] == "e"
    assert alone.ezafe is False


def test_homographs_resolve_from_sentence_context(
    neural: NeuralPhonemizer,
) -> None:
    """The model's headline capability, and why calls are sentence-level."""
    flower = neural.analyze_sentence(["گل", "زیبا", "است"])
    assert flower[0].symbols == ["g", "o", "l"]  # gol = flower

    mud = neural.analyze_sentence(["کفشم", "پر", "از", "گل", "شد"])
    assert mud[3].symbols == ["g", "e", "l"]  # gel = mud

    thanks = neural.analyze_sentence(["خدا", "را", "شکر"])
    assert thanks[2].symbols == ["sh", "o", "k", "r"]  # shokr = thanks


def test_short_a_is_recovered(neural: NeuralPhonemizer) -> None:
    """The distinction the rule backend collapses: short /æ/ vs long /ɑː/."""
    symbols = neural.analyze_sentence(["مدرسه"])[0].symbols
    assert SHORT_A in symbols
    # And long /ɑː/ still comes back as "a".
    assert "a" in neural.analyze_sentence(["کتاب"])[0].symbols


def test_token_graph_end_to_end_has_no_placeholders(
    neural: NeuralPhonemizer,
) -> None:
    builder = TokenGraphBuilder(g2p=neural)
    graph = builder.build("این کتابِ علی است.")

    phonemes = graph.nodes_of_type(NodeType.PHONEME)
    assert phonemes
    assert not [p for p in phonemes if p.value == VOWEL_PLACEHOLDER]
    assert not [p for p in phonemes if p.attributes["is_placeholder"]]

    # And the ezafe reached the graph.
    tokens = graph.nodes_of_type(NodeType.TOKEN)
    assert any(token.attributes["ezafe"] for token in tokens)


def test_full_pipeline_runs_on_the_neural_backend(
    neural: NeuralPhonemizer,
) -> None:
    """linguistics → emotion → prosody → acoustics, on real phonemes."""
    from ava_voice.acoustics.acoustic_engine import generate_acoustics
    from ava_voice.emotion.emotion_fusion import analyze_once
    from ava_voice.prosody.prosody_engine import generate_prosody

    text = "خیلی خوشحالم! امروز عالی بود."
    graph = TokenGraphBuilder(g2p=neural).build(text)
    prosody = generate_prosody(graph, analyze_once(text).vector)
    acoustic = generate_acoustics(graph, prosody.profile, annotate=False)

    assert acoustic.total_duration_ms > 0
    assert acoustic.phoneme_durations
    # No placeholder survived into the acoustic timeline, so nothing is stretched
    # on an invented vowel.
    assert not [
        duration
        for duration in acoustic.phoneme_durations
        if duration.symbol == VOWEL_PLACEHOLDER
    ]


def test_numerals_are_verbalized_or_cleanly_skipped(
    neural: NeuralPhonemizer,
) -> None:
    """Digits expand to words, which trips the word-count guard by design.

    ``۱۴۰۳`` → ``hezar @o cehars/d @o se`` is three words for one input token, so
    :meth:`analyze_sentence` falls back rather than mis-attribute phonemes. The
    assertion is that this degrades safely, not that it is solved — full numeral
    support needs a verbalizer ahead of G2P.
    """
    words = neural.analyze_sentence(["۱۴۰۳"])
    assert len(words) == 1
    assert words[0].symbols  # something came out, no crash


def test_inference_is_deterministic(neural: NeuralPhonemizer) -> None:
    """Beam search with fixed settings must be repeatable."""
    sentence = ["این", "کتاب", "علی", "است"]
    first = [word.symbols for word in neural.analyze_sentence(sentence)]
    second = [word.symbols for word in neural.analyze_sentence(sentence)]
    assert first == second
