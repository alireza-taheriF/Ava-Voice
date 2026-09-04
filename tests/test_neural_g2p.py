"""Unit tests for the neural G2P backend — no model download required.

Every test here mocks the generator, so the suite stays fast and CI needs no
229 MB checkpoint. The counterpart integration tests that load the real model
live in ``test_neural_g2p_integration.py`` behind the ``requires_model`` marker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from ava_voice.linguistics.backend import build_phonemizer
from ava_voice.linguistics.neural_phonemizer import (
    EZAFE_MARKER,
    G2PModelUnavailableError,
    NeuralPhonemizer,
    has_explicit_ezafe,
    is_model_ready_token,
    parse_model_output,
    parse_model_word,
    resolve_model_path,
)
from ava_voice.linguistics.phonemizer import (
    SHORT_A,
    VOWEL_PLACEHOLDER,
    PersianPhonemizer,
)
from ava_voice.linguistics.token_graph import NodeType, TokenGraphBuilder


class _FakeModel:
    """Stands in for ``_LazyModel``: returns canned output, records calls."""

    def __init__(self, *responses: str, raises: Exception | None = None) -> None:
        self._responses = list(responses)
        self._raises = raises
        self.calls: list[list[str]] = []

    def generator(self):
        def generate(sentences: Sequence[str]) -> list[str]:
            self.calls.append(list(sentences))
            if self._raises is not None:
                raise self._raises
            return [self._responses.pop(0)] if self._responses else [""]

        return generate


def _phonemizer(*responses: str, raises: Exception | None = None) -> tuple[
    NeuralPhonemizer, _FakeModel
]:
    model = _FakeModel(*responses, raises=raises)
    return NeuralPhonemizer(model=model), model


# --------------------------------------------------------------------------- #
# Symbol translation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ketab", ("k", "e", "t", "a", "b")),
        ("ke$v/r", ("k", "e", "sh", "v", SHORT_A, "r")),          # کشور
        ("@/st", ("'", SHORT_A, "s", "t")),                       # است
        (";ale", ("zh", "a", "l", "e")),                          # ژاله
        ("b/cce", ("b", SHORT_A, "ch", "ch", "e")),               # بچه
        ("morq", ("m", "o", "r", "gh")),                          # مرغ
        ("m/drese", ("m", SHORT_A, "d", "r", "e", "s", "e")),     # مدرسه
        ("bozorg", ("b", "o", "z", "o", "r", "g")),               # بزرگ
    ],
)
def test_parse_maps_model_symbols_to_project_alphabet(
    raw: str, expected: tuple[str, ...]
) -> None:
    assert parse_model_word(raw).symbols == expected


def test_parse_never_emits_the_vowel_placeholder() -> None:
    """The whole point: the model resolves what the rule engine could not."""
    for raw in ("m/drese", "bozorg", "ke$v/r", "do", "to", "@u"):
        assert VOWEL_PLACEHOLDER not in parse_model_word(raw).symbols


def test_parse_multichar_symbols_are_single_units() -> None:
    # "$" -> "sh" must be one phoneme, not two.
    assert parse_model_word("$/hr").symbols == ("sh", SHORT_A, "h", "r")


def test_parse_splits_words() -> None:
    parsed = parse_model_output("@in ketabe1 @ali")
    assert [word.symbols for word in parsed] == [
        ("'", "i", "n"),
        ("k", "e", "t", "a", "b", "e"),
        ("'", "a", "l", "i"),
    ]


def test_parse_tolerates_extra_whitespace() -> None:
    assert len(parse_model_output("  yek   do  se ")) == 3


def test_parse_drops_stray_digits_rather_than_voicing_them() -> None:
    # A digit that is not the trailing ezafe marker is not a phoneme.
    assert parse_model_word("ke1tab").symbols == ("k", "e", "t", "a", "b")


# --------------------------------------------------------------------------- #
# Ezafe
# --------------------------------------------------------------------------- #
def test_ezafe_marker_sets_the_flag_and_is_not_a_phoneme() -> None:
    parsed = parse_model_word("ketabe" + EZAFE_MARKER)
    assert parsed.ezafe is True
    assert parsed.symbols == ("k", "e", "t", "a", "b", "e")
    assert EZAFE_MARKER not in parsed.symbols


def test_absent_marker_means_no_ezafe() -> None:
    assert parse_model_word("ketab").ezafe is False
    # "xane" (خانه) ends in a *lexical* e, which must not be read as ezafe.
    assert parse_model_word("xane").ezafe is False


def test_ezafe_reaches_word_phonemes_and_tags_the_linking_vowel() -> None:
    g2p, _ = _phonemizer("ketabe1 @/li")
    words = g2p.analyze_sentence(["کتاب", "علی"])

    head, modifier = words
    assert head.ezafe is True
    assert modifier.ezafe is False
    # The vowel stays in the stream (it is spoken) but is tagged.
    assert head.symbols == ["k", "e", "t", "a", "b", "e"]
    assert head.tokens[-1].is_ezafe is True
    assert not any(token.is_ezafe for token in head.tokens[:-1])
    assert not any(token.is_ezafe for token in modifier.tokens)


def test_ezafe_ye_allomorph_after_a_vowel() -> None:
    # خانه بزرگ -> xane-ye bozorg: the linker surfaces as "ye" after a vowel.
    g2p, _ = _phonemizer("xaneye1 bozorg")
    head = g2p.analyze_sentence(["خانه", "بزرگ"])[0]
    assert head.ezafe is True
    assert head.symbols == ["x", "a", "n", "e", "y", "e"]


def test_ezafe_surfaces_on_the_token_graph() -> None:
    g2p, _ = _phonemizer("ketabe1 @/li")
    graph = TokenGraphBuilder(g2p=g2p).build("کتاب علی")

    tokens = graph.nodes_of_type(NodeType.TOKEN)
    assert [token.attributes["ezafe"] for token in tokens] == [True, False]

    phonemes = graph.nodes_of_type(NodeType.PHONEME)
    ezafe_phonemes = [p for p in phonemes if p.attributes["is_ezafe"]]
    assert len(ezafe_phonemes) == 1
    assert ezafe_phonemes[0].value == "e"


def test_explicit_kasra_forces_ezafe_when_the_model_misses_it() -> None:
    """A written kasra is an instruction, not a hint.

    ``GE2PE.generate`` strips harakat before inference, so the model cannot see
    it — and probing showed it does miss it in context: ``این کتابِ علی است``
    returns plain ``ketab``, no marker.
    """
    g2p, _ = _phonemizer("@in ketab @ali @/st")
    words = g2p.analyze_sentence(["این", "کتابِ", "علی", "است"])

    assert [word.ezafe for word in words] == [False, True, False, False]
    head = words[1]
    assert head.symbols == ["k", "e", "t", "a", "b", "e"]
    assert head.tokens[-1].is_ezafe is True


def test_explicit_kasra_does_not_double_the_vowel() -> None:
    # The model already emitted the linker; the kasra must not append another.
    g2p, _ = _phonemizer("ketabe1 @/li")
    head = g2p.analyze_sentence(["کتابِ", "علی"])[0]
    assert head.symbols == ["k", "e", "t", "a", "b", "e"]
    assert head.ezafe is True


def test_explicit_kasra_detection() -> None:
    assert has_explicit_ezafe("کتابِ") is True
    assert has_explicit_ezafe("کتاب") is False
    assert has_explicit_ezafe("") is False


# --------------------------------------------------------------------------- #
# Token gating
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("token", "eligible"),
    [
        ("سلام", True),
        ("می‌روم", True),
        ("۱۴۰۳", True),   # digits are verbalized by the model
        ("1403", True),
        ("Django", False),
        ("!!!", False),
        ("", False),
        ("   ", False),
        ("...", False),
    ],
)
def test_gating_keeps_degenerate_input_away_from_the_model(
    token: str, eligible: bool
) -> None:
    """Non-Persian input makes the model emit 512 tokens of gibberish.

    Probed: "" and "!!!" both return "mkk nik mikon fo mkon …"; "Django" returns
    "ngoude ngourde ngo/ngole …".
    """
    assert is_model_ready_token(token) is eligible


def test_ineligible_tokens_use_the_fallback_and_keep_position() -> None:
    g2p, model = _phonemizer("s/lam xubi")
    words = g2p.analyze_sentence(["سلام", "Django", "خوبی"])

    # Only the Persian words were sent, joined into one sentence.
    assert model.calls == [["سلام خوبی"]]
    assert len(words) == 3
    # The Latin token was phonemized by the rule engine, in its original slot.
    assert words[1].symbols == PersianPhonemizer().phonemize_word("Django")
    assert words[0].symbols == ["s", SHORT_A, "l", "a", "m"]
    assert words[2].symbols == ["x", "u", "b", "i"]


def test_all_ineligible_means_the_model_is_never_called() -> None:
    g2p, model = _phonemizer()
    words = g2p.analyze_sentence(["!!!", "Django"])
    assert model.calls == []
    assert len(words) == 2


def test_empty_sentence_is_not_sent_to_the_model() -> None:
    g2p, model = _phonemizer()
    assert g2p.analyze_sentence([]) == []
    assert model.calls == []


# --------------------------------------------------------------------------- #
# Sentence-level context
# --------------------------------------------------------------------------- #
def test_whole_sentence_goes_in_one_call() -> None:
    """Homograph disambiguation needs context, so batching per word would lose it.

    Probed: "گل" alone returns "ole"; in "گل زیبا است" it returns "gol".
    """
    g2p, model = _phonemizer("gol ziba @/st")
    g2p.analyze_sentence(["گل", "زیبا", "است"])
    assert model.calls == [["گل زیبا است"]]


def test_token_graph_issues_a_single_inference_per_sentence() -> None:
    g2p, model = _phonemizer("k/f$/m por @/z gel $od")
    TokenGraphBuilder(g2p=g2p).build("کفشم پر از گل شد")
    assert len(model.calls) == 1


# --------------------------------------------------------------------------- #
# Fallback behaviour
# --------------------------------------------------------------------------- #
def test_word_count_mismatch_falls_back_wholesale() -> None:
    """The model may merge or split words; mis-attributing phonemes is worse.

    Probed: "می روم" (2 words) returns 1; "۱۴۰۳" (1 word) returns 3.
    """
    g2p, _ = _phonemizer("mir/v/m")  # one word back for two in
    words = g2p.analyze_sentence(["می", "روم"])
    assert len(words) == 2
    legacy = PersianPhonemizer()
    assert words[0].symbols == legacy.phonemize_word("می")
    assert words[1].symbols == legacy.phonemize_word("روم")


def test_empty_model_output_falls_back() -> None:
    g2p, _ = _phonemizer("")
    words = g2p.analyze_sentence(["کتاب"])
    assert words[0].symbols == PersianPhonemizer().phonemize_word("کتاب")


def test_word_parsing_to_nothing_falls_back() -> None:
    # A word made entirely of ignored characters yields no phonemes.
    g2p, _ = _phonemizer("ketab -")
    words = g2p.analyze_sentence(["کتاب", "علی"])
    assert words[1].symbols == PersianPhonemizer().phonemize_word("علی")


def test_inference_error_falls_back_without_raising() -> None:
    g2p, _ = _phonemizer(raises=RuntimeError("CUDA out of memory"))
    words = g2p.analyze_sentence(["کتاب"])
    assert words[0].symbols == PersianPhonemizer().phonemize_word("کتاب")


def test_missing_model_error_propagates() -> None:
    """A misconfigured install must be loud, not silently degraded."""
    g2p, _ = _phonemizer(raises=G2PModelUnavailableError("no weights"))
    with pytest.raises(G2PModelUnavailableError):
        g2p.analyze_sentence(["کتاب"])


def test_fallback_preserves_the_pipeline_end_to_end() -> None:
    g2p, _ = _phonemizer(raises=RuntimeError("boom"))
    graph = TokenGraphBuilder(g2p=g2p).build("کتاب روی میز است.")
    assert graph.nodes_of_type(NodeType.TOKEN)
    assert graph.nodes_of_type(NodeType.PHONEME)


# --------------------------------------------------------------------------- #
# Interface parity
# --------------------------------------------------------------------------- #
def test_neural_backend_satisfies_the_phonemizer_interface() -> None:
    g2p, _ = _phonemizer("ketab")
    assert g2p.phonemize_word("کتاب") == ["k", "e", "t", "a", "b"]


def test_analyze_word_returns_phoneme_tokens() -> None:
    g2p, _ = _phonemizer("ketab")
    tokens = g2p.analyze_word("کتاب")
    assert [t.symbol for t in tokens] == ["k", "e", "t", "a", "b"]


def test_analyze_word_on_blank_input_is_empty_and_calls_nothing() -> None:
    g2p, model = _phonemizer()
    assert g2p.analyze_word("   ") == []
    assert model.calls == []


def test_phonemize_flattens_across_words() -> None:
    g2p, _ = _phonemizer("do se")
    assert g2p.phonemize("دو سه") == ["d", "o", "s", "e"]


# --------------------------------------------------------------------------- #
# Lazy loading
# --------------------------------------------------------------------------- #
def test_constructing_the_backend_loads_nothing() -> None:
    """Importing and constructing must not touch disk or import torch."""
    import sys

    g2p = NeuralPhonemizer(model_path="/nonexistent/path")
    assert g2p is not None
    assert "torch" not in sys.modules


def test_model_path_resolution_errors_are_actionable(tmp_path: Path) -> None:
    with pytest.raises(G2PModelUnavailableError, match="not found"):
        resolve_model_path(tmp_path / "missing")

    # A directory with no config.json is not an unpacked checkpoint.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(G2PModelUnavailableError, match="config.json"):
        resolve_model_path(empty)


def test_model_path_env_var_is_honoured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "homo-ge2pe"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    monkeypatch.setenv("AVA_G2P_MODEL_PATH", str(checkpoint))
    assert resolve_model_path() == checkpoint


def test_error_message_names_both_ways_out() -> None:
    with pytest.raises(G2PModelUnavailableError) as excinfo:
        resolve_model_path("/definitely/not/here")
    message = str(excinfo.value)
    assert "g2p-neural" in message
    assert "AVA_G2P_BACKEND=legacy" in message


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #
def test_build_phonemizer_defaults_to_legacy() -> None:
    assert isinstance(build_phonemizer("legacy"), PersianPhonemizer)


def test_build_phonemizer_neural_does_not_load_the_model() -> None:
    backend = build_phonemizer("neural", model_path="/nonexistent")
    assert isinstance(backend, NeuralPhonemizer)


def test_build_phonemizer_unknown_backend_degrades_to_legacy() -> None:
    # A typo in an environment variable should not take the process down.
    assert isinstance(build_phonemizer("nueral"), PersianPhonemizer)


def test_settings_expose_the_backend_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from ava_voice.core.config import Settings, get_settings

    assert Settings().g2p_backend == "legacy"

    monkeypatch.setenv("AVA_G2P_BACKEND", "neural")
    get_settings.cache_clear()
    try:
        assert get_settings().g2p_backend == "neural"
    finally:
        get_settings.cache_clear()

    with pytest.raises(ValueError):
        Settings(g2p_backend="tensorflow")  # type: ignore[arg-type]


def test_legacy_backend_reports_no_ezafe() -> None:
    """The rule engine cannot detect ezafe; it must say so rather than guess."""
    words = PersianPhonemizer().analyze_sentence(["کتاب", "علی"])
    assert [word.ezafe for word in words] == [False, False]
