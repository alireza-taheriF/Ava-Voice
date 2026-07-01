"""Persian grapheme-to-phoneme (G2P) conversion (Phase 1A).

Converts normalized Persian text into a deterministic phoneme sequence while
**preserving ambiguity** wherever the orthography is genuinely under-specified
(Persian rarely writes short vowels). Two mechanisms encode that uncertainty:

* **Vowel restoration placeholder** (:data:`VOWEL_PLACEHOLDER`) — inserted
  where a short vowel is phonotactically required between consonants but is not
  written. A later, model-driven vowel-restoration stage (out of scope for
  Phase 1A) can replace these placeholders with concrete vowels.
* **Ambiguity markers** — recorded per phoneme in :class:`PhonemeToken` for
  graphemes with multiple legal readings (e.g. ``و`` → ``v`` / ``o`` / ``u``).

The module ships a small, explicit **seed pronunciation lexicon** for high
-frequency words so common tokens phonemize exactly and deterministically; a
rule-based fallback handles everything else. This mirrors production TTS front
ends, which always combine a curated lexicon with a G2P fallback.

Phoneme alphabet
----------------
A compact ASCII-friendly scheme is used (multi-character symbols such as
``sh``/``ch``/``zh``/``gh`` are allowed). Vowels: ``a e o i u`` (short ``a`` and
long ``â`` are both rendered ``a`` at this layer). Consonant clusters and the
glottal stop ``'`` are represented explicitly.

Example
-------
>>> phonemize_word("سلام")
['s', 'a', 'l', 'a', 'm']
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from ava_voice.core.logger import get_logger

_logger = get_logger("linguistics.phonemizer")

ZWNJ: Final[str] = "\u200c"

#: Placeholder emitted where a short vowel is required but not written.
VOWEL_PLACEHOLDER: Final[str] = "V"

#: The set of symbols treated as syllable nuclei (vowels) by downstream stages.
VOWEL_SYMBOLS: Final[frozenset[str]] = frozenset(
    {"a", "e", "o", "i", "u", VOWEL_PLACEHOLDER}
)


def is_vowel(phoneme: str) -> bool:
    """Return ``True`` if ``phoneme`` is a vowel (nucleus) symbol."""
    return phoneme in VOWEL_SYMBOLS


# --------------------------------------------------------------------------- #
# Grapheme → phoneme tables
# --------------------------------------------------------------------------- #
# Unambiguous consonants. Multiple graphemes may share one phoneme (e.g. the
# three Persian /z/ letters), which is expected and lossless for speech.
_CONSONANTS: Final[dict[str, str]] = {
    "ب": "b", "پ": "p", "ت": "t", "ث": "s", "ج": "j", "چ": "ch",
    "ح": "h", "خ": "x", "د": "d", "ذ": "z", "ر": "r", "ز": "z",
    "ژ": "zh", "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t",
    "ظ": "z", "غ": "gh", "ف": "f", "ق": "gh", "ک": "k", "گ": "g",
    "ل": "l", "م": "m", "ن": "n", "ه": "h",
}

# Long-vowel / semivowel bearers with a *default* reading. These are inherently
# ambiguous (context decides consonant vs. vowel), so each records alternatives.
_AMBIGUOUS_GRAPHEMES: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "و": ("v", ("u", "o", "v")),   # WAW: consonant /v/ or vowels /u/,/o/
    "ی": ("i", ("i", "y", "e")),   # YEH: vowel /i/ or glide /y/ or /e/
    "ا": ("a", ("a", "'")),        # ALEF: long /a/ or glottal seat
    "آ": ("a", ("a",)),            # ALEF MADDA: long /a/
    "ع": ("'", ("'", "a")),        # AYN: glottal / vocalic colouring
    "ء": ("'", ("'",)),            # HAMZA: glottal stop
    "أ": ("'", ("'", "a")),
    "ئ": ("'", ("'", "y")),
}

# Harakat (short-vowel diacritics). When present they *disambiguate* vowels.
_HARAKAT: Final[dict[str, str]] = {
    "\u064e": "a",  # FATHA
    "\u0650": "e",  # KASRA
    "\u064f": "o",  # DAMMA
    "\u0652": "",   # SUKUN (no vowel)
}
_TASHDID: Final[str] = "\u0651"  # SHADDA (gemination)

# Seed pronunciation lexicon (normalized surface form -> phoneme sequence).
# Deliberately small and explicit; extend as the engine matures.
_SEED_LEXICON: Final[dict[str, tuple[str, ...]]] = {
    "سلام": ("s", "a", "l", "a", "m"),
    "می‌روم": ("m", "i", "r", "a", "v", "a", "m"),
    "می‌رود": ("m", "i", "r", "a", "v", "a", "d"),
    "خوب": ("x", "u", "b"),
    "خانه": ("x", "a", "n", "e"),
    "کتاب": ("k", "e", "t", "a", "b"),
    "آب": ("a", "b"),
    "ایران": ("i", "r", "a", "n"),
    "دوست": ("d", "u", "s", "t"),
    "روز": ("r", "u", "z"),
}


@dataclass(frozen=True)
class PhonemeToken:
    """A single phoneme together with its provenance and ambiguity metadata.

    Attributes
    ----------
    symbol:
        The chosen phoneme symbol (or :data:`VOWEL_PLACEHOLDER`).
    source_grapheme:
        The grapheme that produced this phoneme (empty for inserted vowels).
    is_ambiguous:
        ``True`` when the source grapheme had multiple legal readings.
    alternatives:
        Other legal readings, for a downstream disambiguation model.
    is_placeholder:
        ``True`` when this is an inserted vowel-restoration placeholder.
    """

    symbol: str
    source_grapheme: str = ""
    is_ambiguous: bool = False
    alternatives: tuple[str, ...] = field(default_factory=tuple)
    is_placeholder: bool = False


class PersianPhonemizer:
    """Deterministic Persian G2P converter (lexicon + rule-based fallback)."""

    def __init__(
        self, lexicon: dict[str, tuple[str, ...]] | None = None
    ) -> None:
        # Copy to keep the instance's lexicon isolated and mutable per-instance.
        self._lexicon: dict[str, tuple[str, ...]] = dict(_SEED_LEXICON)
        if lexicon:
            self._lexicon.update(lexicon)

    # -- Public API --------------------------------------------------------- #
    def phonemize_word(self, word: str) -> list[str]:
        """Return the flat phoneme-symbol sequence for a single ``word``."""
        return [tok.symbol for tok in self.analyze_word(word)]

    def phonemize(self, text: str) -> list[str]:
        """Phonemize whitespace-separated ``text`` into a flat symbol list.

        Word boundaries are not encoded in the flat output; use
        :meth:`analyze_word` / the token graph when boundary information is
        required.
        """
        phonemes: list[str] = []
        for word in text.split():
            phonemes.extend(self.phonemize_word(word))
        return phonemes

    def analyze_word(self, word: str) -> list[PhonemeToken]:
        """Return rich :class:`PhonemeToken` objects for ``word``.

        Resolution order:

        1. Exact match against the seed lexicon (authoritative).
        2. Deterministic rule-based fallback with ambiguity/placeholder marks.
        """
        clean = word.strip()
        if not clean:
            return []

        lexical = self._lexicon.get(clean)
        if lexical is not None:
            return [PhonemeToken(symbol=s) for s in lexical]

        return self._rule_based(clean)

    # -- Rule-based fallback ------------------------------------------------ #
    def _rule_based(self, word: str) -> list[PhonemeToken]:
        """Grapheme-by-grapheme G2P with short-vowel placeholder insertion.

        A short-vowel placeholder is inserted between two adjacent consonants
        (no written vowel/harakat between them), reflecting Persian's
        predominantly CV syllable structure while preserving the fact that the
        vowel's identity is unresolved.
        """
        tokens: list[PhonemeToken] = []
        # Strip ZWNJ for phonemization; it is an orthographic, not phonetic, mark.
        graphemes = [g for g in word if g != ZWNJ]

        for grapheme in graphemes:
            if grapheme in _HARAKAT:
                vowel = _HARAKAT[grapheme]
                if vowel:
                    tokens.append(
                        PhonemeToken(symbol=vowel, source_grapheme=grapheme)
                    )
                continue
            if grapheme == _TASHDID:
                # Gemination: duplicate the previous consonant if present.
                if tokens and not is_vowel(tokens[-1].symbol):
                    prev = tokens[-1]
                    tokens.append(
                        PhonemeToken(symbol=prev.symbol, source_grapheme=grapheme)
                    )
                continue

            if grapheme in _AMBIGUOUS_GRAPHEMES:
                default, alts = _AMBIGUOUS_GRAPHEMES[grapheme]
                tokens.append(
                    PhonemeToken(
                        symbol=default,
                        source_grapheme=grapheme,
                        is_ambiguous=len(alts) > 1,
                        alternatives=alts,
                    )
                )
                continue

            if grapheme in _CONSONANTS:
                phoneme = _CONSONANTS[grapheme]
                # Insert a placeholder short vowel between two consonants.
                if tokens and not is_vowel(tokens[-1].symbol):
                    tokens.append(
                        PhonemeToken(symbol=VOWEL_PLACEHOLDER, is_placeholder=True)
                    )
                tokens.append(
                    PhonemeToken(symbol=phoneme, source_grapheme=grapheme)
                )
                continue

            # Unknown grapheme (Latin letters, residual symbols): pass through
            # lower-cased so the pipeline never silently drops information.
            _logger.debug("Unmapped grapheme", extra={"grapheme": grapheme})
            tokens.append(PhonemeToken(symbol=grapheme.lower(), source_grapheme=grapheme))

        return tokens


_default_phonemizer: Final[PersianPhonemizer] = PersianPhonemizer()


def phonemize_word(word: str) -> list[str]:
    """Phonemize a single word using the module-level default phonemizer."""
    return _default_phonemizer.phonemize_word(word)


def phonemize(text: str) -> list[str]:
    """Phonemize text using the module-level default phonemizer."""
    return _default_phonemizer.phonemize(text)
