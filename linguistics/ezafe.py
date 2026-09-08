"""Ezafe (اضافه) detection and allomorph selection (Phase 0).

Persian ezafe is a spoken linker, not a letter. A final ``e`` is otherwise
ambiguous: ``خانه`` → *xane* (lexical) vs ``خانه بزرگ`` → *xane-ye* (linker).
This module is the single authority for:

* **explicit kasra** — a writer who types ``کتابِ`` is marking the linker;
* **allomorph selection** — consonant-final heads take ``-e``, vowel-final
  heads take ``-ye``.

Both G2P backends (legacy and neural) call these helpers so the acoustic
:class:`~ava_voice.acoustics.ezafe_modulator.EzafeModulator` can treat the
linker as a first-class event. Nothing here is learned; everything is a
closed-form, deterministic rule of Persian phonology.
"""

from __future__ import annotations

import unicodedata
from typing import Final, Sequence

from ava_voice.linguistics.phonemizer import (
    PhonemeToken,
    WordPhonemes,
    is_vowel,
)

#: Kasra (U+0650). Word-final kasra is the orthographic ezafe mark.
KASRA: Final[str] = "ِ"

#: Spoken linker vowel.
EZAFE_VOWEL: Final[str] = "e"

#: On-glide of the vowel-final allomorph (*-ye*).
EZAFE_GLIDE: Final[str] = "y"


def strip_ezafe_kasra(word: str) -> str:
    """Remove the last kasra so G2P sees the lexical spelling.

    ``خانهِ`` must hit the ``خانه`` lexicon entry (*xane*), not a
    letter-by-letter fallback that treats ``ه`` as ``h``.
    """
    index = (word or "").rfind(KASRA)
    if index < 0:
        return word
    return word[:index] + word[index + 1 :]


def has_explicit_ezafe(word: str) -> bool:
    """Return ``True`` when ``word`` carries a written kasra ezafe mark.

    Combining marks may sit after the kasra (shadda, tanwin); we still treat
    a kasra that is the last *vowel diacritic* on the word as explicit ezafe
    when it appears at the end of the grapheme string.

    Examples
    --------
    >>> has_explicit_ezafe("کتابِ")
    True
    >>> has_explicit_ezafe("کتاب")
    False
    >>> has_explicit_ezafe("خانه")
    False
    """
    stripped = (word or "").strip()
    if not stripped:
        return False
    # Walk from the end, skipping other combining marks, looking for kasra.
    for char in reversed(stripped):
        if char == KASRA:
            return True
        category = _combining_category(char)
        if category:
            continue
        return False
    return False


def _combining_category(char: str) -> bool:
    """Return ``True`` if ``char`` is a combining mark (not a letter)."""
    return unicodedata.category(char).startswith("M")


def classify_allomorph(
    symbols: Sequence[str], ezafe_index: int
) -> str:
    """Return ``"ye"`` or ``"e"`` for the linker at ``ezafe_index``.

    The neural backend tags only the *last* phoneme of the linker. For
    *xane-ye* that is the final ``e``, whose previous symbol is ``y``. A
    vowel immediately before the tagged linker (same word) also selects
    ``ye``, matching the phonological rule.
    """
    if ezafe_index < 0 or ezafe_index >= len(symbols):
        return "e"
    symbol = symbols[ezafe_index]
    if symbol == EZAFE_GLIDE:
        return "ye"
    if ezafe_index > 0:
        previous = symbols[ezafe_index - 1]
        if previous == EZAFE_GLIDE or is_vowel(previous):
            return "ye"
    return "e"


def apply_explicit_ezafe(word_phonemes: WordPhonemes) -> WordPhonemes:
    """Force the ezafe flag and ensure a spoken linker vowel is present.

    * Consonant-final → append ``e`` (or mark a trailing ``e`` already
      produced by a written kasra).
    * Vowel-final (and not already ``… y e``) → append ``y e``.

    The last phoneme is tagged ``is_ezafe`` so acoustics can find the linker
    without guessing from a trailing ``e``.
    """
    tokens = list(word_phonemes.tokens)
    if not tokens:
        linker = PhonemeToken(
            symbol=EZAFE_VOWEL, source_grapheme=KASRA, is_ezafe=True
        )
        return WordPhonemes(tokens=(linker,), ezafe=True)

    # Always *append* the linker. A lexical final ``e`` (خانه → xane) is not
    # the ezafe; tagging it would collapse the two events the invention
    # exists to keep apart.
    last = tokens[-1]
    if is_vowel(last.symbol):
        tokens.append(PhonemeToken(symbol=EZAFE_GLIDE, source_grapheme=KASRA))
    tokens.append(
        PhonemeToken(symbol=EZAFE_VOWEL, source_grapheme=KASRA, is_ezafe=True)
    )
    return WordPhonemes(tokens=tuple(tokens), ezafe=True)
