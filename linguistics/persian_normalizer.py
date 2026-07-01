"""Persian orthographic normalization (Phase 1A).

This module is the *first* deterministic stage of the Persian Linguistic
Intelligence Layer. Its sole responsibility is to convert arbitrary,
real-world Persian input into a **canonical orthographic surface form** that
downstream stages (ambiguity resolution, phonemization, stress detection) can
rely on.

Design principles
-----------------
* **Deterministic** — identical input always yields identical output. No
  randomness, no locale-dependent behaviour, no network or model calls.
* **Single responsibility** — this stage performs *orthographic* normalization
  only. It intentionally does **not** convert colloquial verbs to their formal
  spoken forms (e.g. ``میرم`` → ``می‌روم``); that is a *semantic/spoken*
  transformation owned by :mod:`ava_voice.linguistics.ambiguity_resolver`.
* **Composable** — the pipeline is expressed as an ordered list of pure
  ``str -> str`` steps so individual steps are unit-testable in isolation and
  the ordering is auditable (important for a patent-grade specification).

Pipeline order (each step is a pure function)
---------------------------------------------
1. Unicode NFC composition.
2. Character unification (Arabic → Persian code points).
3. Diacritic / control-character hygiene (tatweel, BOM, bidi marks).
4. Emoji & pictograph stripping.
5. Digit unification (Arabic-Indic / Latin → configurable target).
6. Punctuation normalization (unify Persian punctuation, collapse repeats).
7. Abbreviation expansion (deterministic lexicon).
8. نیم‌فاصله (ZWNJ) normalization (affix binding + hygiene).
9. Whitespace normalization (collapse + trim).

Examples
--------
>>> normalizer = PersianNormalizer()
>>> normalizer.normalize("مي رم")          # char-unify + ZWNJ affix binding
'می‌رم'
>>> normalizer.normalize("ســلام")          # tatweel removal
'سلام'
>>> normalizer.normalize("قيمت ٢٣٤ تومان")   # char-unify + digit unify
'قیمت ۲۳۴ تومان'

Note that the *full* spoken canonical ``مي رم`` → ``می‌روم`` is produced by
chaining this normalizer with the ambiguity resolver.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Final, Literal

from ava_voice.core.logger import get_logger

_logger = get_logger("linguistics.normalizer")

# Zero-Width Non-Joiner — the technical name for the Persian "نیم‌فاصله".
ZWNJ: Final[str] = "\u200c"

DigitSystem = Literal["fa", "en"]

# --------------------------------------------------------------------------- #
# Character unification tables
# --------------------------------------------------------------------------- #
# Map visually/semantically equivalent Arabic code points to their canonical
# Persian counterparts. Harakat (short-vowel diacritics) are deliberately
# preserved here because they carry vowel information the phonemizer consumes.
_CHARACTER_UNIFICATION: Final[dict[str, str]] = {
    "\u064a": "\u06cc",  # ARABIC YEH (ي)            -> PERSIAN YEH (ی)
    "\u0649": "\u06cc",  # ARABIC ALEF MAKSURA (ى)   -> PERSIAN YEH (ی)
    "\u0643": "\u06a9",  # ARABIC KAF (ك)            -> PERSIAN KEHEH (ک)
    "\u0629": "\u0647",  # ARABIC TEH MARBUTA (ة)    -> HEH (ه)
    "\u0623": "\u0627",  # ARABIC ALEF W/ HAMZA (أ)  -> ALEF (ا)
    "\u0625": "\u0627",  # ARABIC ALEF W/ HAMZA (إ)  -> ALEF (ا)
    "\u0624": "\u0648",  # ARABIC WAW W/ HAMZA (ؤ)   -> WAW (و)
    "\u0626": "\u06cc",  # ARABIC YEH W/ HAMZA (ئ)   -> PERSIAN YEH (ی)
    "\u06c0": "\u0647",  # HEH W/ YEH ABOVE (ۀ)      -> HEH (ه)
    "\u0640": "",        # ARABIC TATWEEL (ـ)        -> removed (kashida)
}

# Control / invisible characters that must never survive normalization.
# ZWNJ (U+200C) is intentionally excluded — it is handled explicitly.
_INVISIBLE_CHARACTERS: Final[tuple[str, ...]] = (
    "\ufeff",  # BOM / ZERO WIDTH NO-BREAK SPACE
    "\u200b",  # ZERO WIDTH SPACE
    "\u200d",  # ZERO WIDTH JOINER
    "\u200e",  # LEFT-TO-RIGHT MARK
    "\u200f",  # RIGHT-TO-LEFT MARK
    "\u061c",  # ARABIC LETTER MARK
)

# --------------------------------------------------------------------------- #
# Digit tables
# --------------------------------------------------------------------------- #
_PERSIAN_DIGITS: Final[str] = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC_INDIC_DIGITS: Final[str] = "٠١٢٣٤٥٦٧٨٩"
_LATIN_DIGITS: Final[str] = "0123456789"

_TO_PERSIAN_DIGITS: Final[dict[int, int]] = {
    **{ord(a): ord(p) for a, p in zip(_ARABIC_INDIC_DIGITS, _PERSIAN_DIGITS)},
    **{ord(l): ord(p) for l, p in zip(_LATIN_DIGITS, _PERSIAN_DIGITS)},
}
_TO_LATIN_DIGITS: Final[dict[int, int]] = {
    **{ord(a): ord(l) for a, l in zip(_ARABIC_INDIC_DIGITS, _LATIN_DIGITS)},
    **{ord(p): ord(l) for p, l in zip(_PERSIAN_DIGITS, _LATIN_DIGITS)},
}

# --------------------------------------------------------------------------- #
# Punctuation table
# --------------------------------------------------------------------------- #
# Unify Latin punctuation to its Persian counterpart so a single downstream
# punctuation vocabulary is sufficient (see :mod:`punctuation_mapper`).
_PUNCTUATION_UNIFICATION: Final[dict[str, str]] = {
    "?": "؟",   # QUESTION MARK      -> ARABIC QUESTION MARK
    ",": "،",   # COMMA              -> ARABIC COMMA
    ";": "؛",   # SEMICOLON          -> ARABIC SEMICOLON
    "٫": "٫",   # keep decimal separator
    "…": "…",
}

# Abbreviation → expansion. Deterministic, closed lexicon. Keys are matched as
# whole tokens after character unification.
_ABBREVIATIONS: Final[dict[str, str]] = {
    "ه.ش": "هجری شمسی",
    "ه.ق": "هجری قمری",
    "ه.م": "هجری میلادی",
    "ج.ا.ا": "جمهوری اسلامی ایران",
    "الخ": "و غیره",
    "ص.": "صفحه",
    "ر.ک": "رجوع کنید",
}

# Emoji / pictograph ranges (BMP + astral). Compiled once at import time.
_EMOJI_PATTERN: Final[re.Pattern[str]] = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols, pictographs, emoji extensions
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "\U0001f000-\U0001f0ff"  # mahjong/domino/playing cards
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0001f1e6-\U0001f1ff"  # regional indicator symbols (flags)
    "\U00002190-\U000021ff"  # arrows
    "\U00002b00-\U00002bff"  # misc symbols & arrows
    "]+",
    flags=re.UNICODE,
)

# ZWNJ-binding affixes. These bind to the adjacent word with a نیم‌فاصله rather
# than a full space. Kept intentionally conservative to remain deterministic
# and avoid over-binding.
_PREFIX_BINDING: Final[re.Pattern[str]] = re.compile(
    r"(?:^|(?<=\s))(ن?می)[ ]+(?=\S)"
)
_SUFFIX_PLURAL_BINDING: Final[re.Pattern[str]] = re.compile(
    r"(?<=\S)[ ]+(ها(?:ی|یی)?)(?=\s|$)"
)
_SUFFIX_COMPARATIVE_BINDING: Final[re.Pattern[str]] = re.compile(
    r"(?<=\S)[ ]+(تر(?:ین)?)(?=\s|$)"
)


@dataclass(frozen=True)
class NormalizerConfig:
    """Immutable configuration flags controlling normalization behaviour.

    Every flag defaults to the production-recommended value. The dataclass is
    frozen so a configured :class:`PersianNormalizer` is safe to share across
    threads and requests.
    """

    unify_characters: bool = True
    remove_invisibles: bool = True
    strip_emoji: bool = True
    unify_digits: bool = True
    digit_system: DigitSystem = "fa"
    normalize_punctuation: bool = True
    expand_abbreviations: bool = True
    apply_zwnj_binding: bool = True
    collapse_whitespace: bool = True


class PersianNormalizer:
    """Deterministic Persian orthographic normalizer.

    The normalizer holds an immutable :class:`NormalizerConfig` and exposes a
    single public :meth:`normalize` method. Internally the transformation is an
    ordered list of pure ``str -> str`` steps, each individually testable via
    the corresponding ``_step_*`` method.
    """

    def __init__(self, config: NormalizerConfig | None = None) -> None:
        self.config = config or NormalizerConfig()
        self._pipeline: tuple[Callable[[str], str], ...] = self._build_pipeline()

    # -- Pipeline assembly -------------------------------------------------- #
    def _build_pipeline(self) -> tuple[Callable[[str], str], ...]:
        """Assemble the ordered list of enabled transformation steps."""
        cfg = self.config
        steps: list[Callable[[str], str]] = [self._step_nfc]
        if cfg.unify_characters:
            steps.append(self._step_unify_characters)
        if cfg.remove_invisibles:
            steps.append(self._step_remove_invisibles)
        if cfg.strip_emoji:
            steps.append(self._step_strip_emoji)
        if cfg.unify_digits:
            steps.append(self._step_unify_digits)
        if cfg.normalize_punctuation:
            steps.append(self._step_normalize_punctuation)
        if cfg.expand_abbreviations:
            steps.append(self._step_expand_abbreviations)
        if cfg.apply_zwnj_binding:
            steps.append(self._step_zwnj_binding)
        if cfg.collapse_whitespace:
            steps.append(self._step_collapse_whitespace)
        return tuple(steps)

    # -- Public API --------------------------------------------------------- #
    def normalize(self, text: str) -> str:
        """Return the canonical orthographic form of ``text``.

        Parameters
        ----------
        text:
            Arbitrary Persian (or mixed) input. ``None``-safety is the caller's
            responsibility; an empty string returns an empty string.
        """
        if not text:
            return ""
        result = text
        for step in self._pipeline:
            result = step(result)
        return result

    # -- Individual, independently-testable steps --------------------------- #
    @staticmethod
    def _step_nfc(text: str) -> str:
        """Apply Unicode NFC composition for a stable code-point baseline."""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def _step_unify_characters(text: str) -> str:
        """Map Arabic code points to their canonical Persian equivalents."""
        return text.translate(
            {ord(src): (dst or None) for src, dst in _CHARACTER_UNIFICATION.items()}
        )

    @staticmethod
    def _step_remove_invisibles(text: str) -> str:
        """Strip bidi/zero-width control characters (preserving ZWNJ)."""
        table = {ord(ch): None for ch in _INVISIBLE_CHARACTERS}
        return text.translate(table)

    @staticmethod
    def _step_strip_emoji(text: str) -> str:
        """Remove emoji and pictographic symbols."""
        return _EMOJI_PATTERN.sub("", text)

    def _step_unify_digits(self, text: str) -> str:
        """Unify all digit forms to the configured digit system."""
        table = (
            _TO_PERSIAN_DIGITS
            if self.config.digit_system == "fa"
            else _TO_LATIN_DIGITS
        )
        return text.translate(table)

    @staticmethod
    def _step_normalize_punctuation(text: str) -> str:
        """Unify punctuation glyphs, collapse repeats, and fix spacing."""
        text = text.translate(
            {ord(src): dst for src, dst in _PUNCTUATION_UNIFICATION.items()}
        )
        # Collapse 2+ identical sentence-final marks (؟؟؟ -> ؟, !! -> !).
        text = re.sub(r"([!؟?])\1{1,}", r"\1", text)
        # Collapse ellipses of 2+ dots into a single ellipsis glyph.
        text = re.sub(r"\.{2,}", "…", text)
        # Remove whitespace *before* closing punctuation.
        text = re.sub(r"[ ]+([،؛؟!\.…:])", r"\1", text)
        return text

    @staticmethod
    def _step_expand_abbreviations(text: str) -> str:
        """Expand known abbreviations using whole-token replacement."""
        if not _ABBREVIATIONS:
            return text
        for abbr, expansion in _ABBREVIATIONS.items():
            pattern = re.compile(
                r"(?:^|(?<=\s))" + re.escape(abbr) + r"(?=\s|$)"
            )
            text = pattern.sub(expansion, text)
        return text

    @staticmethod
    def _step_zwnj_binding(text: str) -> str:
        """Normalize نیم‌فاصله usage and bind affixes with ZWNJ.

        Order matters: hygiene first (dedupe / de-space existing ZWNJ), then
        affix binding for prefixes (می/نمی) and suffixes (ها/تر families).
        """
        # Hygiene: no spaces around ZWNJ, and never more than one in a row.
        text = re.sub(rf"[ ]*{ZWNJ}[ ]*", ZWNJ, text)
        text = re.sub(rf"{ZWNJ}{{2,}}", ZWNJ, text)
        # Prefix binding: "می رود" -> "می‌رود".
        text = _PREFIX_BINDING.sub(rf"\1{ZWNJ}", text)
        # Suffix binding: "کتاب ها" -> "کتاب‌ها"; "بزرگ تر" -> "بزرگ‌تر".
        text = _SUFFIX_PLURAL_BINDING.sub(rf"{ZWNJ}\1", text)
        text = _SUFFIX_COMPARATIVE_BINDING.sub(rf"{ZWNJ}\1", text)
        return text

    @staticmethod
    def _step_collapse_whitespace(text: str) -> str:
        """Collapse runs of whitespace to single spaces and trim ends."""
        return re.sub(r"\s+", " ", text).strip()


# Module-level default instance for convenient, allocation-free reuse.
_default_normalizer: Final[PersianNormalizer] = PersianNormalizer()


def normalize(text: str) -> str:
    """Normalize ``text`` using the module-level default normalizer.

    Convenience wrapper around :meth:`PersianNormalizer.normalize` for callers
    that do not need custom configuration.
    """
    return _default_normalizer.normalize(text)
