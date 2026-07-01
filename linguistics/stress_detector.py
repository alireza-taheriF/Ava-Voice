"""Syllabification & stress detection (Phase 1A).

Given a phoneme sequence (from :mod:`ava_voice.linguistics.phonemizer`) this
module produces a **syllable-level stress map**: it syllabifies the sequence,
assigns lexical stress, and flags emphasis and vowel-elongation candidates.

The algorithm is deterministic and rule-based, encoding well-established
Persian prosodic regularities:

* **Syllabification** — Persian permits a single-consonant onset (no initial
  clusters) and CVC / CVCC codas. Between two vowels, the final intervocalic
  consonant becomes the onset of the following syllable (maximal-onset,
  bounded to one consonant); any remaining consonants close the previous
  syllable.
* **Lexical stress** — Persian content words are predominantly **final-stressed**.
  The imperfective/negative verbal prefixes ``می``/``نمی`` are the principal
  exception and attract stress to the prefix syllable.
* **Elongation candidates** — long vowels and orthographically repeated letters
  (e.g. ``سلاااام``) mark syllables that may be lengthened for expressive
  effect by a later prosody stage.

The output feeds the emotional-prosody engine; it deliberately encodes
*candidates and positions*, not final acoustic durations.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Final, Sequence

from ava_voice.core.logger import get_logger
from ava_voice.linguistics.phonemizer import VOWEL_PLACEHOLDER, is_vowel

_logger = get_logger("linguistics.stress")

ZWNJ: Final[str] = "\u200c"

# Long vowels are elongation candidates by default.
_LONG_VOWELS: Final[frozenset[str]] = frozenset({"a", "i", "u"})

# Verbal prefixes (ZWNJ-insensitive) that attract stress to the first syllable.
_STRESS_INITIAL_PREFIXES: Final[tuple[str, ...]] = ("نمی", "می")


class StressLevel(str, enum.Enum):
    """Discrete stress levels assigned to a syllable."""

    NONE = "none"
    PRIMARY = "primary"
    EMPHATIC = "emphatic"  # primary stress further boosted by emphasis context


@dataclass(frozen=True)
class Syllable:
    """A single syllable and its prosodic annotations.

    Attributes
    ----------
    index:
        Zero-based position of the syllable within its word.
    onset / nucleus / coda:
        The syllable's phoneme components. ``nucleus`` is always a single vowel
        symbol; ``onset`` holds at most one consonant; ``coda`` may hold zero or
        more consonants.
    stress_level:
        Assigned :class:`StressLevel`.
    is_elongation_candidate:
        ``True`` if the syllable may be lengthened for expressive prosody.
    """

    index: int
    onset: tuple[str, ...]
    nucleus: str
    coda: tuple[str, ...]
    stress_level: StressLevel = StressLevel.NONE

    is_elongation_candidate: bool = False

    @property
    def phonemes(self) -> tuple[str, ...]:
        """Return the full phoneme tuple ``onset + nucleus + coda``."""
        return (*self.onset, self.nucleus, *self.coda)

    @property
    def is_stressed(self) -> bool:
        """Return ``True`` if the syllable carries any stress."""
        return self.stress_level is not StressLevel.NONE


@dataclass(frozen=True)
class StressMap:
    """Syllable-level stress map for a single word.

    Attributes
    ----------
    word:
        The source (surface) word, retained for alignment/debugging.
    syllables:
        Ordered syllables with their stress annotations.
    stressed_index:
        Index of the primary-stressed syllable, or ``-1`` if the word has no
        vowel nucleus (e.g. an isolated consonant or symbol).
    """

    word: str
    syllables: tuple[Syllable, ...]
    stressed_index: int

    @property
    def emphasis_positions(self) -> tuple[int, ...]:
        """Return indices of syllables marked as emphatic."""
        return tuple(
            s.index for s in self.syllables if s.stress_level is StressLevel.EMPHATIC
        )

    @property
    def elongation_positions(self) -> tuple[int, ...]:
        """Return indices of syllables flagged as elongation candidates."""
        return tuple(s.index for s in self.syllables if s.is_elongation_candidate)


class StressDetector:
    """Deterministic syllabifier and stress annotator."""

    def detect(
        self,
        phonemes: Sequence[str],
        *,
        word: str = "",
        emphasis: bool = False,
    ) -> StressMap:
        """Build a :class:`StressMap` from a phoneme sequence.

        Parameters
        ----------
        phonemes:
            The phoneme symbols for a single word.
        word:
            The originating surface word. Used for prefix detection (stress) and
            repeated-letter elongation detection.
        emphasis:
            When ``True`` (e.g. the word precedes ``!``), the primary-stressed
            syllable is upgraded to :attr:`StressLevel.EMPHATIC`.
        """
        raw_syllables = self._syllabify(list(phonemes))
        if not raw_syllables:
            return StressMap(word=word, syllables=(), stressed_index=-1)

        stressed_index = self._select_stressed_index(raw_syllables, word)
        has_repeat_elongation = self._has_repeated_letters(word)

        syllables: list[Syllable] = []
        for syl in raw_syllables:
            level = StressLevel.NONE
            if syl.index == stressed_index:
                level = StressLevel.EMPHATIC if emphasis else StressLevel.PRIMARY

            elongation = (
                syl.nucleus in _LONG_VOWELS
                or (has_repeat_elongation and syl.index == stressed_index)
            )
            syllables.append(
                Syllable(
                    index=syl.index,
                    onset=syl.onset,
                    nucleus=syl.nucleus,
                    coda=syl.coda,
                    stress_level=level,
                    is_elongation_candidate=elongation,
                )
            )

        return StressMap(
            word=word,
            syllables=tuple(syllables),
            stressed_index=stressed_index,
        )

    # -- Syllabification ---------------------------------------------------- #
    def _syllabify(self, phonemes: list[str]) -> list[Syllable]:
        """Partition ``phonemes`` into syllables (single-consonant onsets).

        The nucleus of each syllable is a vowel. Consonants preceding the first
        vowel form its onset; a lone intervocalic consonant becomes the onset of
        the following syllable; extra intervocalic consonants close the previous
        syllable; trailing consonants form the final coda.
        """
        vowel_positions = [i for i, p in enumerate(phonemes) if is_vowel(p)]
        if not vowel_positions:
            return []

        syllables: list[Syllable] = []
        for order, vpos in enumerate(vowel_positions):
            # Onset: the consonant immediately before this vowel, if it was not
            # already consumed as the previous syllable's coda.
            if order == 0:
                onset_start = 0
            else:
                prev_vpos = vowel_positions[order - 1]
                gap = vpos - prev_vpos - 1  # consonants between the two vowels
                # Last intervocalic consonant is this syllable's onset.
                onset_start = vpos - 1 if gap >= 1 else vpos

            onset = tuple(phonemes[onset_start:vpos])

            # Coda: consonants after the vowel up to the next syllable's onset.
            if order + 1 < len(vowel_positions):
                next_vpos = vowel_positions[order + 1]
                gap = next_vpos - vpos - 1
                # One consonant -> onset of next syllable; keep it out of coda.
                coda_end = next_vpos - 1 if gap >= 1 else vpos + 1
                coda = tuple(phonemes[vpos + 1 : coda_end])
            else:
                coda = tuple(phonemes[vpos + 1 :])  # trailing consonants

            syllables.append(
                Syllable(index=order, onset=onset, nucleus=phonemes[vpos], coda=coda)
            )

        return syllables

    # -- Stress selection --------------------------------------------------- #
    @staticmethod
    def _select_stressed_index(syllables: list[Syllable], word: str) -> int:
        """Return the index of the primary-stressed syllable.

        Rule: verbal prefixes ``می``/``نمی`` attract stress to the first
        syllable; otherwise Persian words are final-stressed.
        """
        surface = word.replace(ZWNJ, "")
        for prefix in _STRESS_INITIAL_PREFIXES:
            if surface.startswith(prefix):
                return 0
        return syllables[-1].index

    @staticmethod
    def _has_repeated_letters(word: str) -> bool:
        """Detect 3+ consecutive identical letters (expressive elongation)."""
        surface = word.replace(ZWNJ, "")
        run = 1
        for prev, cur in zip(surface, surface[1:]):
            run = run + 1 if cur == prev else 1
            if run >= 3:
                return True
        return False


_default_detector: Final[StressDetector] = StressDetector()


def detect_stress(
    phonemes: Sequence[str], *, word: str = "", emphasis: bool = False
) -> StressMap:
    """Detect stress using the module-level default detector."""
    return _default_detector.detect(phonemes, word=word, emphasis=emphasis)
