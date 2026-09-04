"""Phoneme duration mapping (Phase 2A).

Timing is the skeleton of speech. This module assigns each phoneme a **base
duration in milliseconds** from three deterministic ingredients:

1. **Intrinsic phoneme length** — vowels are longer than consonants; Persian
   long vowels (``a`` = /ɑ/, ``i`` = /i/, ``u`` = /u/) are longer than the short
   vowels (``e``, ``o``); fricatives/affricates run longer than stops; the
   restored-vowel placeholder ``V`` is deliberately short.
2. **Lexical stress** — the nucleus of the stressed syllable is lengthened
   (primary stress) or lengthened further (emphatic stress), the acoustic
   correlate of Persian final-syllable prominence.
3. **Speaking rate** — the global :class:`~ava_voice.prosody.prosody_schema.ProsodyProfile`
   ``speaking_rate`` divides the duration: fast → compress, slow → stretch.

Persian syllable integrity
--------------------------
Compression is **bounded**: every phoneme has a per-class *floor* so that fast
speech never collapses a syllable below perceptibility. Crucially the vowel
nucleus floor is high enough that each syllable keeps an audible core — Persian
is a syllable-timed, vowel-rich language and losing nuclei would destroy
intelligibility. Elongation (the *opposite* direction) is applied later by
:mod:`~ava_voice.acoustics.elongation_controller`; this module only sets the
neutral base.

All outputs are pure functions of the inputs — no randomness, no ML.
"""

from __future__ import annotations

from typing import Final, Sequence

from ava_voice.acoustics.acoustic_schema import PhonemeUnit
from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import clamp
from ava_voice.linguistics.stress_detector import StressLevel

_logger = get_logger("acoustics.duration")


# Intrinsic base durations (ms at speaking_rate = 1.0).
_VOWEL_BASE_MS: Final[dict[str, float]] = {
    "a": 115.0,  # long /ɑ/
    "i": 110.0,  # long /i/
    "u": 110.0,  # long /u/
    "e": 85.0,   # short
    "o": 85.0,   # short
    "æ": 85.0,   # short /æ/ — emitted by the neural G2P backend
    "V": 70.0,   # restored-vowel placeholder (kept short/underspecified)
}
_DEFAULT_VOWEL_MS: Final[float] = 95.0

# A small consonant table; multi-char symbols (sh/ch/zh/gh) are longer.
_CONSONANT_BASE_MS: Final[dict[str, float]] = {
    "sh": 95.0,
    "ch": 95.0,
    "zh": 95.0,
    "gh": 85.0,
    "s": 85.0,
    "x": 85.0,  # خ — a fricative, so longer than the stop default
    "z": 80.0,
    "f": 78.0,
    "h": 60.0,
    "m": 68.0,
    "n": 68.0,
    "r": 55.0,
    "l": 62.0,
    "y": 55.0,
    "v": 60.0,
    "j": 88.0,  # ج — affricate
    "'": 45.0,  # glottal stop
}
_DEFAULT_CONSONANT_MS: Final[float] = 68.0  # stops: b d g k p t

# Stress lengthening multipliers applied to the stressed syllable's nucleus.
_STRESS_MULTIPLIER: Final[dict[str, float]] = {
    StressLevel.NONE.value: 1.0,
    StressLevel.PRIMARY.value: 1.20,
    StressLevel.EMPHATIC.value: 1.40,
}

# Per-class minimum durations (ms) protecting Persian syllable integrity even at
# maximum compression.
_VOWEL_FLOOR_MS: Final[float] = 55.0
_CONSONANT_FLOOR_MS: Final[float] = 28.0


class DurationMapper:
    """Deterministic phoneme base-duration mapper."""

    def base_duration(self, unit: PhonemeUnit, speaking_rate: float) -> float:
        """Return the base duration (ms) for a single phoneme ``unit``.

        Applies intrinsic length, stress lengthening (nucleus only), the global
        rate divisor, and finally the per-class integrity floor.
        """
        intrinsic = self._intrinsic(unit)
        # Stress lengthens the nucleus of the stressed syllable.
        if unit.role == "nucleus":
            intrinsic *= _STRESS_MULTIPLIER.get(unit.stress_level, 1.0)

        rate = clamp(speaking_rate, 0.5, 2.0)
        scaled = intrinsic / rate

        floor = _VOWEL_FLOOR_MS if unit.is_vowel else _CONSONANT_FLOOR_MS
        return max(scaled, floor)

    def map_units(
        self, units: Sequence[PhonemeUnit], speaking_rate: float
    ) -> tuple[float, ...]:
        """Return base durations for an ordered sequence of ``units``."""
        return tuple(self.base_duration(u, speaking_rate) for u in units)

    # -- Internals ---------------------------------------------------------- #
    @staticmethod
    def _intrinsic(unit: PhonemeUnit) -> float:
        """Return the intrinsic (rate-1.0, unstressed) duration for a phoneme."""
        if unit.is_vowel:
            return _VOWEL_BASE_MS.get(unit.symbol, _DEFAULT_VOWEL_MS)
        return _CONSONANT_BASE_MS.get(unit.symbol, _DEFAULT_CONSONANT_MS)


_default_mapper: Final[DurationMapper] = DurationMapper()


def base_duration(unit: PhonemeUnit, speaking_rate: float) -> float:
    """Compute a phoneme base duration using the module-level default mapper."""
    return _default_mapper.base_duration(unit, speaking_rate)


def map_units(
    units: Sequence[PhonemeUnit], speaking_rate: float
) -> tuple[float, ...]:
    """Map a unit sequence to base durations using the default mapper."""
    return _default_mapper.map_units(units, speaking_rate)
