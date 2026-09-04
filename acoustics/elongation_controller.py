"""Vowel elongation control (Phase 2A).

Elongation (کِشش) is the expressive lengthening of a vowel beyond its neutral
duration. In Persian it is a powerful affective device — drawn-out vowels signal
emphasis, warmth, sarcasm, or drama, and are frequently written out
orthographically (``سلاااام`` for a warm, extended *salâm*).

This controller decides **which vowels stretch and by how much**, given:

* the neutral base durations from :mod:`~ava_voice.acoustics.duration_mapper`,
* the per-phoneme elongation flags carried on each
  :class:`~ava_voice.acoustics.acoustic_schema.PhonemeUnit` (set by the aligner
  from the linguistics stress map — long vowels and orthographically repeated
  letters), and
* the global expressive ``elongation_factor`` plus the per-token ``elongate``
  decision from the prosody layer.

Elongation is confined to **syllable nuclei** (vowels): stretching consonants
would smear articulation, so only nuclei are lengthened, preserving Persian
syllable structure. Orthographic repetition (``سلاااام``) surfaces as several
successive vowel nuclei each flagged as candidates, so the total lengthening
scales naturally with how many extra letters the writer typed — a deterministic,
faithful rendering of the intended drawl.

All outputs are pure functions of the inputs — no randomness, no ML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

from ava_voice.acoustics.acoustic_schema import ElongationSegment, PhonemeUnit
from ava_voice.core.logger import get_logger
from ava_voice.emotion.emotion_schema import clamp

_logger = get_logger("acoustics.elongation")

# Hard ceiling on the applied factor, mirroring the prosody schema's bound so a
# runaway profile can never produce absurd durations.
_MAX_FACTOR: Final[float] = 2.5


@dataclass(frozen=True)
class ElongationResult:
    """The outcome of applying elongation to one token's phonemes.

    Attributes
    ----------
    durations:
        The adjusted per-phoneme durations (same order/length as the input).
    segments:
        One :class:`ElongationSegment` per phoneme that was actually lengthened.
    """

    durations: tuple[float, ...]
    segments: tuple[ElongationSegment, ...]


class ElongationController:
    """Deterministic controller of expressive vowel elongation."""

    def apply(
        self,
        units: Sequence[PhonemeUnit],
        base_durations: Sequence[float],
        elongation_factor: float,
        token_elongate: bool,
    ) -> ElongationResult:
        """Elongate eligible vowels within a single token.

        A vowel nucleus is elongated when it is flagged as an elongation
        candidate (long vowel / repeated letter) *or* the whole token was marked
        for elongation by the prosody layer. The multiplier is the global
        ``elongation_factor`` (clamped to ``[1.0, 2.5]``); non-eligible phonemes
        keep their base duration.

        Parameters
        ----------
        units:
            The token's ordered phoneme units.
        base_durations:
            Their neutral base durations (parallel to ``units``).
        elongation_factor:
            The global expressive multiplier from the prosody profile.
        token_elongate:
            Whether the prosody layer flagged this token for elongation.
        """
        factor = clamp(elongation_factor, 1.0, _MAX_FACTOR)
        adjusted: list[float] = []
        segments: list[ElongationSegment] = []

        for unit, base in zip(units, base_durations):
            eligible = (
                unit.is_vowel
                and unit.role in ("nucleus", "none")
                and (unit.is_elongation_candidate or token_elongate)
            )
            if eligible and factor > 1.0:
                stretched = base * factor
                adjusted.append(stretched)
                segments.append(
                    ElongationSegment(
                        token_id=unit.token_id,
                        phoneme_index=unit.index,
                        symbol=unit.symbol,
                        factor=factor,
                        base_ms=base,
                        added_ms=stretched - base,
                    )
                )
            else:
                adjusted.append(base)

        return ElongationResult(
            durations=tuple(adjusted), segments=tuple(segments)
        )


_default_controller: Final[ElongationController] = ElongationController()


def apply_elongation(
    units: Sequence[PhonemeUnit],
    base_durations: Sequence[float],
    elongation_factor: float,
    token_elongate: bool,
) -> ElongationResult:
    """Apply elongation using the module-level default controller."""
    return _default_controller.apply(
        units, base_durations, elongation_factor, token_elongate
    )
