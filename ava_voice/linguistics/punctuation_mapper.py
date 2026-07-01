"""Punctuation → speech-control-signal mapping (Phase 1A).

Punctuation is the orthographic surface of prosody. This module maps Persian
(and unified Latin) punctuation marks to a small, closed vocabulary of
**speech control signals** that later prosody/TTS stages consume. The mapping
is deterministic and total: every recognized mark yields exactly one signal.

The vocabulary is intentionally engine-agnostic (pauses, pitch and emphasis
directives) so it can drive any downstream acoustic model without leaking
model-specific parameters into this linguistic layer.

Examples
--------
>>> map_punctuation(".")
<ControlSignal.PAUSE_MEDIUM: 'pause_medium'>
>>> map_punctuation("!")
<ControlSignal.EMPHASIS_BOOST: 'emphasis_boost'>
>>> map_punctuation("؟")
<ControlSignal.RISING_PITCH: 'rising_pitch'>
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final


class ControlSignal(str, enum.Enum):
    """Closed vocabulary of prosodic control signals.

    String-valued so signals serialize cleanly into the token graph and JSON.
    """

    PAUSE_SHORT = "pause_short"
    PAUSE_MEDIUM = "pause_medium"
    PAUSE_LONG = "pause_long"
    RISING_PITCH = "rising_pitch"
    FALLING_PITCH = "falling_pitch"
    EMPHASIS_BOOST = "emphasis_boost"
    PITCH_LOWER = "pitch_lower"
    TRAILING_OFF = "trailing_off"


# Deterministic, total mapping over the recognized punctuation set. Latin marks
# are included defensively in case normalization is skipped upstream.
_PUNCTUATION_MAP: Final[dict[str, ControlSignal]] = {
    ".": ControlSignal.PAUSE_MEDIUM,
    "،": ControlSignal.PAUSE_SHORT,
    ",": ControlSignal.PAUSE_SHORT,
    "؛": ControlSignal.PAUSE_MEDIUM,
    ";": ControlSignal.PAUSE_MEDIUM,
    ":": ControlSignal.PAUSE_SHORT,
    "!": ControlSignal.EMPHASIS_BOOST,
    "؟": ControlSignal.RISING_PITCH,
    "?": ControlSignal.RISING_PITCH,
    "…": ControlSignal.TRAILING_OFF,
    "—": ControlSignal.PAUSE_SHORT,
    "–": ControlSignal.PAUSE_SHORT,
    "-": ControlSignal.PAUSE_SHORT,
    "(": ControlSignal.PITCH_LOWER,
    ")": ControlSignal.PITCH_LOWER,
    "«": ControlSignal.PITCH_LOWER,
    "»": ControlSignal.PITCH_LOWER,
    "\n": ControlSignal.PAUSE_LONG,
}


@dataclass(frozen=True)
class PunctuationSignal:
    """A punctuation occurrence resolved to a control signal.

    Attributes
    ----------
    mark:
        The source punctuation character.
    signal:
        The mapped :class:`ControlSignal`.
    position:
        Character offset of the mark in the source text (for alignment).
    """

    mark: str
    signal: ControlSignal
    position: int


def is_punctuation(char: str) -> bool:
    """Return ``True`` if ``char`` is a recognized punctuation mark."""
    return char in _PUNCTUATION_MAP


def map_punctuation(mark: str) -> ControlSignal | None:
    """Map a single punctuation ``mark`` to its :class:`ControlSignal`.

    Returns ``None`` for unrecognized characters so callers can decide how to
    treat unknown marks (drop, warn, or pass through).
    """
    return _PUNCTUATION_MAP.get(mark)


def extract_signals(text: str) -> list[PunctuationSignal]:
    """Scan ``text`` and return every recognized punctuation signal in order.

    The scan is position-preserving so downstream stages can align pauses and
    pitch directives to token boundaries.
    """
    signals: list[PunctuationSignal] = []
    for index, char in enumerate(text):
        signal = _PUNCTUATION_MAP.get(char)
        if signal is not None:
            signals.append(
                PunctuationSignal(mark=char, signal=signal, position=index)
            )
    return signals
