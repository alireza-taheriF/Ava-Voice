"""Persian sentence segmentation (Phase 0).

The rest of the pipeline is sentence-scoped: one emotional vector, one
prosody profile, one acoustic plan. Feeding a whole paragraph through that
stack collapses mixed affect into a single reading. This module splits raw
text on Persian/ASCII sentence terminals so :func:`ava_voice.core.speech_plan.plan_speech`
can run the stack *per sentence* and let emotional memory carry across.

Terminals recognised: ``.`` ``!`` ``?`` ``؟`` ``…``. The delimiter stays on
the sentence that owns it (so ``!!!`` still reaches the intensity estimator).
Runs of terminals are kept together. Newlines do not split by themselves —
Persian writers often wrap mid-sentence.

The splitter is deliberately conservative: it does not try to protect
abbreviations (``دکتر.``) because the project's abbreviation expander already
runs inside the normalizer on each sentence. Deterministic, no ML.
"""

from __future__ import annotations

import re
from typing import Final

_TERMINAL_RE: Final[re.Pattern[str]] = re.compile(
    r"([^.!?؟…]+(?:[.!?؟…]+)?)",
)


def segment_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, preserving terminal punctuation.

    Empty or whitespace-only input yields an empty list. Surrounding
    whitespace on each sentence is stripped; internal spacing is kept.

    Examples
    --------
    >>> segment_sentences("سلام! خوبی؟")
    ['سلام!', 'خوبی؟']
    >>> segment_sentences("یک جمله بدون نقطه")
    ['یک جمله بدون نقطه']
    """
    raw = text or ""
    if not raw.strip():
        return []
    parts = [part.strip() for part in _TERMINAL_RE.findall(raw)]
    return [part for part in parts if part]
