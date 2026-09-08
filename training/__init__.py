"""Training / calibration entry (Phase 0 hook).

The fit against aligned audio is Phase 2. Until then this package re-exports
the prior-writing helper so ``python -m ava_voice.corpus`` and training share
one calibration file.
"""

from __future__ import annotations

from ava_voice.corpus.calibrate import write_calibration

__all__ = ["write_calibration"]
