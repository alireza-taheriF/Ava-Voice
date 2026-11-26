"""Voice cloning: record your own voice and train a personal Ava voice.

* :mod:`.record` — browser recording app with a spoken consent step and live QC;
* :mod:`.dataset` — QC, trimming, Whisper check, Piper training set in Ava's phonemes;
* :mod:`.install` — put a trained voice into Ava and smoke-test it.

Training itself runs Piper on a GPU (see ``TRAINING.md``).
"""

from __future__ import annotations

__all__: list[str] = ["record", "dataset", "install", "audio"]
