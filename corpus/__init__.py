"""Licensed Persian corpus ingest and symbolic annotation.

Only sources with an SPDX licence are accepted. Commercial podcasts and
YouTube rips are refused on purpose.
"""

from __future__ import annotations

from ava_voice.corpus.annotate import annotate_corpus
from ava_voice.corpus.calibrate import write_calibration
from ava_voice.corpus.ingest import ingest, load_corpus
from ava_voice.corpus.schema import LicenseRecord, Utterance
from ava_voice.corpus.seed import seed_utterances

__all__ = [
    "LicenseRecord",
    "Utterance",
    "seed_utterances",
    "ingest",
    "load_corpus",
    "annotate_corpus",
    "write_calibration",
]
