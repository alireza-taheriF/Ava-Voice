"""Calibration hook (Phase 0 write / Phase 2 fit).

Phase 0 cannot fit duration/F0 coefficients: that needs forced alignment
against licensed *audio*. This module writes the current closed-form
constants and the annotation summary side by side so a later fit can
replace the numbers without changing the functional form of the claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ava_voice.corpus.annotate import SUMMARY_JSON, annotate_corpus
from ava_voice.corpus.ingest import default_data_dir

CALIBRATION_JSON = "calibration.json"

# The claimed functional form. Values are the Phase 0 priors; Phase 2
# overwrites ``values`` and bumps ``version``.
_PRIOR: dict[str, Any] = {
    "version": "0.1.0-prior",
    "fitted": False,
    "form": {
        "duration": "α = (1-0.28f)(1+0.30 i max(0,v))(1+0.12 s) · 1.08_ye",
        "f0_st": "ΔF0 = 1.10 i max(0,v) - 0.90 f - 0.25 s",
        "energy": "g = 0.10 i (1-0.65 f)",
    },
    "values": {
        "k_formality_clip": 0.28,
        "k_warm_stretch": 0.30,
        "k_sad_linger": 0.12,
        "ye_extra": 1.08,
        "k_pitch_warm": 1.10,
        "k_pitch_formal": 0.90,
        "k_pitch_sad": 0.25,
        "k_energy": 0.10,
        "k_energy_formal": 0.65,
        "speaker_f0_hz_default": 165.0,
    },
    "notes": (
        "Priors only. Do not file these numbers as measured. Phase 2 fits "
        "them on aligned Common Voice + owned gold audio."
    ),
}


def write_calibration(*, data_dir: Path | None = None, annotate: bool = True) -> dict[str, Any]:
    """Write ``calibration.json``, optionally refreshing the annotation summary."""
    data_dir = Path(data_dir or default_data_dir())
    summary = None
    if annotate:
        summary = annotate_corpus(data_dir=data_dir)
    payload = dict(_PRIOR)
    payload["annotation_summary"] = summary
    path = data_dir / CALIBRATION_JSON
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
