"""Adaptive Persian Acoustic Layer (Phase 2A).

The deterministic, rule-based stage that converts *speech behavior* (a Phase 1C
:class:`~ava_voice.prosody.prosody_schema.ProsodyProfile`) plus linguistic
structure (a Phase 1A :class:`~ava_voice.linguistics.token_graph.TokenGraph`)
into concrete, synthesis-ready **acoustic parameters**: an
:class:`~ava_voice.acoustics.acoustic_schema.AcousticProfile`.

No machine learning, no randomness, and **no waveform generation** happen here —
this layer stops at frame- and phoneme-level parameters, mirroring the
deterministic design of :mod:`ava_voice.linguistics`, :mod:`ava_voice.emotion`
and :mod:`ava_voice.prosody`.

Deterministic pipeline
----------------------
``align (phonemes+stress) → base durations → elongation → pause shaping
→ assemble timeline → pitch curve → energy curve → assemble → annotate``

Public surface
--------------
Each stage lives in its own single-responsibility module and is re-exported
here for ergonomic imports:

* :mod:`~ava_voice.acoustics.acoustic_schema` — the ``AcousticProfile`` contract.
* :mod:`~ava_voice.acoustics.pitch_envelope` — frame-level F0 curve.
* :mod:`~ava_voice.acoustics.duration_mapper` — phoneme base durations.
* :mod:`~ava_voice.acoustics.energy_envelope` — frame-level amplitude curve.
* :mod:`~ava_voice.acoustics.pause_shaper` — pause metadata → silence segments.
* :mod:`~ava_voice.acoustics.elongation_controller` — expressive vowel elongation.
* :mod:`~ava_voice.acoustics.phoneme_aligner` — phoneme/stress/timing alignment.
* :mod:`~ava_voice.acoustics.acoustic_engine` — the ``AcousticEngine`` orchestrator.
"""

from __future__ import annotations

from ava_voice.acoustics.acoustic_engine import (
    AcousticEngine,
    generate_acoustics,
)
from ava_voice.acoustics.acoustic_schema import (
    DEFAULT_HOP_MS,
    AcousticProfile,
    Curve,
    ElongationSegment,
    PauseSegment,
    PhonemeDuration,
    PhonemeUnit,
    StressBoost,
    TimingEntry,
)
from ava_voice.acoustics.duration_mapper import (
    DurationMapper,
    base_duration,
    map_units,
)
from ava_voice.acoustics.elongation_controller import (
    ElongationController,
    ElongationResult,
    apply_elongation,
)
from ava_voice.acoustics.energy_envelope import (
    EnergyEnvelopeGenerator,
    build_energy_curve,
)
from ava_voice.acoustics.pause_shaper import (
    PauseShape,
    PauseShaper,
    shape_pause,
)
from ava_voice.acoustics.phoneme_aligner import (
    AlignmentResult,
    PauseSlot,
    PhonemeAligner,
    TokenSlot,
    extract_sequence,
)
from ava_voice.acoustics.pitch_envelope import (
    PitchEnvelopeGenerator,
    build_pitch_curve,
)

__all__ = [
    # schema
    "AcousticProfile",
    "PhonemeUnit",
    "PhonemeDuration",
    "Curve",
    "PauseSegment",
    "ElongationSegment",
    "StressBoost",
    "TimingEntry",
    "DEFAULT_HOP_MS",
    # duration
    "DurationMapper",
    "base_duration",
    "map_units",
    # elongation
    "ElongationController",
    "ElongationResult",
    "apply_elongation",
    # pause
    "PauseShaper",
    "PauseShape",
    "shape_pause",
    # pitch
    "PitchEnvelopeGenerator",
    "build_pitch_curve",
    # energy
    "EnergyEnvelopeGenerator",
    "build_energy_curve",
    # aligner
    "PhonemeAligner",
    "AlignmentResult",
    "TokenSlot",
    "PauseSlot",
    "extract_sequence",
    # engine
    "AcousticEngine",
    "generate_acoustics",
]
