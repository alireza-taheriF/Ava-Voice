"""Prosody Intelligence Layer (Phase 1C).

The deterministic, rule-based core of the *Persian Adaptive Emotional Prosody
Engine*. It converts emotional meaning (a Phase 1B
:class:`~ava_voice.emotion.emotion_schema.EmotionalVector`) plus linguistic
structure (a Phase 1A :class:`~ava_voice.linguistics.token_graph.TokenGraph`)
into concrete *speech behavior*: a bounded, serializable
:class:`~ava_voice.prosody.prosody_schema.ProsodyProfile`.

No machine learning and no randomness are used at this layer — every stage is a
pure, auditable function, mirroring :mod:`ava_voice.linguistics` and
:mod:`ava_voice.emotion`.

Deterministic pipeline
----------------------
``pitch → tempo → energy → pause → emphasis → contour → assemble → annotate``

Public surface
--------------
Each stage lives in its own single-responsibility module and is re-exported
here for ergonomic imports:

* :mod:`~ava_voice.prosody.prosody_schema` — the ``ProsodyProfile`` contract.
* :mod:`~ava_voice.prosody.pitch_mapper` — emotion → pitch height/variability.
* :mod:`~ava_voice.prosody.tempo_mapper` — emotion → speaking rate.
* :mod:`~ava_voice.prosody.energy_mapper` — emotion → vocal energy.
* :mod:`~ava_voice.prosody.pause_mapper` — punctuation/stress/emotion → pauses.
* :mod:`~ava_voice.prosody.emphasis_mapper` — token emphasis & elongation.
* :mod:`~ava_voice.prosody.contour_generator` — sentence intonation contour.
* :mod:`~ava_voice.prosody.prosody_engine` — the ``ProsodyEngine`` orchestrator.
"""

from __future__ import annotations

from ava_voice.prosody.contour_generator import (
    ContourGenerator,
    ContourResult,
    generate_contour,
)
from ava_voice.prosody.emphasis_mapper import (
    EmphasisMapper,
    EmphasisPlan,
    TokenEmphasis,
    map_emphasis,
)
from ava_voice.prosody.energy_mapper import EnergyMapper, EnergyProfile, map_energy
from ava_voice.prosody.pause_mapper import (
    PauseMapper,
    PausePlacement,
    PausePlan,
    map_pauses,
)
from ava_voice.prosody.pitch_mapper import (
    PitchMapper,
    PitchProfile,
    PitchTendency,
    map_pitch,
)
from ava_voice.prosody.prosody_engine import (
    ProsodyEngine,
    ProsodyResult,
    generate_prosody,
)
from ava_voice.prosody.prosody_schema import (
    NEUTRAL_PROFILE,
    PROFILE_FIELDS,
    ContourShape,
    ProsodyProfile,
    blend_by_emotion,
)
from ava_voice.prosody.tempo_mapper import (
    TempoMapper,
    TempoProfile,
    TempoRegularity,
    map_tempo,
)

__all__ = [
    # schema
    "ProsodyProfile",
    "ContourShape",
    "PROFILE_FIELDS",
    "NEUTRAL_PROFILE",
    "blend_by_emotion",
    # pitch
    "PitchMapper",
    "PitchProfile",
    "PitchTendency",
    "map_pitch",
    # tempo
    "TempoMapper",
    "TempoProfile",
    "TempoRegularity",
    "map_tempo",
    # energy
    "EnergyMapper",
    "EnergyProfile",
    "map_energy",
    # pause
    "PauseMapper",
    "PausePlan",
    "PausePlacement",
    "map_pauses",
    # emphasis
    "EmphasisMapper",
    "EmphasisPlan",
    "TokenEmphasis",
    "map_emphasis",
    # contour
    "ContourGenerator",
    "ContourResult",
    "generate_contour",
    # engine
    "ProsodyEngine",
    "ProsodyResult",
    "generate_prosody",
]
