"""Emotional Vector Engine (Phase 1B).

The deterministic, rule-based foundation of the future *Persian Adaptive
Emotional Prosody Engine*. It turns a Persian sentence (or a Phase 1A
:class:`~ava_voice.linguistics.token_graph.TokenGraph`) into a bounded,
serializable :class:`EmotionalVector` describing its affect.

No machine learning and no randomness are used at this layer: every stage is a
pure, auditable function, mirroring the design of
:mod:`ava_voice.linguistics`.

Deterministic pipeline
----------------------
``classify (lexical) → map (valence/arousal/dominance) → estimate (intensity)
→ assemble → contextual adjust (memory) → fuse``

Public surface
--------------
Each stage lives in its own single-responsibility module and is re-exported
here for ergonomic imports:

* :mod:`~ava_voice.emotion.emotion_schema` — the ``EmotionalVector`` contract.
* :mod:`~ava_voice.emotion.classifier` — Persian lexical/phrasal classification.
* :mod:`~ava_voice.emotion.valence_arousal` — categorical → VAD mapping.
* :mod:`~ava_voice.emotion.intensity_estimator` — expressive-intensity scoring.
* :mod:`~ava_voice.emotion.contextual_analyzer` — temporal/context shaping.
* :mod:`~ava_voice.emotion.emotion_memory` — rolling window, trend, momentum.
* :mod:`~ava_voice.emotion.emotion_fusion` — the ``EmotionEngine`` orchestrator.
"""

from __future__ import annotations

from ava_voice.emotion.classifier import (
    EmotionClassifier,
    LexicalEmotion,
    classify,
    classify_tokens,
)
from ava_voice.emotion.contextual_analyzer import (
    ContextAdjustment,
    ContextualAnalyzer,
    adjust,
)
from ava_voice.emotion.emotion_fusion import (
    EmotionAnalysis,
    EmotionEngine,
    analyze_once,
)
from ava_voice.emotion.emotion_memory import (
    DEFAULT_WINDOW,
    EmotionalDelta,
    EmotionalMomentum,
    EmotionalState,
    EmotionMemory,
)
from ava_voice.emotion.emotion_schema import (
    NEUTRAL,
    PRIMARY_DIMENSIONS,
    SECONDARY_DIMENSIONS,
    EmotionalVector,
    PrimaryEmotion,
    clamp,
)
from ava_voice.emotion.intensity_estimator import (
    IntensityEstimate,
    IntensityEstimator,
    estimate_intensity,
)
from ava_voice.emotion.valence_arousal import (
    ValenceArousalMapper,
    VADCoordinate,
    map_to_vad,
)

__all__ = [
    # schema
    "EmotionalVector",
    "PrimaryEmotion",
    "PRIMARY_DIMENSIONS",
    "SECONDARY_DIMENSIONS",
    "NEUTRAL",
    "clamp",
    # classifier
    "EmotionClassifier",
    "LexicalEmotion",
    "classify",
    "classify_tokens",
    # valence/arousal
    "ValenceArousalMapper",
    "VADCoordinate",
    "map_to_vad",
    # intensity
    "IntensityEstimator",
    "IntensityEstimate",
    "estimate_intensity",
    # context
    "ContextualAnalyzer",
    "ContextAdjustment",
    "adjust",
    # memory
    "EmotionMemory",
    "EmotionalState",
    "EmotionalDelta",
    "EmotionalMomentum",
    "DEFAULT_WINDOW",
    # fusion
    "EmotionEngine",
    "EmotionAnalysis",
    "analyze_once",
]
