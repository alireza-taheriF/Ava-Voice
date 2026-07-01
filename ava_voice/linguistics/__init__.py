"""Persian Linguistic Intelligence Layer (Phase 1A).

A deterministic, patent-grade front end that transforms raw Persian text into a
richly annotated sentence graph, ready for the future Persian Emotional Prosody
Engine. No TTS/acoustic modeling is performed at this layer.

Deterministic pipeline
-----------------------
``normalize`` → ``resolve ambiguity`` → ``phonemize`` → ``detect stress`` →
``map punctuation`` → ``assemble token graph``

Public surface
--------------
Each stage lives in its own single-responsibility module and is re-exported
here for ergonomic imports:

* :mod:`~ava_voice.linguistics.persian_normalizer`
* :mod:`~ava_voice.linguistics.ambiguity_resolver`
* :mod:`~ava_voice.linguistics.phonemizer`
* :mod:`~ava_voice.linguistics.stress_detector`
* :mod:`~ava_voice.linguistics.punctuation_mapper`
* :mod:`~ava_voice.linguistics.token_graph`
"""

from __future__ import annotations

from ava_voice.linguistics.ambiguity_resolver import (
    AmbiguityResolver,
    Resolution,
    resolve_text,
    resolve_token,
)
from ava_voice.linguistics.persian_normalizer import (
    NormalizerConfig,
    PersianNormalizer,
    normalize,
)
from ava_voice.linguistics.phonemizer import (
    PersianPhonemizer,
    PhonemeToken,
    phonemize,
    phonemize_word,
)
from ava_voice.linguistics.punctuation_mapper import (
    ControlSignal,
    PunctuationSignal,
    extract_signals,
    map_punctuation,
)
from ava_voice.linguistics.stress_detector import (
    StressDetector,
    StressLevel,
    StressMap,
    Syllable,
    detect_stress,
)
from ava_voice.linguistics.token_graph import (
    Edge,
    Node,
    NodeType,
    RelationType,
    TokenGraph,
    TokenGraphBuilder,
    build_token_graph,
)

__all__ = [
    # normalizer
    "PersianNormalizer",
    "NormalizerConfig",
    "normalize",
    # ambiguity
    "AmbiguityResolver",
    "Resolution",
    "resolve_token",
    "resolve_text",
    # phonemizer
    "PersianPhonemizer",
    "PhonemeToken",
    "phonemize",
    "phonemize_word",
    # stress
    "StressDetector",
    "StressMap",
    "Syllable",
    "StressLevel",
    "detect_stress",
    # punctuation
    "ControlSignal",
    "PunctuationSignal",
    "map_punctuation",
    "extract_signals",
    # token graph
    "TokenGraph",
    "TokenGraphBuilder",
    "Node",
    "Edge",
    "NodeType",
    "RelationType",
    "build_token_graph",
]
