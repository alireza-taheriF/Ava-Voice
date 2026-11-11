"""End-to-end speech-plan compiler (Phase 0).

This is the first consumer of :mod:`ava_voice.core.pipeline`'s *idea*: a
single function that walks

``segment → token graph → emotion (with memory) → prosody → acoustics``

for every sentence and returns a serialisable :class:`SpeechPlan`. Domain
modules stay independently importable; this module is the orchestration the
README said was missing.

A shared :class:`~ava_voice.emotion.emotion_fusion.EmotionEngine` instance
keeps short-term memory across sentences, so a paragraph can shift or
continue affect instead of resetting on every period.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ava_voice.acoustics.acoustic_engine import AcousticEngine
from ava_voice.acoustics.acoustic_schema import AcousticProfile
from ava_voice.acoustics.speaker_baseline import DEFAULT_SPEAKER, SpeakerBaseline
from ava_voice.emotion.emotion_fusion import EmotionAnalysis, EmotionEngine
from ava_voice.linguistics.sentence_segmenter import segment_sentences
from ava_voice.linguistics.token_graph import TokenGraph, build_token_graph
from ava_voice.prosody.prosody_engine import ProsodyEngine, ProsodyResult


@dataclass(frozen=True)
class SentencePlan:
    """One sentence's full symbolic speech plan."""

    index: int
    text: str
    analysis: EmotionAnalysis
    prosody: ProsodyResult
    acoustic: AcousticProfile

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "index": self.index,
            "text": self.text,
            "analysis": self.analysis.to_dict(),
            "prosody": self.prosody.to_dict(),
            "acoustic": self.acoustic.to_dict(),
        }


@dataclass(frozen=True)
class SpeechPlan:
    """A paragraph-level plan: ordered sentence plans plus speaker baseline."""

    text: str
    sentences: tuple[SentencePlan, ...]
    speaker_f0_hz: float

    @property
    def total_duration_ms(self) -> float:
        """Sum of per-sentence acoustic durations."""
        return round(sum(s.acoustic.total_duration_ms for s in self.sentences), 4)

    @property
    def ezafe_count(self) -> int:
        """Number of modulated ezafe linkers across the paragraph."""
        return sum(len(s.acoustic.ezafe_segments) for s in self.sentences)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "text": self.text,
            "sentences": [s.to_dict() for s in self.sentences],
            "speaker_f0_hz": self.speaker_f0_hz,
            "total_duration_ms": self.total_duration_ms,
            "ezafe_count": self.ezafe_count,
        }


def plan_speech(
    text: str,
    *,
    speaker: SpeakerBaseline | None = None,
    emotion_engine: EmotionEngine | None = None,
    annotate: bool = True,
) -> SpeechPlan:
    """Compile ``text`` into a :class:`SpeechPlan`.

    Parameters
    ----------
    text:
        Raw Persian (or mixed) text. Split into sentences first.
    speaker:
        F0 baseline. Defaults to 165 Hz.
    emotion_engine:
        Optional shared engine (keeps conversational memory). A fresh
        engine is created when omitted.
    annotate:
        When ``True``, each sentence graph is annotated in place by the
        emotion / prosody / acoustic engines.
    """
    speaker = speaker or DEFAULT_SPEAKER
    emotion_engine = emotion_engine or EmotionEngine()
    prosody_engine = ProsodyEngine()
    acoustic_engine = AcousticEngine()

    planned: list[SentencePlan] = []
    for index, sentence in enumerate(segment_sentences(text)):
        graph: TokenGraph = build_token_graph(sentence)
        analysis = emotion_engine.analyze_graph(graph, annotate=annotate)
        prosody = prosody_engine.generate(graph, analysis.vector, annotate=annotate)
        acoustic = acoustic_engine.generate(
            graph,
            prosody.profile,
            vector=analysis.vector,
            speaker=speaker,
            annotate=annotate,
        )
        planned.append(
            SentencePlan(
                index=index,
                text=sentence,
                analysis=analysis,
                prosody=prosody,
                acoustic=acoustic,
            )
        )
    return SpeechPlan(
        text=text,
        sentences=tuple(planned),
        speaker_f0_hz=speaker.f0_hz,
    )
