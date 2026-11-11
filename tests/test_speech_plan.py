"""Sentence segmentation and the end-to-end speech plan."""

from __future__ import annotations

from ava_voice.core.speech_plan import plan_speech
from ava_voice.emotion.emotion_fusion import EmotionEngine
from ava_voice.linguistics.sentence_segmenter import segment_sentences


def test_segment_keeps_terminals() -> None:
    assert segment_sentences("سلام! خوبی؟") == ["سلام!", "خوبی؟"]
    assert segment_sentences("یک جمله بدون نقطه") == ["یک جمله بدون نقطه"]
    assert segment_sentences("   ") == []


def test_plan_speech_splits_and_uses_memory() -> None:
    text = "خیلی خوشحالم! اما خیلی ناراحتم."
    engine = EmotionEngine()
    plan = plan_speech(text, emotion_engine=engine, annotate=False)
    assert len(plan.sentences) == 2
    assert plan.sentences[0].analysis.vector.dominant_emotion is not None
    assert len(engine.memory.states()) == 2
    assert plan.total_duration_ms > 0
    assert plan.speaker_f0_hz == 165.0
    assert plan.to_dict()["ezafe_count"] == plan.ezafe_count


def test_plan_speech_detects_ezafe_across_paragraph() -> None:
    plan = plan_speech("سلام! کتابِ علی را خواندم.", annotate=False)
    assert plan.ezafe_count >= 1
    assert any(s.acoustic.ezafe_segments for s in plan.sentences)
