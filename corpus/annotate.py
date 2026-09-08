"""Run the symbolic speech plan over a licensed corpus (Phase 0).

No audio is required. Each transcript is compiled; we record ezafe hits,
register labels, dominant emotion and duration. This is the first
measurement pass — it characterises *what the invention does on real
Persian sentences*, which Phase 2 will compare to aligned speech.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ava_voice.core.logger import get_logger
from ava_voice.core.speech_plan import SpeechPlan, plan_speech
from ava_voice.corpus.ingest import CORPUS_JSONL, default_data_dir, load_corpus
from ava_voice.corpus.schema import Utterance

_logger = get_logger("corpus.annotate")

ANNOTATIONS_JSONL = "annotations.jsonl"
SUMMARY_JSON = "annotation_summary.json"


def annotate_corpus(
    *,
    data_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Annotate utterances under ``data_dir`` and write a summary.

    ``limit`` caps the number of rows (useful in tests). Seed-only corpora
    are always small enough to run in full.
    """
    data_dir = Path(data_dir or default_data_dir())
    utterances = load_corpus(data_dir)
    if limit is not None:
        utterances = utterances[: max(0, limit)]

    records: list[dict[str, Any]] = []
    ezafe_total = 0
    register_counts: dict[str, int] = {}
    emotion_counts: dict[str, int] = {}
    source_ezafe: dict[str, int] = {}

    for utterance in utterances:
        try:
            plan = plan_speech(utterance.text, annotate=False)
        except Exception as exc:
            _logger.warning(
                "Skipping utterance that failed to plan",
                extra={"id": utterance.id, "error": str(exc)},
            )
            continue
        record = _record(utterance, plan)
        records.append(record)
        ezafe_total += record["ezafe_count"]
        source_ezafe[utterance.source] = (
            source_ezafe.get(utterance.source, 0) + record["ezafe_count"]
        )
        for register in record["registers"]:
            register_counts[register] = register_counts.get(register, 0) + 1
        for emotion in record["emotions"]:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / ANNOTATIONS_JSONL
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "n_annotated": len(records),
        "n_source_rows": len(utterances),
        "ezafe_count": ezafe_total,
        "ezafe_per_source": source_ezafe,
        "register_counts": register_counts,
        "emotion_counts": emotion_counts,
        "mean_duration_ms": (
            round(sum(r["total_duration_ms"] for r in records) / len(records), 2)
            if records
            else 0.0
        ),
        "corpus_file": CORPUS_JSONL,
        "annotations_file": ANNOTATIONS_JSONL,
    }
    (data_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _logger.info("Annotation summary written", extra=summary)
    return summary


def _record(utterance: Utterance, plan: SpeechPlan) -> dict[str, Any]:
    registers: list[str] = []
    emotions: list[str] = []
    allomorphs: list[str] = []
    for sentence in plan.sentences:
        dominant = sentence.analysis.vector.dominant_emotion
        emotions.append(dominant.value if dominant is not None else "neutral")
        for segment in sentence.acoustic.ezafe_segments:
            registers.append(segment.register)
            allomorphs.append(segment.allomorph)
    return {
        "id": utterance.id,
        "source": utterance.source,
        "license": utterance.license.spdx,
        "text": utterance.text,
        "n_sentences": len(plan.sentences),
        "ezafe_count": plan.ezafe_count,
        "registers": registers,
        "allomorphs": allomorphs,
        "emotions": emotions,
        "total_duration_ms": plan.total_duration_ms,
        "speaker_f0_hz": plan.speaker_f0_hz,
    }
