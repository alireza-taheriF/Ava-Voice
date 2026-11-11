"""Licensed corpus ingest: seed is always present; unsigned sources fail."""

from __future__ import annotations

import pytest

from ava_voice.corpus.annotate import annotate_corpus
from ava_voice.corpus.calibrate import write_calibration
from ava_voice.corpus.ingest import ingest, load_corpus
from ava_voice.corpus.seed import seed_utterances
from ava_voice.corpus.sources import REFUSAL, refuse_unlicensed


def test_seed_has_ezafe_and_no_ezafe_pairs() -> None:
    rows = seed_utterances()
    tags = {tag for row in rows for tag in row.tags}
    assert "ezafe" in tags
    assert "no-ezafe" in tags
    assert "formality" in tags
    assert all(row.license.spdx for row in rows)


def test_refuse_unlicensed_podcast() -> None:
    with pytest.raises(ValueError, match="Refusing"):
        refuse_unlicensed("youtube-rip", None)
    with pytest.raises(ValueError, match="Refusing"):
        refuse_unlicensed("unknown", "CC0-1.0")


def test_offline_ingest_and_annotate(tmp_path) -> None:
    rows = ingest(data_dir=tmp_path, include_common_voice=False)
    assert len(rows) == len(seed_utterances())
    loaded = load_corpus(tmp_path)
    assert len(loaded) == len(rows)
    summary = annotate_corpus(data_dir=tmp_path)
    assert summary["n_annotated"] == len(rows)
    assert summary["ezafe_count"] >= 1
    cal = write_calibration(data_dir=tmp_path, annotate=False)
    assert cal["fitted"] is False
    assert (tmp_path / "calibration.json").is_file()
    assert REFUSAL


def test_blocked_host_is_rejected() -> None:
    from ava_voice.corpus.ingest import UnlicensedSourceError, _assert_allowed_url

    with pytest.raises(UnlicensedSourceError):
        _assert_allowed_url("https://www.youtube.com/watch?v=abc")
