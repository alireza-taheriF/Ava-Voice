"""Fetch and write only licensed Persian text (Phase 0).

Default sources
---------------
* **ava-seed** — sentences written for this repo.
* **common-voice-fa** — official Common Voice *sentence lists* from the
  ``common-voice/common-voice`` GitHub tree (CC0). These are the prompts
  speakers read; they are not ripped from a podcast or a video.

Audio clips (also CC0) are a Phase 1 download and are not fetched here:
they are large and need a separate, explicit ``--audio`` path later.

The fetcher only talks to :data:`~ava_voice.corpus.sources.ALLOWED_FETCH_HOSTS`.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from ava_voice.core.config import PROJECT_ROOT
from ava_voice.core.logger import get_logger
from ava_voice.corpus.schema import Utterance
from ava_voice.corpus.seed import seed_utterances
from ava_voice.corpus.sources import (
    ALLOWED_FETCH_HOSTS,
    CC0,
    REFUSAL,
    license_for,
)

_logger = get_logger("corpus.ingest")

DEFAULT_DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
CORPUS_JSONL: Final[str] = "corpus.jsonl"
MANIFEST_JSON: Final[str] = "manifest.json"

# Official Common Voice Persian sentence tree (CC0). Not media.
_CV_FA_LISTING: Final[str] = (
    "https://api.github.com/repos/common-voice/common-voice/contents/server/data/fa"
)
_CV_FA_RAW_PREFIX: Final[str] = (
    "https://raw.githubusercontent.com/common-voice/common-voice/main/server/data/fa/"
)

_MAX_REMOTE_SENTENCES: Final[int] = 4000
_TIMEOUT_S: Final[int] = 45


class UnlicensedSourceError(ValueError):
    """Raised when ingest is asked to take unsigned media."""


def default_data_dir() -> Path:
    """Return the on-disk corpus directory (created on write)."""
    return DEFAULT_DATA_DIR


def ingest(
    *,
    data_dir: Path | None = None,
    include_common_voice: bool = True,
    max_remote: int = _MAX_REMOTE_SENTENCES,
) -> list[Utterance]:
    """Build the working corpus and persist it under ``data_dir``.

    Always includes the owned seed. Optionally appends Common Voice
    Persian sentence lists (CC0) fetched from GitHub.
    """
    data_dir = Path(data_dir or DEFAULT_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    utterances = list(seed_utterances())
    remote_error: str | None = None
    if include_common_voice:
        try:
            remote = fetch_common_voice_sentences(limit=max_remote)
            utterances.extend(remote)
        except Exception as exc:  # network is optional; seed must still land
            remote_error = str(exc)
            _logger.warning(
                "Common Voice sentence fetch failed; keeping the owned seed",
                extra={"error": remote_error},
            )

    utterances = _dedupe(utterances)
    _write_jsonl(data_dir / CORPUS_JSONL, utterances)
    manifest = {
        "n_utterances": len(utterances),
        "sources": _count_sources(utterances),
        "licenses": sorted({u.license.spdx for u in utterances}),
        "common_voice_error": remote_error,
        "policy": REFUSAL,
    }
    (data_dir / MANIFEST_JSON).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _logger.info(
        "Corpus written",
        extra={"path": str(data_dir / CORPUS_JSONL), "n": len(utterances)},
    )
    return utterances


def fetch_common_voice_sentences(*, limit: int = _MAX_REMOTE_SENTENCES) -> list[Utterance]:
    """Download CC0 Common Voice Persian prompts from GitHub.

    Raises
    ------
    UnlicensedSourceError
        If a URL is not on the allow-list.
    urllib.error.URLError
        If the network request fails.
    """
    license_for("common-voice-fa")
    texts: list[str] = []
    for url in _common_voice_file_urls():
        _assert_allowed_url(url)
        try:
            payload = _http_get(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                _logger.info("Skipping missing Common Voice file", extra={"url": url})
                continue
            raise
        texts.extend(_lines_from_payload(payload))
    unique = _unique_keep_order(texts)
    rows: list[Utterance] = []
    for text in unique[: max(0, limit)]:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        rows.append(
            Utterance(
                id=f"cv-fa-{digest}",
                text=text,
                source="common-voice-fa",
                license=CC0,
                tags=("common-voice",),
                metadata={"origin": "common-voice/server/data/fa"},
            )
        )
    return rows


def load_corpus(data_dir: Path | None = None) -> list[Utterance]:
    """Read ``corpus.jsonl`` if it exists, otherwise return the seed only."""
    path = Path(data_dir or DEFAULT_DATA_DIR) / CORPUS_JSONL
    if not path.is_file():
        return list(seed_utterances())
    rows: list[Utterance] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(Utterance.from_dict(json.loads(line)))
    return rows


def _common_voice_file_urls() -> list[str]:
    """Return raw URLs for ``.txt`` files in the official ``fa`` sentence tree."""
    _assert_allowed_url(_CV_FA_LISTING)
    listing = json.loads(_http_get(_CV_FA_LISTING))
    names: list[str] = []
    if isinstance(listing, list):
        for item in listing:
            name = str(item.get("name", ""))
            if item.get("type") == "file" and name.endswith(".txt"):
                names.append(name)
    if not names:
        names = ["sentence-collector.txt"]
    # Collector first — it is the largest CC0 prompt list.
    names.sort(key=lambda n: (n != "sentence-collector.txt", n))
    return [_CV_FA_RAW_PREFIX + name for name in names]


def _assert_allowed_url(url: str) -> None:
    host = urlparse(url).hostname or ""
    if host not in ALLOWED_FETCH_HOSTS:
        raise UnlicensedSourceError(REFUSAL + f" Blocked host: {host}")


def _http_get(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Ava-Voice-corpus/0.1 (licensed-source ingest)"},
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
        return response.read().decode("utf-8", errors="replace")


def _lines_from_payload(payload: str) -> list[str]:
    lines: list[str] = []
    for raw in payload.splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        # Some collector files are "sentence<TAB>meta".
        text = text.split("\t", 1)[0].strip()
        if _looks_persian(text):
            lines.append(text)
    return lines


def _looks_persian(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" for char in text)


def _unique_keep_order(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _dedupe(rows: list[Utterance]) -> list[Utterance]:
    seen: set[str] = set()
    out: list[Utterance] = []
    for row in rows:
        key = row.text.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _count_sources(rows: list[Utterance]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.source] = counts.get(row.source, 0) + 1
    return counts


def _write_jsonl(path: Path, rows: list[Utterance]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
