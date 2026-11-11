"""Registry of sources this project will accept.

Unsigned or commercial media is not a source. The ingest CLI consults
:data:`ALLOWED_SOURCES` and :func:`refuse_unlicensed` before writing a row.
"""

from __future__ import annotations

from ava_voice.corpus.schema import LicenseRecord

CC0: LicenseRecord = LicenseRecord(
    spdx="CC0-1.0",
    name="Creative Commons Zero v1.0 Universal",
    url="https://creativecommons.org/publicdomain/zero/1.0/",
    allows_ml=True,
    notes="Mozilla Common Voice sentences and clips are released under CC0.",
)

OWNED_SEED: LicenseRecord = LicenseRecord(
    spdx="LicenseRef-Ava-Owned",
    name="Ava Voice owned evaluation sentences",
    url="https://github.com/alireza-taheriF/Ava-Voice",
    allows_ml=True,
    notes="Written for this repository; not copied from a podcast or video.",
)

ALLOWED_SOURCES: dict[str, LicenseRecord] = {
    "ava-seed": OWNED_SEED,
    "common-voice-fa": CC0,
}

# Hosts we will fetch sentence lists from. Not a media crawler.
ALLOWED_FETCH_HOSTS: frozenset[str] = frozenset(
    {
        "raw.githubusercontent.com",
        "github.com",
        "api.github.com",
        "huggingface.co",
        "cdn-datasets.huggingface.co",
    }
)

REFUSAL = (
    "Refusing unlicensed media. Commercial Persian podcasts and YouTube / "
    "Aparat rips cannot be ingested. Add an SPDX licence to the manifest, "
    "or drop files you own / recorded under data/local/ with a signed "
    "manifest.json."
)


def license_for(source: str) -> LicenseRecord:
    """Return the licence for a known ``source``, or raise ``ValueError``."""
    record = ALLOWED_SOURCES.get(source)
    if record is None:
        raise ValueError(REFUSAL + f" Unknown source: {source!r}")
    return record


def refuse_unlicensed(source: str, spdx: str | None) -> None:
    """Raise if ``source`` is unknown or ``spdx`` is empty."""
    if not spdx or source not in ALLOWED_SOURCES:
        raise ValueError(REFUSAL)
