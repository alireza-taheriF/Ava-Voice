"""Licensed utterance records for calibration (Phase 0 / Phase 1).

Every row carries an SPDX licence. The ingest pipeline refuses a source
that cannot name one — this is what keeps the corpus usable in a patent
file and in later model training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LicenseRecord:
    """A machine-readable licence attached to one source or utterance."""

    spdx: str
    name: str
    url: str
    allows_ml: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "spdx": self.spdx,
            "name": self.name,
            "url": self.url,
            "allows_ml": self.allows_ml,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LicenseRecord":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            spdx=str(data["spdx"]),
            name=str(data["name"]),
            url=str(data["url"]),
            allows_ml=bool(data.get("allows_ml", True)),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class Utterance:
    """One transcribed (and optionally recorded) Persian utterance."""

    id: str
    text: str
    source: str
    license: LicenseRecord
    locale: str = "fa"
    audio_path: str | None = None
    speaker_id: str | None = None
    duration_s: float | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "license": self.license.to_dict(),
            "locale": self.locale,
            "audio_path": self.audio_path,
            "speaker_id": self.speaker_id,
            "duration_s": self.duration_s,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Utterance":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            source=str(data["source"]),
            license=LicenseRecord.from_dict(data["license"]),
            locale=str(data.get("locale", "fa")),
            audio_path=data.get("audio_path"),
            speaker_id=data.get("speaker_id"),
            duration_s=(
                float(data["duration_s"]) if data.get("duration_s") is not None else None
            ),
            tags=tuple(data.get("tags", ())),
            metadata=dict(data.get("metadata") or {}),
        )
