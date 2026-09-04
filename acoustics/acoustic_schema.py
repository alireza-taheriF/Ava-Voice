"""Acoustic profile schema (Phase 2A).

This module defines the **deterministic contract** of the Adaptive Persian
Acoustic Layer: a single, immutable, serializable :class:`AcousticProfile` that
turns *how a sentence should sound* (a
:class:`~ava_voice.prosody.prosody_schema.ProsodyProfile`) into concrete,
frame- and phoneme-level **synthesis parameters** — the last symbolic step
before an actual vocoder/waveform stage (which is intentionally *out of scope*
here; no audio is produced).

Where :mod:`ava_voice.prosody` answers *what speech behavior* to apply, this
layer answers *the exact numbers*: how many milliseconds each phoneme lasts,
the F0 value at every 10 ms frame, the amplitude envelope, where silence sits on
the timeline, which vowels stretch and by how much.

Contract fields
---------------
``phoneme_durations``
    Ordered per-phoneme timings (:class:`PhonemeDuration`) with absolute
    start/end times on the sentence timeline.
``pitch_curve``
    Frame-level F0 contour (:class:`Curve`, unit = semitone offset from
    baseline), sampled every ``frame_hop_ms``.
``energy_curve``
    Frame-level amplitude/energy contour (:class:`Curve`, unit = normalized
    ``[0, 1]``), sampled every ``frame_hop_ms``.
``pause_segments``
    Silence spans (:class:`PauseSegment`) placed on the timeline.
``elongation_segments``
    Per-phoneme expressive lengthening records (:class:`ElongationSegment`).
``stress_boosts``
    Per-token prominence boosts (:class:`StressBoost`).
``timing_map``
    The master timeline (:class:`TimingEntry`) of tokens and pauses with
    absolute start/end times.

Design guarantees
-----------------
* **Deterministic** — no randomness, clock, or I/O anywhere in the layer.
* **Immutable & serializable** — the dataclass is frozen and every field
  round-trips through :meth:`to_dict` / :meth:`from_dict` for snapshot testing
  and for embedding into the linguistic
  :class:`~ava_voice.linguistics.token_graph.TokenGraph`.
* **Self-describing** — durations are milliseconds; curves carry their unit and
  hop; the timeline is absolute — so any downstream engine can consume it
  without hidden conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

#: Default frame hop for the pitch/energy curves, in milliseconds (100 fps).
DEFAULT_HOP_MS: Final[float] = 10.0


@dataclass(frozen=True)
class PhonemeUnit:
    """A phoneme paired with the stress/syllable context synthesis needs.

    This is the shared *intermediate* contract of the acoustic layer: the
    :mod:`~ava_voice.acoustics.phoneme_aligner` extracts a sequence of these from
    the token graph (aligning phoneme symbols to their syllable roles and stress),
    and the duration / elongation stages consume them. Hosting it in the schema
    module keeps every stage decoupled (no import cycles) while sharing one
    canonical descriptor.

    Attributes
    ----------
    index:
        Global 0-based phoneme index across the whole sentence.
    node_id:
        Id of the originating ``PHONEME`` node (for graph annotation).
    token_id:
        Id of the owning ``TOKEN`` node.
    symbol:
        The phoneme symbol.
    is_vowel:
        Whether the phoneme is a syllable nucleus.
    role:
        Syllable role: ``"onset"``, ``"nucleus"`` or ``"coda"`` (``"none"`` when
        syllable structure was unavailable).
    syllable_index:
        0-based syllable index within the token (``-1`` if unknown).
    stress_level:
        Stress level string of the owning syllable.
    is_elongation_candidate:
        Whether this phoneme (a nucleus) may be expressively lengthened.
    """

    index: int
    node_id: str | None
    token_id: str | None
    symbol: str
    is_vowel: bool
    role: str
    syllable_index: int
    stress_level: str
    is_elongation_candidate: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "index": self.index,
            "node_id": self.node_id,
            "token_id": self.token_id,
            "symbol": self.symbol,
            "is_vowel": self.is_vowel,
            "role": self.role,
            "syllable_index": self.syllable_index,
            "stress_level": self.stress_level,
            "is_elongation_candidate": self.is_elongation_candidate,
        }


@dataclass(frozen=True)
class PhonemeDuration:
    """Absolute timing of a single phoneme on the sentence timeline.

    Attributes
    ----------
    index:
        Global 0-based phoneme index across the whole sentence.
    symbol:
        The phoneme symbol (see :mod:`ava_voice.linguistics.phonemizer`).
    token_id:
        Id of the owning ``TOKEN`` node.
    duration_ms:
        Final duration in milliseconds (after rate + stress + elongation).
    start_ms, end_ms:
        Absolute start/end on the timeline (``end_ms - start_ms == duration_ms``).
    is_vowel:
        Whether the phoneme is a syllable nucleus (vowel).
    stress_level:
        The stress level of the owning syllable (``none``/``primary``/``emphatic``).
    elongated:
        Whether expressive elongation was applied to this phoneme.
    """

    index: int
    symbol: str
    token_id: str | None
    duration_ms: float
    start_ms: float
    end_ms: float
    is_vowel: bool
    stress_level: str
    elongated: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "index": self.index,
            "symbol": self.symbol,
            "token_id": self.token_id,
            "duration_ms": self.duration_ms,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "is_vowel": self.is_vowel,
            "stress_level": self.stress_level,
            "elongated": self.elongated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhonemeDuration":
        """Rebuild from a mapping produced by :meth:`to_dict`."""
        return cls(
            index=int(data["index"]),
            symbol=str(data["symbol"]),
            token_id=data.get("token_id"),
            duration_ms=float(data["duration_ms"]),
            start_ms=float(data["start_ms"]),
            end_ms=float(data["end_ms"]),
            is_vowel=bool(data["is_vowel"]),
            stress_level=str(data["stress_level"]),
            elongated=bool(data["elongated"]),
        )


@dataclass(frozen=True)
class Curve:
    """A uniformly-sampled, frame-level control curve.

    Attributes
    ----------
    unit:
        Human-readable unit label (e.g. ``"semitone"`` or ``"normalized"``).
    hop_ms:
        The time step between successive frames, in milliseconds.
    values:
        The ordered frame values.
    """

    unit: str
    hop_ms: float
    values: tuple[float, ...] = field(default_factory=tuple)

    @property
    def n_frames(self) -> int:
        """Return the number of frames in the curve."""
        return len(self.values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "unit": self.unit,
            "hop_ms": self.hop_ms,
            "n_frames": self.n_frames,
            "values": list(self.values),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Curve":
        """Rebuild from a mapping produced by :meth:`to_dict`."""
        return cls(
            unit=str(data["unit"]),
            hop_ms=float(data["hop_ms"]),
            values=tuple(float(v) for v in data.get("values", ())),
        )


@dataclass(frozen=True)
class PauseSegment:
    """A silence span on the timeline.

    Attributes
    ----------
    node_id:
        Id of the originating ``PAUSE`` node.
    signal:
        The control-signal string that produced the pause.
    start_ms, end_ms:
        Absolute start/end of the silence on the timeline.
    duration_ms:
        Length of the silence (``end_ms - start_ms``).
    category:
        Coarse role: ``"breath"``, ``"clause"`` or ``"sentence"``.
    """

    node_id: str
    signal: str
    start_ms: float
    end_ms: float
    duration_ms: float
    category: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "node_id": self.node_id,
            "signal": self.signal,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PauseSegment":
        """Rebuild from a mapping produced by :meth:`to_dict`."""
        return cls(
            node_id=str(data["node_id"]),
            signal=str(data["signal"]),
            start_ms=float(data["start_ms"]),
            end_ms=float(data["end_ms"]),
            duration_ms=float(data["duration_ms"]),
            category=str(data["category"]),
        )


@dataclass(frozen=True)
class ElongationSegment:
    """A record of expressive lengthening applied to one phoneme.

    Attributes
    ----------
    token_id:
        Id of the owning ``TOKEN`` node.
    phoneme_index:
        Global phoneme index that was elongated.
    symbol:
        The elongated phoneme symbol (always a vowel/nucleus).
    factor:
        The multiplier applied to the base duration (``>= 1.0``).
    base_ms:
        The pre-elongation duration.
    added_ms:
        The extra duration contributed by elongation (``base_ms * (factor - 1)``).
    """

    token_id: str | None
    phoneme_index: int
    symbol: str
    factor: float
    base_ms: float
    added_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "token_id": self.token_id,
            "phoneme_index": self.phoneme_index,
            "symbol": self.symbol,
            "factor": self.factor,
            "base_ms": self.base_ms,
            "added_ms": self.added_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ElongationSegment":
        """Rebuild from a mapping produced by :meth:`to_dict`."""
        return cls(
            token_id=data.get("token_id"),
            phoneme_index=int(data["phoneme_index"]),
            symbol=str(data["symbol"]),
            factor=float(data["factor"]),
            base_ms=float(data["base_ms"]),
            added_ms=float(data["added_ms"]),
        )


@dataclass(frozen=True)
class StressBoost:
    """A per-token prominence boost.

    Attributes
    ----------
    token_id:
        Id of the ``TOKEN`` node.
    surface:
        The token's surface text (for readability).
    boost:
        Prominence boost in ``[0, 1]``.
    emphasized:
        Whether the token is emphasized at all.
    """

    token_id: str
    surface: str | None
    boost: float
    emphasized: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "token_id": self.token_id,
            "surface": self.surface,
            "boost": self.boost,
            "emphasized": self.emphasized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StressBoost":
        """Rebuild from a mapping produced by :meth:`to_dict`."""
        return cls(
            token_id=str(data["token_id"]),
            surface=data.get("surface"),
            boost=float(data["boost"]),
            emphasized=bool(data["emphasized"]),
        )


@dataclass(frozen=True)
class TimingEntry:
    """A single entry on the master timeline (a token or a pause).

    Attributes
    ----------
    kind:
        ``"token"`` or ``"pause"``.
    ref_id:
        Id of the referenced ``TOKEN`` / ``PAUSE`` node.
    label:
        Surface text (tokens) or signal name (pauses), for readability.
    start_ms, end_ms:
        Absolute start/end on the timeline.
    """

    kind: str
    ref_id: str
    label: str | None
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        """Return the entry's duration (``end_ms - start_ms``)."""
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "kind": self.kind,
            "ref_id": self.ref_id,
            "label": self.label,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimingEntry":
        """Rebuild from a mapping produced by :meth:`to_dict`."""
        return cls(
            kind=str(data["kind"]),
            ref_id=str(data["ref_id"]),
            label=data.get("label"),
            start_ms=float(data["start_ms"]),
            end_ms=float(data["end_ms"]),
        )


@dataclass(frozen=True)
class AcousticProfile:
    """Immutable, serializable, synthesis-ready acoustic parameters.

    This is the deliverable of the Adaptive Persian Acoustic Layer. It is fully
    self-describing (units, hops, absolute times) so any downstream acoustic /
    vocoder stage can consume it directly.
    """

    phoneme_durations: tuple[PhonemeDuration, ...] = field(default_factory=tuple)
    pitch_curve: Curve = field(
        default_factory=lambda: Curve(unit="semitone", hop_ms=DEFAULT_HOP_MS)
    )
    energy_curve: Curve = field(
        default_factory=lambda: Curve(unit="normalized", hop_ms=DEFAULT_HOP_MS)
    )
    pause_segments: tuple[PauseSegment, ...] = field(default_factory=tuple)
    elongation_segments: tuple[ElongationSegment, ...] = field(default_factory=tuple)
    stress_boosts: tuple[StressBoost, ...] = field(default_factory=tuple)
    timing_map: tuple[TimingEntry, ...] = field(default_factory=tuple)
    total_duration_ms: float = 0.0
    frame_hop_ms: float = DEFAULT_HOP_MS

    def to_dict(self) -> dict[str, Any]:
        """Return a fully JSON-serializable representation of the profile."""
        return {
            "phoneme_durations": [p.to_dict() for p in self.phoneme_durations],
            "pitch_curve": self.pitch_curve.to_dict(),
            "energy_curve": self.energy_curve.to_dict(),
            "pause_segments": [p.to_dict() for p in self.pause_segments],
            "elongation_segments": [e.to_dict() for e in self.elongation_segments],
            "stress_boosts": [s.to_dict() for s in self.stress_boosts],
            "timing_map": [t.to_dict() for t in self.timing_map],
            "total_duration_ms": self.total_duration_ms,
            "frame_hop_ms": self.frame_hop_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AcousticProfile":
        """Rebuild an :class:`AcousticProfile` from :meth:`to_dict` output."""
        return cls(
            phoneme_durations=tuple(
                PhonemeDuration.from_dict(d)
                for d in data.get("phoneme_durations", ())
            ),
            pitch_curve=Curve.from_dict(data["pitch_curve"])
            if "pitch_curve" in data
            else Curve(unit="semitone", hop_ms=DEFAULT_HOP_MS),
            energy_curve=Curve.from_dict(data["energy_curve"])
            if "energy_curve" in data
            else Curve(unit="normalized", hop_ms=DEFAULT_HOP_MS),
            pause_segments=tuple(
                PauseSegment.from_dict(d) for d in data.get("pause_segments", ())
            ),
            elongation_segments=tuple(
                ElongationSegment.from_dict(d)
                for d in data.get("elongation_segments", ())
            ),
            stress_boosts=tuple(
                StressBoost.from_dict(d) for d in data.get("stress_boosts", ())
            ),
            timing_map=tuple(
                TimingEntry.from_dict(d) for d in data.get("timing_map", ())
            ),
            total_duration_ms=float(data.get("total_duration_ms", 0.0)),
            frame_hop_ms=float(data.get("frame_hop_ms", DEFAULT_HOP_MS)),
        )
