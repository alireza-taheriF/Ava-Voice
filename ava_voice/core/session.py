"""Session lifecycle management for Ava Voice.

A :class:`Session` represents a single logical interaction (an HTTP client, a
WebSocket connection, or a batch job) and carries a mutable, user-scoped state
bag alongside lifecycle timestamps.

The :class:`SessionManager` owns creation, lookup, TTL-based expiry and cleanup
of sessions. It is backed by an async-safe in-process store and is designed so
its persistence layer can later be swapped for the shared
:mod:`ava_voice.core.cache` backend without changing call sites.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import enum
import uuid
from dataclasses import dataclass, field
from typing import Any

from ava_voice.core.logger import get_logger

_logger = get_logger("core.session")


def _utcnow() -> _dt.datetime:
    """Return the current timezone-aware UTC timestamp."""
    return _dt.datetime.now(tz=_dt.timezone.utc)


class SessionState(str, enum.Enum):
    """Lifecycle states a session transitions through."""

    CREATED = "created"
    ACTIVE = "active"
    IDLE = "idle"
    CLOSED = "closed"
    EXPIRED = "expired"


@dataclass
class Session:
    """A single user session and its associated mutable state.

    Attributes
    ----------
    session_id:
        Opaque unique identifier (UUID4 hex by default).
    user_id:
        Optional identifier of the owning user/principal.
    state:
        Current :class:`SessionState`.
    data:
        Free-form, user-scoped state placeholder. Domain modules stash
        per-session context here (e.g. selected voice, conversation buffer).
    created_at / last_active_at:
        UTC lifecycle timestamps used for idle/expiry accounting.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str | None = None
    state: SessionState = SessionState.CREATED
    data: dict[str, Any] = field(default_factory=dict)
    created_at: _dt.datetime = field(default_factory=_utcnow)
    last_active_at: _dt.datetime = field(default_factory=_utcnow)

    def touch(self) -> None:
        """Mark the session as active and refresh ``last_active_at``."""
        self.last_active_at = _utcnow()
        if self.state in (SessionState.CREATED, SessionState.IDLE):
            self.state = SessionState.ACTIVE

    def is_expired(self, ttl_seconds: float) -> bool:
        """Return ``True`` if the session has been idle longer than ``ttl``."""
        age = (_utcnow() - self.last_active_at).total_seconds()
        return age > ttl_seconds


class SessionManager:
    """Async-safe manager owning the lifecycle of :class:`Session` objects.

    Parameters
    ----------
    ttl_seconds:
        Idle time after which a session is considered expired by
        :meth:`purge_expired`.
    """

    def __init__(self, ttl_seconds: float = 1800.0) -> None:
        self._sessions: dict[str, Session] = {}
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()

    async def create(
        self, user_id: str | None = None, **initial_data: Any
    ) -> Session:
        """Create, store and return a new :class:`Session`."""
        session = Session(user_id=user_id, data=dict(initial_data))
        async with self._lock:
            self._sessions[session.session_id] = session
        _logger.info(
            "Session created",
            extra={"session_id": session.session_id, "user_id": user_id},
        )
        return session

    async def get(self, session_id: str) -> Session | None:
        """Return the session for ``session_id`` or ``None`` if unknown."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def touch(self, session_id: str) -> Session | None:
        """Refresh a session's activity timestamp and return it."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.touch()
            return session

    async def close(self, session_id: str) -> None:
        """Mark a session closed and remove it from the store."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session.state = SessionState.CLOSED
            _logger.info("Session closed", extra={"session_id": session_id})

    async def purge_expired(self) -> int:
        """Remove idle-expired sessions and return the number purged."""
        async with self._lock:
            expired = [
                sid
                for sid, session in self._sessions.items()
                if session.is_expired(self._ttl_seconds)
            ]
            for sid in expired:
                self._sessions[sid].state = SessionState.EXPIRED
                del self._sessions[sid]
        if expired:
            _logger.info("Expired sessions purged", extra={"count": len(expired)})
        return len(expired)

    async def count(self) -> int:
        """Return the number of currently tracked sessions."""
        async with self._lock:
            return len(self._sessions)
