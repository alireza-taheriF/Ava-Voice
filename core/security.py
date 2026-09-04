"""API authentication and transport limits.

Two independent concerns, both driven by :class:`~ava_voice.core.config.Settings`:

* **Authentication** — a shared secret supplied in the ``X-API-Key`` header,
  compared in constant time. When ``AVA_API_KEY`` is unset, authentication is
  *disabled* so local development needs no setup; :func:`require_api_key` then
  admits every request. That default is deliberate but load-bearing — a
  deployment that forgets the variable is wide open, so
  :func:`warn_if_unauthenticated` logs a warning at startup and
  :func:`create_app` refuses to boot in production without one.

* **WebSocket limits** — a bounded concurrent-connection count and a maximum
  inbound frame size. Both were already declared in the settings
  (``AVA_WS_MAX_CONNECTIONS``, ``AVA_WS_MAX_MESSAGE_BYTES``) and enforced
  nowhere, so any client could open unbounded sessions and send frames of any
  size.

Close codes follow RFC 6455: ``1008`` policy violation (auth failure, capacity),
``1009`` message too big.
"""

from __future__ import annotations

import asyncio
import secrets
from typing import Final

from fastapi import Header, HTTPException, Request, WebSocket, status

from ava_voice.core.config import Settings, get_settings
from ava_voice.core.logger import get_logger

_logger = get_logger("core.security")

#: Header carrying the shared secret.
API_KEY_HEADER: Final[str] = "X-API-Key"

# RFC 6455 close codes used by the WebSocket guards.
WS_POLICY_VIOLATION: Final[int] = status.WS_1008_POLICY_VIOLATION
WS_MESSAGE_TOO_BIG: Final[int] = status.WS_1009_MESSAGE_TOO_BIG


def verify_api_key(candidate: str | None, expected: str | None) -> bool:
    """Return ``True`` if ``candidate`` authenticates against ``expected``.

    A falsy ``expected`` means authentication is disabled and everything passes.
    The comparison uses :func:`secrets.compare_digest` so a wrong key takes the
    same time to reject regardless of how many leading characters were right.

    Examples
    --------
    >>> verify_api_key(None, None)      # auth disabled
    True
    >>> verify_api_key("secret", "secret")
    True
    >>> verify_api_key("wrong", "secret")
    False
    >>> verify_api_key(None, "secret")
    False
    """
    if not expected:
        return True
    if not candidate:
        return False
    return secrets.compare_digest(candidate, expected)


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """FastAPI dependency enforcing the ``X-API-Key`` header.

    The expected key comes from ``request.app.state.settings`` when the app was
    built with explicit settings, falling back to the process-wide singleton.

    Raises
    ------
    HTTPException
        ``401`` when a key is configured and the header is missing or wrong.
    """
    settings: Settings = getattr(request.app.state, "settings", None) or get_settings()
    if verify_api_key(x_api_key, settings.api_key):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key.",
        headers={"WWW-Authenticate": API_KEY_HEADER},
    )


def warn_if_unauthenticated(settings: Settings) -> None:
    """Log a warning when the API is reachable without authentication."""
    if not settings.auth_enabled:
        _logger.warning(
            "AVA_API_KEY is not set — every endpoint is unauthenticated. "
            "Do not expose this process to an untrusted network.",
            extra={"environment": settings.environment},
        )


class ConnectionLimiter:
    """Bounded counter of concurrent WebSocket connections.

    Enforces ``AVA_WS_MAX_CONNECTIONS``. Guarded by an :class:`asyncio.Lock`, so
    the check and the increment cannot interleave — a plain compare-then-add
    would let two simultaneous handshakes both pass the check at capacity.

    Scope is the process. Behind multiple workers each has its own limit; a
    shared limit needs the cache backend, which is future work.
    """

    __slots__ = ("_active", "_limit", "_lock")

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self._limit = limit
        self._active = 0
        self._lock = asyncio.Lock()

    @property
    def limit(self) -> int:
        """Return the configured maximum number of concurrent connections."""
        return self._limit

    @property
    def active(self) -> int:
        """Return the number of currently held slots."""
        return self._active

    async def acquire(self) -> bool:
        """Take a slot, returning ``False`` when already at capacity."""
        async with self._lock:
            if self._active >= self._limit:
                return False
            self._active += 1
            return True

    async def release(self) -> None:
        """Return a slot. Never drops below zero."""
        async with self._lock:
            self._active = max(0, self._active - 1)


async def reject_oversized_frame(
    websocket: WebSocket, message: str, max_bytes: int
) -> bool:
    """Close ``websocket`` when ``message`` exceeds ``max_bytes``.

    The size is measured in UTF-8 bytes, matching how the frame travelled, not in
    characters — Persian text is 2 bytes per character, so a character-based
    check would admit roughly twice the configured payload.

    Returns
    -------
    bool
        ``True`` if the frame was rejected and the socket closed.
    """
    size = len(message.encode("utf-8"))
    if size <= max_bytes:
        return False
    _logger.warning(
        "Rejecting oversized WebSocket frame",
        extra={"size_bytes": size, "max_bytes": max_bytes},
    )
    await websocket.close(
        code=WS_MESSAGE_TOO_BIG, reason=f"Frame exceeds {max_bytes} bytes."
    )
    return True


async def authenticate_websocket(websocket: WebSocket, expected: str | None) -> bool:
    """Authenticate a WebSocket handshake before accepting it.

    The key may arrive in the ``X-API-Key`` header or, for browser clients that
    cannot set headers on a WebSocket, in an ``api_key`` query parameter. Note the
    query parameter is visible in access logs and proxy traces, so prefer the
    header wherever the client allows it.

    Returns ``True`` when the connection may proceed. On failure the socket is
    closed with ``1008`` (policy violation) *without* being accepted, so no
    application frame is ever exchanged with an unauthenticated client.
    """
    if not expected:
        return True
    candidate = websocket.headers.get(API_KEY_HEADER) or websocket.query_params.get(
        "api_key"
    )
    if verify_api_key(candidate, expected):
        return True
    _logger.warning("Rejecting unauthenticated WebSocket handshake")
    await websocket.close(code=WS_POLICY_VIOLATION, reason="Invalid or missing API key.")
    return False
