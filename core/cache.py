"""Redis-compatible cache abstraction for Ava Voice.

Defines a minimal async cache interface (:class:`BaseCache`) with two
interchangeable backends:

* :class:`InMemoryCache` — a dependency-free, TTL-aware dictionary cache used
  for local development, testing and as a graceful fallback.
* :class:`RedisCache`   — a thin async wrapper over ``redis.asyncio`` that is
  activated when the ``redis`` package is installed and a server is reachable.

The project is therefore *Redis-ready* without hard-requiring Redis at import
time: :func:`build_cache` selects the best available backend based on settings.
"""

from __future__ import annotations

import abc
import time
from typing import Any, Final

from ava_voice.core.config import RedisSettings, Settings, get_settings
from ava_voice.core.logger import get_logger

_logger = get_logger("core.cache")

_MISSING: Final = object()


class BaseCache(abc.ABC):
    """Abstract async key/value cache contract.

    Implementations must be safe to use from asynchronous request handlers.
    Values are opaque to the cache; serialization (if any) is the backend's
    responsibility.
    """

    @abc.abstractmethod
    async def get(self, key: str, default: Any = None) -> Any:
        """Return the value for ``key`` or ``default`` if missing/expired."""

    @abc.abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store ``value`` under ``key`` with an optional TTL in seconds."""

    @abc.abstractmethod
    async def delete(self, key: str) -> None:
        """Remove ``key`` from the cache (no-op if absent)."""

    @abc.abstractmethod
    async def exists(self, key: str) -> bool:
        """Return ``True`` if ``key`` is present and not expired."""

    @abc.abstractmethod
    async def clear(self) -> None:
        """Remove every entry owned by this cache."""

    async def close(self) -> None:
        """Release any underlying resources. Default implementation is a no-op."""
        return None


class InMemoryCache(BaseCache):
    """Process-local, TTL-aware in-memory cache.

    Intended for development, tests and as a fallback when Redis is not
    available. Not shared across processes and not persistent.
    """

    def __init__(self, default_ttl: int | None = None) -> None:
        self._default_ttl = default_ttl
        # Maps key -> (value, expiry_epoch_or_None).
        self._store: dict[str, tuple[Any, float | None]] = {}

    def _is_expired(self, expiry: float | None) -> bool:
        return expiry is not None and expiry <= time.monotonic()

    async def get(self, key: str, default: Any = None) -> Any:
        entry = self._store.get(key, _MISSING)
        if entry is _MISSING:
            return default
        value, expiry = entry  # type: ignore[misc]
        if self._is_expired(expiry):
            self._store.pop(key, None)
            return default
        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expiry = time.monotonic() + effective_ttl if effective_ttl else None
        self._store[key] = (value, expiry)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        return await self.get(key, _MISSING) is not _MISSING

    async def clear(self) -> None:
        self._store.clear()


class RedisCache(BaseCache):
    """Async cache backed by Redis via ``redis.asyncio``.

    The ``redis`` dependency is imported lazily so the foundation remains
    importable without it. Instantiate via :meth:`from_settings` to build a
    client from :class:`~ava_voice.core.config.RedisSettings`.
    """

    def __init__(self, client: Any, default_ttl: int | None = None) -> None:
        self._client = client
        self._default_ttl = default_ttl

    @classmethod
    def from_settings(cls, settings: RedisSettings) -> RedisCache:
        """Build a :class:`RedisCache` from Redis settings.

        Raises
        ------
        RuntimeError
            If the optional ``redis`` package is not installed.
        """
        try:
            from redis import asyncio as aioredis  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "The 'redis' package is required for RedisCache. "
                "Install it with `pip install redis`."
            ) from exc

        client = aioredis.from_url(settings.dsn, decode_responses=True)
        return cls(client, default_ttl=settings.default_ttl_seconds)

    async def get(self, key: str, default: Any = None) -> Any:
        value = await self._client.get(key)
        return default if value is None else value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        await self._client.set(key, value, ex=effective_ttl)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def exists(self, key: str) -> bool:
        return bool(await self._client.exists(key))

    async def clear(self) -> None:
        await self._client.flushdb()

    async def close(self) -> None:
        await self._client.aclose()


def build_cache(settings: Settings | None = None) -> BaseCache:
    """Return the best available cache backend for the current settings.

    Attempts to construct a :class:`RedisCache`; on any failure (missing
    dependency or misconfiguration) it logs a warning and gracefully falls back
    to :class:`InMemoryCache`. Actual connectivity is verified lazily on first
    use by the Redis client itself.
    """
    settings = settings or get_settings()
    try:
        cache = RedisCache.from_settings(settings.redis)
        _logger.info("Using RedisCache backend", extra={"dsn": settings.redis.dsn})
        return cache
    except Exception as exc:
        _logger.warning(
            "Falling back to InMemoryCache: %s",
            exc,
            extra={"backend": "in_memory"},
        )
        return InMemoryCache(default_ttl=settings.redis.default_ttl_seconds)
