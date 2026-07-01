"""Centralized application configuration for Ava Voice.

All runtime configuration is expressed as strongly-typed Pydantic settings that
are populated from environment variables (and an optional ``.env`` file). A
single cached :class:`Settings` instance is shared process-wide via
:func:`get_settings`, giving every module one authoritative source of truth for
model paths, Redis connectivity, WebSocket tuning and server parameters.

Example
-------
>>> from ava_voice.core.config import get_settings
>>> settings = get_settings()
>>> settings.redis.dsn  # doctest: +SKIP
'redis://localhost:6379/0'
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: ``.../ava_voice/core/config.py`` -> project root two levels up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

Environment = Literal["development", "staging", "production", "test"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class ModelSettings(BaseSettings):
    """Filesystem locations for model artifacts and weights.

    Paths are resolved lazily; the foundation does not load any weights yet, so
    the directories are simply advertised for downstream modules to consume.
    """

    model_config = SettingsConfigDict(env_prefix="AVA_MODEL_", extra="ignore")

    root_dir: Path = Field(
        default=PROJECT_ROOT / "models",
        description="Base directory containing all model artifacts.",
    )
    tts_dir: Path = Field(
        default=PROJECT_ROOT / "models" / "tts",
        description="Directory for text-to-speech model weights.",
    )
    emotion_dir: Path = Field(
        default=PROJECT_ROOT / "models" / "emotion",
        description="Directory for emotion / prosody model weights.",
    )
    clone_dir: Path = Field(
        default=PROJECT_ROOT / "models" / "clone",
        description="Directory for voice-cloning speaker encoders.",
    )
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto",
        description="Preferred torch device. 'auto' resolves at load time.",
    )


class RedisSettings(BaseSettings):
    """Redis connection configuration.

    The project is *Redis-ready*: connectivity is fully configured here, while
    the cache layer (:mod:`ava_voice.core.cache`) can transparently fall back to
    an in-memory backend when Redis is unavailable.
    """

    model_config = SettingsConfigDict(env_prefix="AVA_REDIS_", extra="ignore")

    host: str = Field(default="localhost", description="Redis server hostname.")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port.")
    db: int = Field(default=0, ge=0, description="Redis logical database index.")
    password: str | None = Field(default=None, description="Redis password.")
    ssl: bool = Field(default=False, description="Enable TLS for the connection.")
    default_ttl_seconds: int = Field(
        default=3600,
        ge=1,
        description="Default TTL applied to cache entries with no explicit TTL.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dsn(self) -> str:
        """Return a fully-qualified Redis DSN built from the components."""
        scheme = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"


class WebSocketSettings(BaseSettings):
    """Tuning parameters for the realtime WebSocket transport."""

    model_config = SettingsConfigDict(env_prefix="AVA_WS_", extra="ignore")

    path: str = Field(default="/ws/stream", description="WebSocket route path.")
    heartbeat_interval_seconds: float = Field(
        default=20.0,
        gt=0,
        description="Interval between keep-alive ping frames.",
    )
    max_message_bytes: int = Field(
        default=1_048_576,
        gt=0,
        description="Maximum inbound message size in bytes (default 1 MiB).",
    )
    max_connections: int = Field(
        default=1024,
        gt=0,
        description="Soft cap on concurrent WebSocket connections.",
    )
    receive_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Idle timeout before a connection is considered stale.",
    )


class Settings(BaseSettings):
    """Root application settings aggregating every configuration group.

    Nested settings groups are constructed automatically. Values are sourced,
    in order of precedence, from: explicit init kwargs, environment variables,
    a local ``.env`` file, then the declared defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="AVA_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---------------------------------------------------------
    app_name: str = Field(default="Ava Voice", description="Human-readable name.")
    environment: Environment = Field(
        default="development", description="Deployment environment."
    )
    debug: bool = Field(default=False, description="Enable debug behaviours.")

    # --- HTTP server ---------------------------------------------------------
    host: str = Field(default="0.0.0.0", description="Bind host for the API.")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port.")
    api_prefix: str = Field(default="/api/v1", description="API route prefix.")

    # --- Logging -------------------------------------------------------------
    log_level: LogLevel = Field(default="INFO", description="Root log level.")
    log_dir: Path = Field(
        default=PROJECT_ROOT / "logs",
        description="Directory where rotating log files are written.",
    )
    log_json: bool = Field(
        default=False,
        description="Emit JSON logs instead of human-readable console logs.",
    )

    # --- Nested groups -------------------------------------------------------
    models: ModelSettings = Field(default_factory=ModelSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    websocket: WebSocketSettings = Field(default_factory=WebSocketSettings)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        """Return ``True`` when running in the production environment."""
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance.

    The result is memoized so configuration is parsed from the environment
    exactly once. Call :func:`get_settings.cache_clear` in tests to force a
    reload after mutating environment variables.
    """
    return Settings()
