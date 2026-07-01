"""Core orchestration primitives for Ava Voice.

The :mod:`ava_voice.core` package exposes the foundational building blocks that
every domain module depends on:

* :mod:`~ava_voice.core.config`   — centralized, environment-driven settings.
* :mod:`~ava_voice.core.logger`   — structured console + file logging.
* :mod:`~ava_voice.core.pipeline` — abstract inference pipeline orchestration.
* :mod:`~ava_voice.core.router`   — component registration / discovery.
* :mod:`~ava_voice.core.cache`    — Redis-compatible cache abstraction.
* :mod:`~ava_voice.core.session`  — session lifecycle management.
"""

from __future__ import annotations

from ava_voice.core.cache import BaseCache, InMemoryCache, RedisCache, build_cache
from ava_voice.core.config import Settings, get_settings
from ava_voice.core.logger import configure_logging, get_logger
from ava_voice.core.pipeline import Pipeline, PipelineContext, PipelineStage
from ava_voice.core.router import ComponentRouter, RegistryError, global_router
from ava_voice.core.session import Session, SessionManager, SessionState

__all__ = [
    "BaseCache",
    "InMemoryCache",
    "RedisCache",
    "build_cache",
    "Settings",
    "get_settings",
    "configure_logging",
    "get_logger",
    "Pipeline",
    "PipelineContext",
    "PipelineStage",
    "ComponentRouter",
    "RegistryError",
    "global_router",
    "Session",
    "SessionManager",
    "SessionState",
]
