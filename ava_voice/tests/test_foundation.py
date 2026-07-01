"""Smoke tests validating the Ava Voice foundation.

These tests exercise the core primitives and the FastAPI wiring without
requiring any external services (Redis falls back to the in-memory backend).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ava_voice.core.cache import InMemoryCache
from ava_voice.core.config import get_settings
from ava_voice.core.pipeline import Pipeline, PipelineContext, PipelineStage
from ava_voice.core.router import ComponentRouter, RegistryError
from ava_voice.core.session import SessionManager, SessionState
from ava_voice.main import create_app


def test_settings_defaults() -> None:
    settings = get_settings()
    assert settings.app_name == "Ava Voice"
    assert settings.redis.dsn.startswith("redis://")


def test_health_endpoint() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_websocket_echo_placeholder() -> None:
    settings = get_settings()
    with TestClient(create_app()) as client:
        with client.websocket_connect(settings.websocket.path) as ws:
            created = ws.receive_json()
            assert created["type"] == "session.created"
            ws.send_text("hello")
            echoed = ws.receive_json()
            assert echoed == {
                "type": "echo",
                "session_id": created["session_id"],
                "data": "hello",
            }


@pytest.mark.asyncio
async def test_in_memory_cache_roundtrip() -> None:
    cache = InMemoryCache()
    await cache.set("k", "v")
    assert await cache.get("k") == "v"
    assert await cache.exists("k") is True
    await cache.delete("k")
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_session_lifecycle() -> None:
    manager = SessionManager(ttl_seconds=1800)
    session = await manager.create(user_id="u1")
    assert session.state is SessionState.CREATED
    await manager.touch(session.session_id)
    refreshed = await manager.get(session.session_id)
    assert refreshed is not None and refreshed.state is SessionState.ACTIVE
    await manager.close(session.session_id)
    assert await manager.get(session.session_id) is None


def test_router_registration() -> None:
    router: ComponentRouter[object] = ComponentRouter()

    @router.register("tts", "dummy")
    class DummyEngine:  # noqa: D401 - test stub
        """Test stub engine."""

    assert router.resolve("tts", "dummy") is DummyEngine
    with pytest.raises(RegistryError):
        router.resolve("tts", "missing")


@pytest.mark.asyncio
async def test_pipeline_runs_stages() -> None:
    class DoubleStage(PipelineStage):
        name = "double"

        async def infer(self, context: PipelineContext) -> PipelineContext:
            context.set("value", context.get("value", 0) * 2)
            return context

    pipeline = Pipeline([DoubleStage()], name="test")
    ctx = PipelineContext(request_id="r1", payload={"value": 21})
    result = await pipeline.run(ctx)
    assert result.get("value") == 42
    assert "double" in result.metadata["timings"]
