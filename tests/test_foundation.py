"""Smoke tests validating the Ava Voice foundation.

These tests exercise the core primitives and the FastAPI wiring without
requiring any external services (Redis falls back to the in-memory backend).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ava_voice.core.cache import InMemoryCache
from ava_voice.core.config import Settings, WebSocketSettings, get_settings
from ava_voice.core.pipeline import Pipeline, PipelineContext, PipelineStage
from ava_voice.core.router import ComponentRouter, RegistryError
from ava_voice.core.security import (
    API_KEY_HEADER,
    WS_MESSAGE_TOO_BIG,
    WS_POLICY_VIOLATION,
    ConnectionLimiter,
    verify_api_key,
)
from ava_voice.core.session import SessionManager, SessionState
from ava_voice.main import create_app

API_KEY = "test-secret-key"


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Keep the cached Settings singleton from leaking between tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _client(**overrides: object) -> TestClient:
    """Return a TestClient for an app built from explicit settings overrides."""
    settings = Settings(**overrides)  # type: ignore[arg-type]
    app = create_app(settings)
    app.state.settings = settings
    return TestClient(app)


def test_settings_defaults() -> None:
    settings = get_settings()
    assert settings.app_name == "Ava Voice"
    assert settings.redis.dsn.startswith("redis://")
    # Authentication is opt-in, which the security layer warns about at startup.
    assert settings.api_key is None
    assert settings.auth_enabled is False


def test_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVA_ENVIRONMENT", "production")
    monkeypatch.setenv("AVA_REDIS_PORT", "6380")
    monkeypatch.setenv("AVA_API_KEY", API_KEY)
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.is_production is True
    assert settings.redis.dsn.endswith(":6380/0")
    assert settings.auth_enabled is True


def test_settings_rejects_unknown_environment() -> None:
    with pytest.raises(ValueError):
        Settings(environment="banana")  # type: ignore[arg-type]


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


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("candidate", "expected", "allowed"),
    [
        (None, None, True),        # auth disabled
        ("anything", None, True),
        ("secret", "secret", True),
        ("wrong", "secret", False),
        (None, "secret", False),
        ("", "secret", False),
    ],
)
def test_verify_api_key(candidate: str | None, expected: str | None, allowed: bool) -> None:
    assert verify_api_key(candidate, expected) is allowed


def test_health_is_public_even_with_auth_enabled() -> None:
    # Container and load-balancer probes cannot carry credentials.
    with _client(api_key=API_KEY) as client:
        assert client.get("/health").status_code == 200


def test_protected_route_requires_key() -> None:
    with _client(api_key=API_KEY) as client:
        assert client.get("/api/v1/status").status_code == 401
        assert (
            client.get(
                "/api/v1/status", headers={API_KEY_HEADER: "wrong"}
            ).status_code
            == 401
        )
        response = client.get("/api/v1/status", headers={API_KEY_HEADER: API_KEY})
        assert response.status_code == 200
        assert response.json()["connections"]["limit"] > 0


def test_protected_route_open_when_auth_disabled() -> None:
    with _client() as client:
        assert client.get("/api/v1/status").status_code == 200


def test_production_refuses_to_start_without_a_key() -> None:
    with pytest.raises(RuntimeError, match="AVA_API_KEY"):
        create_app(Settings(environment="production"))
    # With a key it builds fine.
    assert create_app(Settings(environment="production", api_key=API_KEY)) is not None


def test_websocket_requires_key_when_configured() -> None:
    with _client(api_key=API_KEY) as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect("/ws/stream") as ws:
                ws.receive_json()
        assert refused.value.code == WS_POLICY_VIOLATION

        # Header and query-parameter forms both authenticate.
        with client.websocket_connect(
            "/ws/stream", headers={API_KEY_HEADER: API_KEY}
        ) as ws:
            assert ws.receive_json()["type"] == "session.created"
        with client.websocket_connect(f"/ws/stream?api_key={API_KEY}") as ws:
            assert ws.receive_json()["type"] == "session.created"


def test_websocket_rejects_a_wrong_key() -> None:
    with _client(api_key=API_KEY) as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect(
                "/ws/stream", headers={API_KEY_HEADER: "wrong"}
            ) as ws:
                ws.receive_json()
        assert refused.value.code == WS_POLICY_VIOLATION


# --------------------------------------------------------------------------- #
# WebSocket transport limits
# --------------------------------------------------------------------------- #
async def test_connection_limiter_bounds_slots() -> None:
    limiter = ConnectionLimiter(2)
    assert await limiter.acquire() is True
    assert await limiter.acquire() is True
    assert await limiter.acquire() is False  # at capacity
    assert limiter.active == 2

    await limiter.release()
    assert await limiter.acquire() is True

    # Releasing more than was acquired must not go negative.
    for _ in range(5):
        await limiter.release()
    assert limiter.active == 0


async def test_connection_limiter_does_not_oversubscribe_concurrently() -> None:
    """A compare-then-increment without the lock would admit more than `limit`."""
    limiter = ConnectionLimiter(5)
    granted = await asyncio.gather(*[limiter.acquire() for _ in range(50)])
    assert sum(granted) == 5
    assert limiter.active == 5


def test_connection_limiter_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError):
        ConnectionLimiter(0)


def test_websocket_rejects_oversized_frame() -> None:
    with _client(websocket=WebSocketSettings(max_message_bytes=64)) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_text("x" * 10)
            assert ws.receive_json()["data"] == "x" * 10

            ws.send_text("y" * 200)
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_json()
            assert closed.value.code == WS_MESSAGE_TOO_BIG


def test_websocket_frame_limit_counts_bytes_not_characters() -> None:
    """Persian is 2 bytes per character in UTF-8.

    A character-based check would admit roughly twice the configured payload.
    """
    with _client(websocket=WebSocketSettings(max_message_bytes=64)) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_text("س" * 40)  # 40 characters, 80 bytes
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_json()
            assert closed.value.code == WS_MESSAGE_TOO_BIG


def test_websocket_enforces_connection_limit() -> None:
    with _client(websocket=WebSocketSettings(max_connections=1)) as client:
        with client.websocket_connect("/ws/stream") as first:
            assert first.receive_json()["type"] == "session.created"
            with pytest.raises(WebSocketDisconnect) as refused:
                with client.websocket_connect("/ws/stream") as second:
                    second.receive_json()
            assert refused.value.code == WS_POLICY_VIOLATION

        # The slot is released on disconnect, so a later client is admitted.
        with client.websocket_connect("/ws/stream") as third:
            assert third.receive_json()["type"] == "session.created"


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
