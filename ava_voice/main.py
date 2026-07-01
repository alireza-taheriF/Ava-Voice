"""FastAPI application bootstrap for Ava Voice.

Wires the core foundation into a runnable ASGI application:

* Structured logging is configured on startup.
* Shared services (cache, session manager) are created during the lifespan and
  stored on ``app.state`` for dependency access.
* A ``/health`` endpoint reports liveness and basic runtime metadata.
* A WebSocket endpoint provides a realtime *placeholder* that echoes frames and
  keeps a session alive — the realtime pipeline itself is not implemented yet.

Run locally with::

    uvicorn ava_voice.main:app --reload

or simply ``python -m ava_voice.main``.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from ava_voice import __version__
from ava_voice.core.cache import build_cache
from ava_voice.core.config import Settings, get_settings
from ava_voice.core.logger import configure_logging, get_logger
from ava_voice.core.session import SessionManager

_logger = get_logger("main")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage startup/shutdown of shared application services."""
    settings: Settings = get_settings()
    configure_logging(settings)

    app.state.settings = settings
    app.state.cache = build_cache(settings)
    app.state.sessions = SessionManager()

    _logger.info(
        "Ava Voice starting",
        extra={"version": __version__, "environment": settings.environment},
    )
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            await app.state.cache.close()
        _logger.info("Ava Voice shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application instance.

    Kept as a factory so tests can construct isolated apps and future modules
    can mount additional routers here.
    """
    settings = settings or get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        debug=settings.debug,
        lifespan=lifespan,
    )

    @app.get("/health", tags=["system"])
    async def health() -> JSONResponse:
        """Liveness probe returning basic runtime metadata."""
        payload: dict[str, Any] = {
            "status": "ok",
            "service": settings.app_name,
            "version": __version__,
            "environment": settings.environment,
        }
        return JSONResponse(payload)

    @app.websocket(settings.websocket.path)
    async def realtime_stream(websocket: WebSocket) -> None:
        """Realtime streaming placeholder.

        Establishes a session, then echoes inbound frames back to the client.
        This exists to validate transport wiring; the realtime inference
        pipeline is intentionally not implemented yet.
        """
        await websocket.accept()
        sessions: SessionManager = websocket.app.state.sessions
        session = await sessions.create()
        _logger.info(
            "WebSocket connected", extra={"session_id": session.session_id}
        )
        await websocket.send_json(
            {"type": "session.created", "session_id": session.session_id}
        )
        try:
            while True:
                message = await websocket.receive_text()
                await sessions.touch(session.session_id)
                await websocket.send_json(
                    {
                        "type": "echo",
                        "session_id": session.session_id,
                        "data": message,
                    }
                )
        except WebSocketDisconnect:
            _logger.info(
                "WebSocket disconnected",
                extra={"session_id": session.session_id},
            )
        finally:
            await sessions.close(session.session_id)

    return app


#: Module-level ASGI application used by ``uvicorn ava_voice.main:app``.
app = create_app()


def main() -> None:
    """Console entry point: run the app with uvicorn using current settings."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "ava_voice.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
