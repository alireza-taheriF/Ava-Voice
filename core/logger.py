"""Structured logging for Ava Voice.

Provides a single :func:`configure_logging` entry point that wires up both a
console handler and a rotating file handler, plus :func:`get_logger` for
retrieving namespaced loggers throughout the codebase.

Two formats are supported:

* **Human-readable** (default) — concise, colour-free, timestamped lines that
  read well in a terminal or ``docker logs``.
* **JSON** — one structured object per line, suitable for ingestion by log
  aggregators (Loki, ELK, Datadog …). Enabled via ``AVA_LOG_JSON=true``.

Configuration is idempotent: calling :func:`configure_logging` multiple times
will not attach duplicate handlers.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import logging.handlers
from pathlib import Path

from ava_voice.core.config import Settings, get_settings

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

# Standard ``LogRecord`` attributes, used to isolate user-supplied ``extra``
# fields when serializing structured JSON logs.
_RESERVED_RECORD_KEYS = frozenset(logging.makeLogRecord({}).__dict__.keys()) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Serialize log records as single-line JSON objects.

    Any keyword arguments passed via ``logger.info(..., extra={...})`` that are
    not standard record attributes are merged into the emitted object, making
    contextual logging (e.g. ``session_id``) trivial.
    """

    def format(self, record: logging.LogRecord) -> str:
        timestamp = _dt.datetime.fromtimestamp(
            record.created, tz=_dt.timezone.utc
        ).isoformat()
        payload: dict[str, object] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_KEYS and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str, ensure_ascii=False)


def _build_console_handler(json_logs: bool) -> logging.Handler:
    """Create the stream handler targeting stdout."""
    handler = logging.StreamHandler()
    formatter: logging.Formatter = (
        JsonFormatter()
        if json_logs
        else logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT)
    )
    handler.setFormatter(formatter)
    return handler


def _build_file_handler(log_dir: Path, json_logs: bool) -> logging.Handler:
    """Create a size-based rotating file handler under ``log_dir``."""
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "ava_voice.log",
        maxBytes=10 * 1024 * 1024,  # 10 MiB per file.
        backupCount=5,
        encoding="utf-8",
    )
    formatter: logging.Formatter = (
        JsonFormatter()
        if json_logs
        else logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT)
    )
    handler.setFormatter(formatter)
    return handler


def configure_logging(settings: Settings | None = None) -> logging.Logger:
    """Configure the ``ava_voice`` root logger and return it.

    Parameters
    ----------
    settings:
        Optional explicit settings. When omitted, the process-wide cached
        settings from :func:`~ava_voice.core.config.get_settings` are used.

    Returns
    -------
    logging.Logger
        The configured ``ava_voice`` namespace logger.

    Notes
    -----
    The function is idempotent — handlers are only attached once per process.
    """
    settings = settings or get_settings()
    logger = logging.getLogger("ava_voice")
    logger.setLevel(settings.log_level)
    # Do not propagate to the root logger to avoid duplicated emission.
    logger.propagate = False

    if getattr(logger, "_ava_configured", False):
        return logger

    logger.addHandler(_build_console_handler(settings.log_json))
    logger.addHandler(_build_file_handler(settings.log_dir, settings.log_json))

    # Mark as configured so repeated calls become no-ops.
    logger._ava_configured = True  # type: ignore[attr-defined]
    logger.debug("Logging configured", extra={"log_level": settings.log_level})
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced child logger under the ``ava_voice`` root.

    Parameters
    ----------
    name:
        Dotted suffix for the child logger (e.g. ``"core.session"``). When
        ``None``, the ``ava_voice`` root logger itself is returned.
    """
    if not name:
        return logging.getLogger("ava_voice")
    return logging.getLogger(f"ava_voice.{name}")
