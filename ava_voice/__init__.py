"""Ava Voice — production-grade AI voice infrastructure.

This top-level package wires together the core orchestration primitives
(config, logging, pipeline, routing, cache and session management) that the
domain modules (``emotion``, ``tts``, ``clone``, ``realtime`` …) build upon.

Only the foundation is implemented at this stage. Domain modules are kept as
namespace packages with explicit ``NotImplemented`` placeholders so the project
remains importable and runnable end-to-end.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
