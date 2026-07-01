"""Component registration and discovery for Ava Voice.

The :class:`ComponentRouter` is a lightweight, type-aware service registry. It
lets domain modules register named implementations (engines, encoders,
pipelines …) grouped by *category*, so the rest of the application can resolve
them at runtime without hard import dependencies.

Registration supports two ergonomic styles:

>>> router = ComponentRouter()
>>> @router.register("tts", "dummy")
... class DummyEngine: ...
>>> router.resolve("tts", "dummy") is DummyEngine
True

A process-wide :data:`global_router` is provided for convenience, mirroring the
singleton pattern used by :func:`~ava_voice.core.config.get_settings`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Generic, TypeVar

from ava_voice.core.logger import get_logger

_logger = get_logger("core.router")

T = TypeVar("T")


class RegistryError(RuntimeError):
    """Raised when a component registration or lookup fails."""


class ComponentRouter(Generic[T]):
    """Registry mapping ``(category, name)`` pairs to component objects.

    Components can be any object — classes, factories or instances. The router
    stays agnostic; consumers decide how to interpret what they resolve.
    """

    def __init__(self) -> None:
        self._registry: dict[str, dict[str, T]] = defaultdict(dict)

    def register(
        self,
        category: str,
        name: str,
        component: T | None = None,
        *,
        override: bool = False,
    ) -> T | Callable[[T], T]:
        """Register ``component`` under ``(category, name)``.

        May be used directly or as a decorator:

        * Direct: ``router.register("tts", "fast", FastEngine)``
        * Decorator: ``@router.register("tts", "fast")``

        Parameters
        ----------
        override:
            When ``False`` (default) re-registering an existing name raises
            :class:`RegistryError`; set ``True`` to replace silently.

        Returns
        -------
        The registered component (direct form) or a decorator (decorator form).
        """

        def _do_register(target: T) -> T:
            existing = self._registry[category].get(name)
            if existing is not None and not override:
                raise RegistryError(
                    f"Component '{name}' already registered under "
                    f"category '{category}'. Pass override=True to replace it."
                )
            self._registry[category][name] = target
            _logger.debug(
                "Component registered",
                extra={"category": category, "name": name},
            )
            return target

        if component is not None:
            return _do_register(component)
        return _do_register

    def resolve(self, category: str, name: str) -> T:
        """Return the component registered under ``(category, name)``.

        Raises
        ------
        RegistryError
            If no component is registered for the given pair.
        """
        try:
            return self._registry[category][name]
        except KeyError as exc:
            available = ", ".join(sorted(self._registry.get(category, {}))) or "none"
            raise RegistryError(
                f"No component '{name}' in category '{category}'. "
                f"Available: {available}."
            ) from exc

    def unregister(self, category: str, name: str) -> None:
        """Remove a component registration if present (no-op otherwise)."""
        self._registry.get(category, {}).pop(name, None)

    def contains(self, category: str, name: str) -> bool:
        """Return ``True`` if a component is registered for the pair."""
        return name in self._registry.get(category, {})

    def categories(self) -> tuple[str, ...]:
        """Return all known category names."""
        return tuple(self._registry.keys())

    def names(self, category: str) -> tuple[str, ...]:
        """Return all component names registered under ``category``."""
        return tuple(self._registry.get(category, {}).keys())

    def clear(self) -> None:
        """Remove every registration. Primarily useful in tests."""
        self._registry.clear()


#: Process-wide default router shared across the application.
global_router: ComponentRouter[object] = ComponentRouter()
