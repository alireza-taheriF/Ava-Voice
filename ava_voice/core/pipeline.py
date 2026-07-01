"""Abstract inference pipeline orchestration for Ava Voice.

A :class:`Pipeline` executes an ordered sequence of :class:`PipelineStage`
objects, threading a mutable :class:`PipelineContext` through three well-defined
hooks per stage:

``preprocess`` -> ``infer`` -> ``postprocess``

This module deliberately ships *no* concrete inference logic. Domain modules
(``tts``, ``emotion``, ``clone`` …) will subclass :class:`PipelineStage` and
implement the hooks. The orchestration, timing, logging and error propagation
are provided here so every stage behaves consistently.

The API is fully asynchronous to integrate cleanly with FastAPI / WebSocket
request handling and to allow stages to await I/O (model servers, caches, GPUs).
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any

from ava_voice.core.logger import get_logger

_logger = get_logger("core.pipeline")


@dataclass
class PipelineContext:
    """Mutable state carried through a pipeline execution.

    Attributes
    ----------
    request_id:
        Correlation id used for tracing a single request across stages.
    payload:
        Arbitrary input/intermediate data. Stages read from and write to this
        mapping to communicate results downstream.
    metadata:
        Non-payload annotations (timings, model versions, flags …).
    """

    request_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key`` in the payload."""
        self.payload[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Return the payload value for ``key`` or ``default`` if absent."""
        return self.payload.get(key, default)


class PipelineStage(abc.ABC):
    """Base class for a single unit of work within a :class:`Pipeline`.

    Subclasses implement the three hooks. All hooks receive and return the
    shared :class:`PipelineContext`, allowing them to enrich state in place.
    The default :meth:`preprocess` and :meth:`postprocess` are pass-through so
    stages may override only what they need; :meth:`infer` is abstract.
    """

    #: Human-readable stage name; defaults to the subclass name.
    name: str = "stage"

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        elif self.name == "stage":
            self.name = type(self).__name__

    async def preprocess(self, context: PipelineContext) -> PipelineContext:
        """Prepare inputs prior to inference. Pass-through by default."""
        return context

    @abc.abstractmethod
    async def infer(self, context: PipelineContext) -> PipelineContext:
        """Run the core computation for this stage.

        Concrete stages must implement this hook. It is intentionally left
        unimplemented at the foundation layer.
        """
        raise NotImplementedError

    async def postprocess(self, context: PipelineContext) -> PipelineContext:
        """Transform inference outputs. Pass-through by default."""
        return context

    async def run(self, context: PipelineContext) -> PipelineContext:
        """Execute ``preprocess`` -> ``infer`` -> ``postprocess`` in order."""
        context = await self.preprocess(context)
        context = await self.infer(context)
        context = await self.postprocess(context)
        return context


class Pipeline:
    """Ordered, async orchestrator over a list of :class:`PipelineStage`.

    Stages execute sequentially, each mutating the shared context. Per-stage
    wall-clock timings are recorded into ``context.metadata['timings']`` for
    lightweight observability.
    """

    def __init__(
        self, stages: list[PipelineStage] | None = None, *, name: str = "pipeline"
    ) -> None:
        self.name = name
        self._stages: list[PipelineStage] = list(stages or [])

    def add_stage(self, stage: PipelineStage) -> Pipeline:
        """Append ``stage`` to the pipeline and return ``self`` for chaining."""
        self._stages.append(stage)
        return self

    @property
    def stages(self) -> tuple[PipelineStage, ...]:
        """Return the immutable tuple of configured stages."""
        return tuple(self._stages)

    async def run(self, context: PipelineContext) -> PipelineContext:
        """Execute every stage in order against ``context``.

        Raises
        ------
        Exception
            Re-raises any exception thrown by a stage after logging the failure
            with the offending stage name and request id for traceability.
        """
        timings: dict[str, float] = context.metadata.setdefault("timings", {})
        _logger.info(
            "Pipeline started",
            extra={"pipeline": self.name, "request_id": context.request_id},
        )

        for stage in self._stages:
            started = time.perf_counter()
            try:
                context = await stage.run(context)
            except Exception:
                _logger.exception(
                    "Pipeline stage failed",
                    extra={
                        "pipeline": self.name,
                        "stage": stage.name,
                        "request_id": context.request_id,
                    },
                )
                raise
            timings[stage.name] = round(time.perf_counter() - started, 6)

        _logger.info(
            "Pipeline finished",
            extra={
                "pipeline": self.name,
                "request_id": context.request_id,
                "timings": timings,
            },
        )
        return context
