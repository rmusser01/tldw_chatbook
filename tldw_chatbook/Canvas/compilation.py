"""Bound admission to pure compiler work without owning mutations or source caches."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from threading import BoundedSemaphore
from typing import TypeVar

from .limits import CanvasLimitError
from .models import (
    CanvasCompiledPlan,
    CanvasRenderPlan,
    CanvasRenderPlanV2,
    CanvasSourceIdentity,
)
from .profiles import ProfileSnapshot, resolve_profile

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PreparedCanvasDocument:
    """Internal result of owner-admitted preparation; never an archive or wire value."""

    plan: CanvasCompiledPlan
    source_identity: CanvasSourceIdentity
    runtime_profile: str
    parent_profile: str | None
    snapshot: ProfileSnapshot

    @classmethod
    def capture(
        cls,
        plan: CanvasCompiledPlan,
        source: str,
        *,
        parent_profile: str | None,
        snapshot: ProfileSnapshot,
    ) -> PreparedCanvasDocument:
        """Capture selection only from a freshly computed compiler result."""
        if not isinstance(plan, (CanvasRenderPlan, CanvasRenderPlanV2)):
            raise CanvasLimitError("prepared-plan-profile")
        prepared = cls(
            plan,
            CanvasSourceIdentity.from_source(source),
            plan.runtime_profile,
            parent_profile,
            snapshot,
        )
        prepared.validate(source, parent_profile=parent_profile, snapshot=snapshot)
        return prepared

    def validate(
        self,
        source: str,
        *,
        parent_profile: str | None,
        snapshot: ProfileSnapshot,
        offered: CanvasCompiledPlan | None = None,
    ) -> CanvasCompiledPlan:
        """Compare offered identities against independently prepared source/selection."""
        plan = self.plan if offered is None else offered
        if not isinstance(plan, (CanvasRenderPlan, CanvasRenderPlanV2)):
            raise CanvasLimitError("prepared-plan-profile")
        if (
            self.source_identity != CanvasSourceIdentity.from_source(source)
            or plan.source_identity != self.source_identity
        ):
            raise CanvasLimitError("prepared-plan-source")
        resolved = resolve_profile(
            snapshot,
            operation="load",
            parent_profile=self.runtime_profile,
            has_diagrams=False,
        )
        if (
            snapshot is not self.snapshot
            or parent_profile != self.parent_profile
            or plan.runtime_profile != resolved.profile_id
            or not resolved.executable
        ):
            raise CanvasLimitError("prepared-plan-profile")
        return self.plan


def prepare_canvas_document(
    source: str,
    *,
    operation: str,
    parent_profile: str | None,
    snapshot: ProfileSnapshot,
) -> CanvasCompiledPlan:
    """Inspect, select and compile in one parse inside the owner's admission slot."""
    from .compiler import _compile_document

    return _compile_document(
        source, operation=operation, parent_profile=parent_profile, snapshot=snapshot
    )


class CanvasCompilation:
    """Allow two outstanding compilations per existing authority owner, without a queue."""

    def __init__(self) -> None:
        self._slots = BoundedSemaphore(2)

    def _admit(self) -> None:
        if not self._slots.acquire(blocking=False):
            raise RuntimeError("canvas_compilation_busy")

    def _call(self, operation: Callable[[], T]) -> T:
        try:
            return operation()
        finally:
            self._slots.release()

    def run(self, operation: Callable[[], T]) -> T:
        """Compile on an existing tool worker, outside its controller lock."""

        self._admit()
        return self._call(operation)

    async def run_async(self, operation: Callable[[], T]) -> T:
        """Keep admission until the actual worker exits, even if its waiter cancels."""

        self._admit()
        try:
            future = asyncio.get_running_loop().run_in_executor(
                None, self._call, operation
            )
        except BaseException:
            self._slots.release()
            raise
        # A cancelled shield waiter cannot retrieve a later worker failure.
        # Consume it without logging; normal awaiters still receive the error.
        future.add_done_callback(
            lambda completed: None if completed.cancelled() else completed.exception()
        )
        return await asyncio.shield(future)
