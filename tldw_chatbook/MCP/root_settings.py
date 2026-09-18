"""Ordered application lifetime for explicitly admitted MCP root saves."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class RootSaveRequest:
    """Submission identity, independent of the originating widget's lifetime."""

    raw_root: str
    draft_identity: tuple[object, int]
    config_path: Path
    cwd: Path


@dataclass(frozen=True, slots=True)
class RootSaveResult:
    """Compact persistence receipt without exception bodies or UI references."""

    phase: Literal["saved", "cache_refresh", "invalid", "failed", "changed"]
    stored: str | None = None
    caches_reloaded: bool = False
    cache_generation: int | None = None
    file_revision: tuple[int, int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class RootSaveState:
    """Latest admitted request and its pending or terminal outcome."""

    revision: int
    request: RootSaveRequest
    result: RootSaveResult | None = None


class RootSaveUnavailable(RuntimeError):
    """New root saves cannot enter an application that is shutting down."""


class MCPRootSaves:
    """Serialize admitted writes while UI workers only observe their outcomes."""

    def __init__(self) -> None:
        self.state: RootSaveState | None = None
        self.completion_revision = 0
        self._confirmed: RootSaveState | None = None
        self._revision = 0
        self._closed = False
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[RootSaveState]] = set()
        self._latest_task: asyncio.Task[RootSaveState] | None = None

    def submit(
        self, request: RootSaveRequest, write: Callable[[], RootSaveResult]
    ) -> asyncio.Task[RootSaveState]:
        """Register a write before yielding; callers await it through a shield."""
        if self._closed:
            raise RootSaveUnavailable("shutdown")
        if (
            self.state is not None
            and self.state.result is None
            and self.state.request == request
            and self._latest_task is not None
        ):
            return self._latest_task
        self._revision += 1
        state = RootSaveState(self._revision, request)
        task = asyncio.create_task(
            self._run(state, write), name=f"mcp-root-save:{state.revision}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._latest_task = task
        self.state = state
        return task

    async def _run(
        self, state: RootSaveState, write: Callable[[], RootSaveResult]
    ) -> RootSaveState:
        async with self._lock:
            try:
                result = await asyncio.to_thread(write)
                if not isinstance(result, RootSaveResult):
                    result = RootSaveResult("failed")
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
                result = RootSaveResult("failed")
            except Exception:  # noqa: BLE001 - keep exceptions out of retained receipts
                result = RootSaveResult("failed")
            confirmed = self._confirmed
            if (
                result.phase == "saved"
                and not result.caches_reloaded
                and confirmed is not None
                and confirmed.request.config_path == state.request.config_path
                and confirmed.result.phase == "cache_refresh"
                and confirmed.result.cache_generation == result.cache_generation
                and confirmed.result.file_revision == result.file_revision
            ):
                # A no-op does not prove stale runtime caches were repaired.
                result = replace(result, phase="cache_refresh")
            outcome = replace(state, result=result)
            if result.phase in {"saved", "cache_refresh"}:
                self._confirmed = outcome
            if self.state is not None and self.state.revision == state.revision:
                self.state = outcome
            self.completion_revision += 1
            return outcome

    def known_root(
        self,
        config_path: Path,
        cached_root: str,
        generation: int | None = None,
        file_revision: tuple[int, int, int, int] | None = None,
    ) -> str:
        """Keep a committed value visible when its cache publication failed."""
        confirmed = self._confirmed
        if (
            confirmed is not None
            and confirmed.request.config_path == config_path
            and confirmed.result.phase == "cache_refresh"
            and confirmed.result.cache_generation == generation
            and confirmed.result.file_revision == file_revision
            and confirmed.result.stored is not None
        ):
            return confirmed.result.stored
        return cached_root

    def close_admission(self) -> None:
        """Fence future requests without cancelling admitted writes."""
        self._closed = True

    async def close_and_drain(self) -> None:
        """Drain admitted work independently of cancellation of this observer."""
        self.close_admission()
        if self._tasks:
            await asyncio.shield(asyncio.gather(*tuple(self._tasks)))


def get_mcp_root_saves(host: Any) -> MCPRootSaves:
    """Lazily obtain the session owner without retaining a screen."""
    if getattr(host, "_mcp_root_saves_closed", False):
        raise RootSaveUnavailable("shutdown")
    owner = getattr(host, "_mcp_root_saves", None)
    if owner is None:
        owner = MCPRootSaves()
        host._mcp_root_saves = owner
    return owner


def root_config_file_revision(path: Path) -> tuple[int, int, int, int] | None:
    """Track external replacements/edits that do not advance runtime generation."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size
