"""Ordered application lifetime for the two local MCP configuration controls."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

ConfigKey = Literal["workspace_root", "local_tools_enabled"]


@dataclass(frozen=True, slots=True)
class ConfigSaveRequest:
    """Submission identity, independent of the originating widget's lifetime."""

    value: str | bool
    draft_identity: tuple[object, int] | None
    config_path: Path
    cwd: Path | None = None
    key: ConfigKey = "workspace_root"


@dataclass(frozen=True, slots=True)
class ConfigSaveResult:
    """Compact persistence receipt without exception bodies or UI references."""

    phase: Literal["saved", "cache_refresh", "invalid", "failed", "changed"]
    stored: str | bool | None = None
    caches_reloaded: bool = False
    cache_generation: int | None = None
    file_revision: tuple[int, int, int, int] | None = None
    previous_generation: int | None = None
    previous_file_revision: tuple[int, int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class ConfigSaveState:
    """Latest admitted request and its pending or terminal outcome."""

    revision: int
    request: ConfigSaveRequest
    result: ConfigSaveResult | None = None


class ConfigSaveUnavailable(RuntimeError):
    """New local configuration saves cannot enter a closing application."""


class MCPLocalConfigSaves:
    """Serialize admitted writes while UI workers only observe their outcomes."""

    def __init__(self) -> None:
        self.state: ConfigSaveState | None = None
        self._states: dict[ConfigKey, ConfigSaveState] = {}
        self.completion_revision = 0
        self._confirmed: dict[ConfigKey, ConfigSaveState] = {}
        self._revision = 0
        self._closed = False
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[ConfigSaveState]] = set()
        self._latest_task: asyncio.Task[ConfigSaveState] | None = None

    def submit(
        self, request: ConfigSaveRequest, write: Callable[[], ConfigSaveResult]
    ) -> asyncio.Task[ConfigSaveState]:
        """Register a write before yielding; callers await it through a shield.

        Args:
            request: Immutable value, destination and draft identity to save.
            write: Synchronous writer run off-thread after earlier writes finish.

        Returns:
            The admitted task, reused for an identical latest pending request.

        Raises:
            ConfigSaveUnavailable: Admission has closed for shutdown.
            RuntimeError: No application event loop is running.
        """
        if self._closed:
            raise ConfigSaveUnavailable("shutdown")
        if (
            self.state is not None
            and self.state.result is None
            and self.state.request == request
            and self._latest_task is not None
        ):
            return self._latest_task
        self._revision += 1
        state = ConfigSaveState(self._revision, request)
        task = asyncio.create_task(
            self._run(state, write), name=f"mcp-config-save:{state.revision}"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._latest_task = task
        self.state = state
        self._states[request.key] = state
        return task

    def state_for(self, key: ConfigKey) -> ConfigSaveState | None:
        """Return the latest request for one control, independent of its sibling.

        Args:
            key: Configuration control whose latest state is requested.

        Returns:
            Its pending or completed state, or None before any submission.
        """
        return self._states.get(key)

    async def _run(
        self, state: ConfigSaveState, write: Callable[[], ConfigSaveResult]
    ) -> ConfigSaveState:
        async with self._lock:
            try:
                result = await asyncio.to_thread(write)
                if not isinstance(result, ConfigSaveResult):
                    result = ConfigSaveResult("failed")
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
                result = ConfigSaveResult("failed")
            except Exception:  # noqa: BLE001 - keep exceptions out of retained receipts
                result = ConfigSaveResult("failed")
            confirmed = self._confirmed.get(state.request.key)
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
                self._advance_sibling_receipts(outcome)
                self._confirmed[state.request.key] = outcome
            if self.state is not None and self.state.revision == state.revision:
                self.state = outcome
            if self._states[state.request.key].revision == state.revision:
                self._states[state.request.key] = outcome
            self.completion_revision += 1
            return outcome

    def _advance_sibling_receipts(self, outcome: ConfigSaveState) -> None:
        """Carry unchanged sibling values across a known atomic single-key save."""
        result = outcome.result
        if result.previous_file_revision is None:
            return
        for key, confirmed in tuple(self._confirmed.items()):
            prior = confirmed.result
            if (
                key != outcome.request.key
                and confirmed.request.config_path == outcome.request.config_path
                and prior.cache_generation == result.previous_generation
                and prior.file_revision == result.previous_file_revision
            ):
                updated = replace(
                    confirmed,
                    result=replace(
                        prior,
                        phase="saved" if result.caches_reloaded else prior.phase,
                        caches_reloaded=result.caches_reloaded or prior.caches_reloaded,
                        cache_generation=result.cache_generation,
                        file_revision=result.file_revision,
                    ),
                )
                self._confirmed[key] = updated
                if self._states.get(key) == confirmed:
                    self._states[key] = updated

    def known_value(
        self,
        key: ConfigKey,
        config_path: Path,
        cached_value: str | bool,
        generation: int | None = None,
        file_revision: tuple[int, int, int, int] | None = None,
    ) -> str | bool:
        """Keep a committed value visible when its cache publication failed.

        Args:
            key: Configuration control to project.
            config_path: Active configuration file to match against the receipt.
            cached_value: Runtime value used when no matching receipt exists.
            generation: Current runtime cache generation.
            file_revision: Current file identity and modification fingerprint.

        Returns:
            The matching committed value awaiting cache refresh, or cached_value.
        """
        confirmed = self._confirmed.get(key)
        if (
            confirmed is not None
            and confirmed.request.config_path == config_path
            and confirmed.result.phase == "cache_refresh"
            and confirmed.result.cache_generation == generation
            and confirmed.result.file_revision == file_revision
            and confirmed.result.stored is not None
        ):
            return confirmed.result.stored
        return cached_value

    def known_root(
        self,
        config_path: Path,
        cached_root: str,
        generation: int | None = None,
        file_revision: tuple[int, int, int, int] | None = None,
    ) -> str:
        """Read the saved root without mixing it with a master-switch receipt.

        Args:
            config_path: Active configuration file to match against the receipt.
            cached_root: Runtime root used when no matching receipt exists.
            generation: Current runtime cache generation.
            file_revision: Current file identity and modification fingerprint.

        Returns:
            The confirmed root awaiting cache refresh, or cached_root.
        """
        return str(
            self.known_value(
                "workspace_root", config_path, cached_root, generation, file_revision
            )
        )

    def close_admission(self) -> None:
        """Fence future requests without cancelling admitted writes."""
        self._closed = True

    async def close_and_drain(self) -> None:
        """Drain admitted work independently of cancellation of this observer."""
        self.close_admission()
        if self._tasks:
            await asyncio.shield(asyncio.gather(*tuple(self._tasks)))


def get_mcp_local_config_saves(host: Any) -> MCPLocalConfigSaves:
    """Lazily obtain the session owner without retaining a screen.

    Args:
        host: Application object that owns the save coordinator and shutdown flag.

    Returns:
        The existing or newly attached session coordinator.

    Raises:
        ConfigSaveUnavailable: The host has closed save admission for shutdown.
    """
    if getattr(host, "_mcp_local_config_saves_closed", False):
        raise ConfigSaveUnavailable("shutdown")
    owner = getattr(host, "_mcp_local_config_saves", None)
    if owner is None:
        owner = MCPLocalConfigSaves()
        host._mcp_local_config_saves = owner
    return owner


def config_file_revision(path: Path) -> tuple[int, int, int, int] | None:
    """Track external replacements/edits that do not advance runtime generation.

    Args:
        path: Configuration file to inspect.

    Returns:
        Device, inode, modification nanoseconds and size, or None if stat fails.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size
