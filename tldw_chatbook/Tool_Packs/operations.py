"""Application-owned observation of the three admitted Tool Profile writes."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .activation import InstalledToolProfile, ToolPackActivationResult
from .contracts import ToolPackError
from .publication import ToolPackPublicationResult
from .removal import RemovedToolProfile, ToolProfileRemovalResult

Operation = Literal["import", "export", "remove"]
WriteResult = (
    ToolPackActivationResult | ToolPackPublicationResult | ToolProfileRemovalResult
)


class ToolProfileWriteUnavailable(RuntimeError):
    """The owner refused a duplicate write or admission during shutdown."""


@dataclass(frozen=True, slots=True)
class ToolProfileWriteState:
    """Bounded session receipt; no paths, reviews or exception bodies."""

    revision: int
    operation: Operation
    profile_id: str | None
    in_progress: bool
    result: WriteResult | None = None
    error_category: str | None = None


@dataclass(slots=True)
class _ActiveWrite:
    state: ToolProfileWriteState
    cancelled: threading.Event
    task: asyncio.Task[ToolProfileWriteState]


class ToolProfileOperations:
    """Keep admitted threads owned independently of disposable UI workers.

    All methods and state projection belong to the application's event loop.
    Only the supplied write and cancellation probe run off-thread.
    """

    def __init__(self) -> None:
        self._active: dict[Operation, _ActiveWrite] = {}
        self._closed = False
        self._revision = 0
        self.state: ToolProfileWriteState | None = None
        self.completion_revision = 0

    def pending(self, operation: Operation) -> ToolProfileWriteState | None:
        """Return the exact pending operation, including across screen visits.

        Args:
            operation: Import, export or removal operation to inspect.

        Returns:
            The admitted pending state, or None when that operation is inactive.
        """
        active = self._active.get(operation)
        return active.state if active is not None else None

    def start(
        self,
        operation: Operation,
        profile_id: str | None,
        write: Callable[[Callable[[], bool]], object],
    ) -> asyncio.Task[ToolProfileWriteState]:
        """Register one admitted write synchronously, before yielding.

        Callers observe the returned task through ``asyncio.shield``. Cancelling
        their wait must not cancel the admitted task or imply a stopped thread.

        Args:
            operation: Import, export or removal operation to admit.
            profile_id: Profile identity retained in the bounded receipt, if known.
            write: Synchronous writer receiving a cooperative cancellation probe.

        Returns:
            The application-owned task resolving to the final write receipt.

        Raises:
            ToolProfileWriteUnavailable: Shutdown or a duplicate operation blocks admission.
            ValueError: The operation name is unsupported.
            RuntimeError: No application event loop is running.
        """
        if self._closed:
            raise ToolProfileWriteUnavailable("shutdown")
        if operation in self._active:
            raise ToolProfileWriteUnavailable("busy")
        if operation not in {"import", "export", "remove"}:
            raise ValueError("unknown Tool Profile operation")
        self._revision += 1
        pending = ToolProfileWriteState(self._revision, operation, profile_id, True)
        cancelled = threading.Event()
        task = asyncio.create_task(
            self._run(operation, profile_id, write, cancelled),
            name=f"tool-profile-{operation}:{self._revision}",
        )
        self._active[operation] = _ActiveWrite(pending, cancelled, task)
        self.state = pending
        return task

    async def _run(
        self,
        operation: Operation,
        profile_id: str | None,
        write: Callable[[Callable[[], bool]], object],
        cancelled: threading.Event,
    ) -> ToolProfileWriteState:
        fallback = {
            "import": "activation_uncertain",
            "export": "durability_uncertain",
            "remove": "outcome_uncertain",
        }[operation]
        result = None
        error_category = None
        try:
            candidate = await asyncio.to_thread(write, cancelled.is_set)
            if self._valid_result(operation, profile_id, candidate):
                result = candidate
            else:
                error_category = fallback
        except ToolPackError as error:
            error_category = error.category
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise
            # A synchronous callable can itself raise CancelledError. That
            # says nothing about whether its mutation committed.
            error_category = fallback
        except Exception:  # noqa: BLE001 - retain only a stable service failure category
            error_category = fallback
        self._revision += 1
        outcome = ToolProfileWriteState(
            self._revision, operation, profile_id, False, result, error_category
        )
        active = self._active.get(operation)
        if active is not None and active.task is asyncio.current_task():
            self._active.pop(operation)
        self.state = outcome
        self.completion_revision = outcome.revision
        return outcome

    @staticmethod
    def _valid_result(
        operation: Operation, profile_id: str | None, result: object
    ) -> bool:
        """Keep malformed service results out of the timer-driven projection."""
        if operation == "import":
            return (
                type(result) is ToolPackActivationResult
                and type(result.installed) is InstalledToolProfile
                and type(result.installed.profile_id) is str
                and result.installed.profile_id == profile_id
                and type(result.installed.revision) is int
            )
        if operation == "remove":
            return (
                type(result) is ToolProfileRemovalResult
                and type(result.tombstone) is RemovedToolProfile
                and type(result.tombstone.profile_id) is str
                and result.tombstone.profile_id == profile_id
            )
        return (
            type(result) is ToolPackPublicationResult
            and type(result.archive_sha256) is str
            and type(result.committed) is bool
            and type(result.durability_uncertain) is bool
        )

    def close_admission(self) -> None:
        """Close new writes and signal supported pre-publication cancellation."""
        self._closed = True
        for active in self._active.values():
            active.cancelled.set()

    async def close_and_drain(self) -> None:
        """Settle admitted work without propagating caller cancellation to it."""
        self.close_admission()
        tasks = tuple(active.task for active in self._active.values())
        if tasks:
            await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
