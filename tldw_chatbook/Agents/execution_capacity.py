"""App-owned accounting for physical agent operations (ADR-134).

This ledger holds metadata only. No provider calls, callbacks, or joins run
under its lock. Finishing a run leaves its owner alive until all operations
finish, including workers abandoned by a timeout and detached model cleanup.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Literal
from uuid import uuid4

from loguru import logger

from .agent_models import WorkOrigin as WorkOrigin


class CapacityRefused(RuntimeError):
    """A stable, body-free admission refusal reason."""


def _integer(value: object, default: int) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    return default


@dataclass(frozen=True)
class ExecutionSnapshot:
    execution_id: str
    conversation_id: str
    run_id: str | None
    origin: WorkOrigin
    child: bool
    root_finished: bool
    tool_workers: int
    stopping_tool_workers: int
    model_lifelines: int
    stopping_model_lifelines: int


@dataclass(frozen=True)
class CapacitySnapshot:
    closed: bool
    child_executions: int
    stopping_children: int
    tool_workers: int
    stopping_tool_workers: int
    model_lifelines: int
    stopping_model_lifelines: int
    executions: tuple[ExecutionSnapshot, ...]


class RuntimeCapacity:
    """One native Console runtime's capacity; never a process singleton."""

    def __init__(
        self,
        *,
        max_tool_workers: int = 8,
        reserved_manual_tool_workers: int = 2,
        max_child_executions: int = 6,
        reserved_manual_children: int = 2,
    ) -> None:
        self._lock = Lock()
        self._read_settings = False
        self.set_tool_limits(max_tool_workers, reserved_manual_tool_workers)
        self.set_child_limits(max_child_executions, reserved_manual_children)
        self._closed = False
        self._owners: dict[str, ExecutionOwner] = {}

    @classmethod
    def from_settings(cls) -> RuntimeCapacity:
        """Read current agent settings before each tool admission."""
        capacity = cls()
        capacity._read_settings = True
        capacity._refresh_limits()
        return capacity

    def set_tool_limits(self, total: int, reserved_manual: int) -> None:
        """Change future admission without cancelling already admitted work."""
        total = _integer(total, 8)
        total = total if total > 0 else 8
        reserve = max(0, min(_integer(reserved_manual, 2), total))
        with self._lock:
            self.max_tool_workers = total
            self.reserved_manual_tool_workers = reserve

    def _refresh_limits(self) -> None:
        if self._read_settings:
            from .run_log import _setting

            self.set_tool_limits(
                _setting("max_runtime_tool_workers", 8),
                _setting("reserved_manual_tool_workers", 2),
            )
            self.set_child_limits(
                _setting("max_runtime_subagents", 6),
                _setting("reserved_manual_subagents", 2),
            )

    def set_child_limits(self, total: int, reserved_manual: int) -> None:
        """Apply child limits to future launches while retaining occupied leases."""
        total = _integer(total, 6)
        total = total if total > 0 else 6
        reserve = max(0, min(_integer(reserved_manual, 2), total))
        with self._lock:
            self.max_child_executions = total
            self.reserved_manual_children = reserve

    def begin_execution(
        self, *, origin: WorkOrigin, conversation_id: str, child: bool = False
    ) -> ExecutionOwner:
        """Create an owner from trusted dispatch metadata before starting work."""
        if not isinstance(origin, WorkOrigin):
            raise TypeError("origin must be a WorkOrigin")
        self._refresh_limits()
        with self._lock:
            if self._closed:
                raise CapacityRefused("runtime_closed")
            if child:
                children = [owner for owner in self._owners.values() if owner.child]
                if len(children) >= self.max_child_executions:
                    raise CapacityRefused("child_capacity")
                automatic = sum(
                    owner.origin is WorkOrigin.AUTOMATIC for owner in children
                )
                if origin is WorkOrigin.AUTOMATIC and automatic >= (
                    self.max_child_executions - self.reserved_manual_children
                ):
                    raise CapacityRefused("automatic_child_capacity")
            owner = ExecutionOwner(self, origin, conversation_id, child)
            self._owners[owner.execution_id] = owner
            return owner

    def close(self) -> None:
        """Close admission before shutdown drains existing operations."""
        with self._lock:
            self._closed = True

    def snapshot(self) -> CapacitySnapshot:
        """Return detached metadata, including terminal owners still stopping."""
        with self._lock:
            entries = tuple(
                ExecutionSnapshot(
                    owner.execution_id,
                    owner.conversation_id,
                    owner._run_id,
                    owner.origin,
                    owner.child,
                    owner._root_finished,
                    sum(op.kind == "tool" for op in owner._operations),
                    sum(op.kind == "tool" and op._stopping for op in owner._operations),
                    sum(op.kind == "model" for op in owner._operations),
                    sum(
                        op.kind == "model" and op._stopping for op in owner._operations
                    ),
                )
                for owner in self._owners.values()
            )
            return CapacitySnapshot(
                self._closed,
                sum(entry.child for entry in entries),
                sum(
                    entry.child
                    and (
                        entry.root_finished
                        or entry.stopping_tool_workers > 0
                        or entry.stopping_model_lifelines > 0
                    )
                    for entry in entries
                ),
                sum(entry.tool_workers for entry in entries),
                sum(entry.stopping_tool_workers for entry in entries),
                sum(entry.model_lifelines for entry in entries),
                sum(entry.stopping_model_lifelines for entry in entries),
                entries,
            )


_current_owner: ContextVar[ExecutionOwner | None] = ContextVar(
    "agent_execution_owner", default=None
)


def current_execution_owner() -> ExecutionOwner | None:
    """Return this run thread's owner (explicitly bound again in children)."""
    return _current_owner.get()


class ExecutionOwner:
    """A run's root and physical operations, guarded by the runtime lock."""

    def __init__(
        self,
        capacity: RuntimeCapacity,
        origin: WorkOrigin,
        conversation_id: str,
        child: bool,
    ):
        self.capacity = capacity
        self.origin = origin
        self.child = child
        self.conversation_id = conversation_id
        self.execution_id = uuid4().hex
        self._run_id: str | None = None
        self._root_finished = False
        self._operations: set[OwnedOperation] = set()
        self._drain_callbacks: list[Callable[[bool], None]] = []
        self._cleanup_unproven = False
        self._drain_outcome: bool | None = None

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _current_owner.set(self)
        try:
            yield
        finally:
            _current_owner.reset(token)

    def bind_run(self, run_id: str) -> None:
        with self.capacity._lock:
            if self._run_id is not None and self._run_id != run_id:
                raise ValueError("execution already belongs to another run")
            self._run_id = run_id

    @property
    def run_id(self) -> str | None:
        """The created run, or None when execution has not reached persistence."""
        with self.capacity._lock:
            return self._run_id

    def reserve_tool(self) -> OwnedOperation:
        return self._reserve("tool")

    def reserve_model(self) -> OwnedOperation:
        return self._reserve("model")

    def _reserve(self, kind: Literal["tool", "model"]) -> OwnedOperation:
        capacity = self.capacity
        capacity._refresh_limits()
        with capacity._lock:
            if capacity._closed:
                raise CapacityRefused("runtime_closed")
            if self._root_finished:
                raise CapacityRefused("execution_finished")
            if kind == "tool":
                if any(op.kind == "tool" and op._stopping for op in self._operations):
                    raise CapacityRefused("previous_tool_still_running")
                tools = [
                    op
                    for owner in capacity._owners.values()
                    for op in owner._operations
                    if op.kind == "tool"
                ]
                if len(tools) >= capacity.max_tool_workers:
                    raise CapacityRefused("tool_capacity")
                automatic = sum(op.owner.origin is WorkOrigin.AUTOMATIC for op in tools)
                if self.origin is WorkOrigin.AUTOMATIC and automatic >= (
                    capacity.max_tool_workers - capacity.reserved_manual_tool_workers
                ):
                    raise CapacityRefused("automatic_tool_capacity")
            operation = OwnedOperation(self, kind)
            self._operations.add(operation)
            return operation

    def finish_root(self) -> None:
        with self.capacity._lock:
            self._root_finished = True
            callbacks, outcome = self._prune_if_finished()
        self._invoke_callbacks(callbacks, outcome)

    def on_drained(self, callback: Callable[[bool], None]) -> None:
        """Invoke callback once physical ownership drains, outside the lock."""
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self.capacity._lock:
            if self._drain_outcome is None:
                self._drain_callbacks.append(callback)
                return
            outcome = self._drain_outcome
        self._invoke_callbacks((callback,), outcome)

    def mark_cleanup_unproven(self) -> None:
        """Make the eventual drain outcome uncertain without releasing capacity."""
        with self.capacity._lock:
            if self._drain_outcome is None:
                self._cleanup_unproven = True

    @staticmethod
    def _invoke_callbacks(
        callbacks: tuple[Callable[[bool], None], ...], outcome: bool | None
    ) -> None:
        if outcome is None:
            return
        for callback in callbacks:
            try:
                callback(outcome)
            except Exception as exc:  # noqa: BLE001 - callbacks are isolated
                logger.warning(
                    "Execution drain callback failed (exception_type={})",
                    type(exc).__name__,
                )

    def _prune_if_finished(
        self,
    ) -> tuple[tuple[Callable[[bool], None], ...], bool | None]:
        if self._root_finished and not self._operations:
            self.capacity._owners.pop(self.execution_id, None)
            if self._drain_outcome is None:
                self._drain_outcome = not self._cleanup_unproven
                callbacks = tuple(self._drain_callbacks)
                self._drain_callbacks.clear()
                return callbacks, self._drain_outcome
        return (), None


class OwnedOperation:
    """Release only from the worker/driver finally or a failed start."""

    def __init__(self, owner: ExecutionOwner, kind: Literal["tool", "model"]):
        self.owner = owner
        self.kind = kind
        self._stopping = False

    def mark_stopping(self) -> None:
        with self.owner.capacity._lock:
            self._stopping = True

    def finish(self) -> None:
        with self.owner.capacity._lock:
            self.owner._operations.discard(self)
            callbacks, outcome = self.owner._prune_if_finished()
        self.owner._invoke_callbacks(callbacks, outcome)
