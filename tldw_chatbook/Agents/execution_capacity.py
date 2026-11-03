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
        self._tool_observers: set[Callable[[], None]] = set()
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
            previous = getattr(self, "max_tool_workers", total)
            previous_auto = previous - getattr(
                self, "reserved_manual_tool_workers", reserve
            )
            self.max_tool_workers = total
            self.reserved_manual_tool_workers = reserve
        if total > previous or total - reserve > previous_auto:
            self._notify_tool_available()

    def subscribe_tool_availability(
        self, callback: Callable[[], None]
    ) -> Callable[[], None]:
        """Observe actual tool releases/limit increases, outside the capacity lock."""
        with self._lock:
            self._tool_observers.add(callback)

        def unsubscribe() -> None:
            with self._lock:
                self._tool_observers.discard(callback)

        return unsubscribe

    def _notify_tool_available(self) -> None:
        with self._lock:
            observers = tuple(self._tool_observers)
        for callback in observers:
            try:
                callback()
            except Exception:  # noqa: BLE001 - observers cannot prevent retirement
                logger.warning("Tool availability observer failed")

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

    @property
    def is_admitted(self) -> bool:
        """Report native membership until physical retirement, granting no authority."""
        with self.capacity._lock:
            return self.capacity._owners.get(self.execution_id) is self

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


_OMITTED_CONSOLE_RETIREMENT = object()


class OwnedOperation:
    """Release only from the worker/driver finally or a failed start."""

    def __init__(self, owner: ExecutionOwner, kind: Literal["tool", "model"]):
        self.owner = owner
        self.kind = kind
        self._stopping = False

    def mark_stopping(self) -> None:
        with self.owner.capacity._lock:
            self._stopping = True

    def finish(self, *, _console_retirement=_OMITTED_CONSOLE_RETIREMENT) -> None:
        journal = journal_record = source = event = None
        if _console_retirement is not _OMITTED_CONSOLE_RETIREMENT:
            if type(_console_retirement) is not tuple or len(_console_retirement) != 2:
                raise RuntimeError("original Stop retirement unavailable")
            source, event = _console_retirement
            fence, code, defaults, kwdefaults, closure = _NATIVE_STOP_RECEIVER_FENCE
            if (
                _require_original_stop_receivers is not fence
                or fence.__code__ is not code
                or fence.__defaults__ is not defaults
                or fence.__kwdefaults__ is not kwdefaults
                or fence.__closure__ is not closure
            ):
                raise RuntimeError("original Stop receiver changed")
            fence(source, event, self)
            source.check_retirement(event)
            journal_record = getattr(self.owner, "_console_native_drain_journal", None)
            if journal_record is not None:
                journal = source.retirement_journal_callback(event, journal_record)
        with self.owner.capacity._lock:
            if _console_retirement is not _OMITTED_CONSOLE_RETIREMENT and (
                self.owner is not event.owner
                or self not in self.owner._operations
                or getattr(self.owner, "_console_native_drain_journal", None)
                is not journal_record
                or journal is not None
                and journal not in self.owner._drain_callbacks
            ):
                raise RuntimeError("original Stop operation changed")
            released = self.kind == "tool" and self in self.owner._operations
            self.owner._operations.discard(self)
            callbacks, outcome = self.owner._prune_if_finished()
        if (
            _console_retirement is _OMITTED_CONSOLE_RETIREMENT
            or journal is None
            or journal not in callbacks
        ):
            self.owner._invoke_callbacks(callbacks, outcome)
        else:
            for callback in callbacks:
                if callback is journal:
                    with source.retirement_journal(event):
                        self.owner._invoke_callbacks((callback,), outcome)
                else:
                    self.owner._invoke_callbacks((callback,), outcome)
        if released:
            self.owner.capacity._notify_tool_available()


def _require_original_stop_receivers(source, event, operation=None):
    """Raw-check the finite native retirement receivers before any invocation."""
    import inspect
    import sys

    from tldw_chatbook.Chat import console_agent_sources as sources
    from tldw_chatbook.Chat import console_first_send_retirement as receipt

    from . import legacy_hook_ownership as legacy
    from .hooks_v2 import native_commands

    def unchanged(record):
        function, code, defaults, kwdefaults, entries, closure, cells = record
        return (
            function.__code__ is code
            and function.__defaults__ is defaults
            and function.__kwdefaults__ is kwdefaults
            and function.__closure__ is closure
            and (
                kwdefaults is None
                or len(kwdefaults) == len(entries)
                and all(kwdefaults.get(key) is value for key, value in entries)
            )
            and all(cell.cell_contents is value for cell, value in cells)
        )

    if (
        type(source) is not sources._AgentSources
        or type(event) is not native_commands._CommandCustody
        or inspect.getattr_static(sources._AgentSources, "__getattribute__")
        is not object.__getattribute__
        or inspect.getattr_static(native_commands._CommandCustody, "__getattribute__")
        is not object.__getattribute__
    ):
        raise RuntimeError("original Stop receiver changed")
    for name, record, wrapped in sources._NATIVE_STOP_SOURCE_METHODS:
        function = record[0]
        if (
            inspect.getattr_static(source, name, None) is not function
            or inspect.getattr_static(sources._AgentSources, name, None) is not function
            or not unchanged(record)
            or wrapped is not None
            and (
                getattr(function, "__wrapped__", None) is not wrapped[0]
                or not unchanged(wrapped)
            )
        ):
            raise RuntimeError("original Stop source method changed")
    for name, record in receipt._NATIVE_STOP_RECEIPT_FUNCTIONS:
        if vars(receipt).get(name) is not record[0] or not unchanged(record):
            raise RuntimeError("original Stop receipt method changed")
    if (
        tuple(
            vars(receipt).get(name)
            for name in ("_Views", "_FirstSendSource", "_FirstSendReceipt")
        )
        != receipt._NATIVE_STOP_RECEIPT_TYPES
    ):
        raise RuntimeError("original Stop receipt type changed")
    event_methods = {
        name: (function, code, defaults, kwdefaults, closure)
        for name, function, code, defaults, kwdefaults, closure in legacy._NATIVE_STOP_EVENT_METHODS
    }
    for name, (function, code, defaults, kwdefaults, closure) in event_methods.items():
        if (
            inspect.getattr_static(legacy._EventCustody, name, None) is not function
            or function.__code__ is not code
            or function.__defaults__ is not defaults
            or function.__kwdefaults__ is not kwdefaults
            or function.__closure__ is not closure
            or name == "_reconcile_active"
            and inspect.getattr_static(event, name, None) is not function
        ):
            raise RuntimeError("original Stop event method changed")
    command, code, defaults, kwdefaults, closure = (
        native_commands._NATIVE_STOP_COMMAND_RECONCILE
    )
    if (
        inspect.getattr_static(event, "reconcile", None) is not command
        or inspect.getattr_static(native_commands._CommandCustody, "reconcile", None)
        is not command
        or command.__code__ is not code
        or command.__defaults__ is not defaults
        or command.__kwdefaults__ is not kwdefaults
        or command.__closure__ is not closure
    ):
        raise RuntimeError("original Stop event receiver changed")
    event_source = inspect.getattr_static(event, "source", None)
    if (
        type(event_source) is not native_commands._CommandEvent
        or inspect.getattr_static(native_commands._CommandEvent, "__getattribute__")
        is not object.__getattribute__
        or inspect.getattr_static(event_source, "agent_sources", None) is not source
    ):
        raise RuntimeError("original Stop event source changed")
    if operation is not None:
        finish, code, defaults, kwdefaults, entries, closure = (
            _NATIVE_STOP_OPERATION_FINISH
        )
        caller = sys._getframe(2)
        target = caller.f_locals.get("target")
        if (
            type(operation) is not OwnedOperation
            or inspect.getattr_static(OwnedOperation, "__getattribute__")
            is not object.__getattribute__
            or inspect.getattr_static(operation, "finish", None) is not finish
            or inspect.getattr_static(OwnedOperation, "finish", None) is not finish
            or finish.__code__ is not code
            or finish.__defaults__ is not defaults
            or finish.__kwdefaults__ is not kwdefaults
            or finish.__closure__ is not closure
            or tuple((kwdefaults or {}).items()) != entries
            or caller.f_code is not event_methods["_reconcile_active"][1]
            or caller.f_locals.get("self") is not event
            or caller.f_locals.get("retiring_source") is not source
            or type(target) is not legacy._TargetCustody
            or target.event is not event
            or target.operation is not operation
            or operation.owner is not event.owner
        ):
            raise RuntimeError("original Stop finish caller changed")


_NATIVE_STOP_RECEIVER_FENCE = (
    _require_original_stop_receivers,
    _require_original_stop_receivers.__code__,
    _require_original_stop_receivers.__defaults__,
    _require_original_stop_receivers.__kwdefaults__,
    _require_original_stop_receivers.__closure__,
)
_NATIVE_STOP_OPERATION_FINISH = (
    OwnedOperation.finish,
    OwnedOperation.finish.__code__,
    OwnedOperation.finish.__defaults__,
    OwnedOperation.finish.__kwdefaults__,
    tuple((OwnedOperation.finish.__kwdefaults__ or {}).items()),
    OwnedOperation.finish.__closure__,
)
