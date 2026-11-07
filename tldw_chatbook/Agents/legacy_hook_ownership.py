"""Finite custody for actual app-managed legacy hook events (ADR-285/148)."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from inspect import getattr_static as _cleanup_getattr_static
from threading import Condition, Lock, RLock
from types import FunctionType
from typing import Any

from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

from .activation import _guard, execution
from .execution_capacity import (
    CapacityRefused,
    ExecutionOwner,
    OwnedOperation,
    WorkOrigin,
    current_execution_owner,
)

_prompt_handoff = ContextVar("legacy_prompt_hook_handoff", default=None)
_target_operation = ContextVar("legacy_hook_target_operation", default=None)
_CONDITION_LOCK_TYPE = type(RLock())
_EVENT_LOCK_TYPE = type(Lock())


def current_target() -> _TargetCustody | None:
    return _target_operation.get()


def _session_fenced(issuer, session_id):
    runtime = issuer.runtime
    controller = runtime._chat_controller
    return bool(
        session_id in runtime._admission_fenced_sessions
        or controller is not None
        and session_id in controller._session_close_generations
    )


class _EventCustodyRegistry:
    """Shared finite event drain; concrete issuers own their separate sources."""

    def __init__(self):
        self._condition = Condition()
        self._admissions = 0
        self._events = set()

    def returned(self, event):
        with self._condition:
            if event.retired:
                self._events.discard(event)
            self._condition.notify_all()

    def wait_closed(self):
        """Join actual callbacks/handoffs, retaining any unproven physical custody."""
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    not self._admissions
                    and all(
                        event.returned
                        and all(
                            target.released or not target.proven
                            for target in event.targets
                        )
                        and (
                            event.failed_write
                            or any(not target.proven for target in event.targets)
                        )
                        for event in self._events
                    )
                )
            )
            return not self._events

    @property
    def cleanup_pending(self):
        with self._condition:
            return bool(self._admissions or self._events)


class _NativeLegacyHookSource(_EventCustodyRegistry):
    """Sticky original runtime attachment; construction performs no config I/O."""

    def __init__(self, runtime, engine, permissions):
        self.runtime, self.engine, self.permissions = runtime, engine, permissions
        self._workflow_check_lock = RLock()
        self.app = runtime._app
        self.capacity = runtime._execution_capacity
        self.activity = runtime._worktree_activity
        self.callbacks = (
            engine._target_provider,
            engine._notification_targets,
            engine._launch_guard,
            engine._cwd_provider,
        )
        super().__init__()
        self._session_admissions = {}
        self._close_observation = (
            self._condition,
            self._events,
            self._session_admissions,
        )

    @contextmanager
    def capture(self, session_id):
        with self._condition:
            if _session_fenced(self, session_id):
                raise RuntimeError("native hook session is closing")
            self._admissions += 1
            self._session_admissions[session_id] = (
                self._session_admissions.get(session_id, 0) + 1
            )
        try:
            # Pin resources created since attachment before any external read.
            if self.capacity is None:
                self.capacity = self.runtime._execution_capacity
            if self.activity is None:
                self.activity = self.runtime._worktree_activity
            source = _EventSource(self, session_id)
            source.check()
            with execution(sources=source.sources):
                yield source
        finally:
            with self._condition:
                self._admissions -= 1
                remaining = self._session_admissions[session_id] - 1
                if remaining:
                    self._session_admissions[session_id] = remaining
                else:
                    del self._session_admissions[session_id]
                self._condition.notify_all()

    def selected_ready(self, session_id):
        """Observe one session's original capture and event retirement."""
        condition = _cleanup_getattr_static(self, "_condition")
        if type(condition) is not Condition:
            return False
        lock = vars(condition).get("_lock")
        if type(lock) is not _CONDITION_LOCK_TYPE:
            return False
        if not lock.acquire(blocking=False):
            return None
        try:
            admissions = _cleanup_getattr_static(self, "_session_admissions")
            events = _cleanup_getattr_static(self, "_events")
            if (
                type(admissions) is not dict
                or type(events) is not set
                or type(session_id) is not str
            ):
                return False
            count = admissions.get(session_id)
            if count is not None:
                if type(count) is not int or count <= 0:
                    return False
                return None
            for event in events:
                if type(event) is not _EventCustody:
                    return False
                source = _cleanup_getattr_static(event, "source")
                if type(source) is not _EventSource:
                    return False
                selected = _cleanup_getattr_static(source, "session_id")
                if type(selected) is not str:
                    return False
                if selected != session_id:
                    continue
                event_lock = _cleanup_getattr_static(event, "_lock")
                if type(
                    event_lock
                ) is not _EVENT_LOCK_TYPE or event_lock is not _cleanup_getattr_static(
                    event, "_original_lock"
                ):
                    return False
                if not event_lock.acquire(blocking=False):
                    return None
                try:
                    targets = _cleanup_getattr_static(event, "targets")
                    host_returned = _cleanup_getattr_static(event, "host_returned")
                    failed_write = _cleanup_getattr_static(event, "failed_write")
                    if (
                        type(targets) is not tuple
                        or type(host_returned) is not bool
                        or type(failed_write) is not bool
                    ):
                        return False
                    states = []
                    for target in targets:
                        if type(target) is not _TargetCustody:
                            return False
                        returned = _cleanup_getattr_static(target, "returned")
                        proven = _cleanup_getattr_static(target, "proven")
                        if type(returned) is not bool or type(proven) is not bool:
                            return False
                        states.append((returned, proven))
                    if (
                        host_returned
                        and all(returned for returned, _ in states)
                        and (failed_write or any(not proven for _, proven in states))
                    ):
                        return False
                    return None
                finally:
                    event_lock.release()
            return True
        finally:
            lock.release()


class _EventSource:
    def __init__(self, issuer, session_id):
        self.issuer = issuer
        runtime = issuer.runtime
        self.store, self.controller = runtime._chat_store, runtime._chat_controller
        self.session_id = session_id
        self.session = self.store._sessions.get(session_id) if self.store else None
        if session_id and self.session is None:
            raise RuntimeError("native hook session unavailable")
        self.session_identity = self._session_identity()
        self.persistence = getattr(self.store, "persistence", None)
        self.database = getattr(self.persistence, "db", None)
        self.targets = ()
        self.sources = _guard.captured_sources(None)
        from tldw_chatbook.Chat.console_agent_sources import current_agent_sources

        self.agent_sources = current_agent_sources(self.controller)
        from tldw_chatbook.Chat.console_ordinary_tree import current_ordinary_tree

        self.ordinary_tree = current_ordinary_tree(self.controller)
        parent = current_execution_owner()
        handoff = _prompt_handoff.get()
        self.handoff = (
            handoff if handoff and handoff.controller is self.controller else None
        )
        self.origin = (
            WorkOrigin.MANUAL
            if self.handoff is not None
            else parent.origin
            if parent is not None
            and parent.capacity is runtime._execution_capacity
            and parent.is_admitted
            else WorkOrigin.AUTOMATIC
        )
        self.conversation_id = (
            self.session.persisted_conversation_id or self.session.id
            if self.session is not None
            else ""
        )

    def _session_identity(self):
        return tuple(
            getattr(self.session, key, None)
            for key in (
                "incarnation_id",
                "conversation_binding_revision",
                "ephemeral",
                "workspace_id",
            )
        )

    def current_memory(self, *, handoff=False):
        """Only original references and local state; safe inside publication lock."""
        issuer, runtime = self.issuer, self.issuer.runtime
        engine = issuer.engine
        return bool(
            engine._native_source is issuer
            and runtime._run_hooks_engine is engine
            and runtime._hook_permissions is issuer.permissions
            and runtime._app is issuer.app
            and runtime._chat_store is self.store
            and runtime._chat_controller is self.controller
            and (
                self.store is None
                or self.store._sessions.get(self.session_id) is self.session
            )
            and (
                self._session_identity() == self.session_identity
                or handoff
                and self._session_identity()[0] == self.session_identity[0]
                and self._session_identity()[2:] == self.session_identity[2:]
            )
            and getattr(self.store, "persistence", None) is self.persistence
            and getattr(self.persistence, "db", None) is self.database
            and (self.controller is None or self.controller.store is self.store)
            and not runtime._disposed
            and not engine._closed.is_set()
            and not issuer.permissions._closed.is_set()
            and all(
                target in issuer.permissions._published_targets
                for target in self.targets
            )
            and (
                issuer.capacity is None
                or runtime._execution_capacity is issuer.capacity
            )
            and (
                issuer.activity is None or runtime._worktree_activity is issuer.activity
            )
            and issuer.callbacks
            == (
                engine._target_provider,
                engine._notification_targets,
                engine._launch_guard,
                engine._cwd_provider,
            )
            and (self.handoff is None or not self.handoff.cancelled)
        )

    def check(self):
        if (
            _session_fenced(self.issuer, self.session_id)
            or not self.current_memory()
            or _guard.captured_sources(None) != self.sources
        ):
            raise RuntimeError("native hook source changed")

    def admit(self, targets):
        self.targets = targets
        self.check()
        issuer = self.issuer
        capacity = issuer.capacity
        if capacity is None:
            capacity = issuer.runtime.execution_capacity
            issuer.capacity = capacity
        activity = issuer.activity
        if activity is None:
            activity = issuer.runtime.worktree_activity
            issuer.activity = activity
        self.check()
        if self.ordinary_tree is not None:
            owner = self.ordinary_tree.borrow(issuer.runtime)
            operations = []
            try:
                for _target in targets:
                    operations.append(owner.reserve_tool())
                self.check()
                event = _EventCustody(self, None, owner, operations)
                with issuer._condition:
                    fenced = _session_fenced(issuer, self.session_id)
                    if not fenced:
                        issuer._events.add(event)
                if fenced:
                    raise RuntimeError("native hook session is closing")
            except BaseException:
                for operation in operations:
                    operation.finish()
                raise
            if self.handoff is not None:
                self.handoff.attach(event)
            return event
        with activity.prelaunch(footprint=WorktreeFootprint(opaque=True)) as claim:
            if claim is None:
                raise CapacityRefused("hook_activity_busy")
            owner = capacity.begin_execution(
                origin=self.origin, conversation_id=self.conversation_id
            )
            operations = []
            bound = False
            try:
                for _target in targets:
                    operations.append(owner.reserve_tool())
                self.check()
                claim.bind_owner(owner)
                bound = True
                event = _EventCustody(self, claim, owner, operations)
                with issuer._condition:
                    fenced = _session_fenced(issuer, self.session_id)
                    if not fenced:
                        issuer._events.add(event)
                if fenced:
                    raise RuntimeError("native hook session is closing")
            except BaseException:
                for operation in operations:
                    operation.finish()
                if bound:
                    claim.settle(durable=True)
                owner.finish_root()
                raise
            if self.handoff is not None:
                self.handoff.attach(event)
            return event


class _EventCustody:
    # Ordinary legacy events have no required journal payload. The real prompt
    # host separately reports its applicable native write before its handoff.
    required_writes = ()

    def __init__(self, source, claim, owner, operations):
        self.source, self.claim, self.owner = source, claim, owner
        self.targets = tuple(_TargetCustody(self, op) for op in operations)
        self._lock = Lock()
        self._original_lock = self._lock
        self.host_returned = False
        self.failed_write = False
        self.retired = self.settling = False
        self._reconciling = self._reconcile_pending = False

    @property
    def returned(self):
        return self.host_returned and all(target.returned for target in self.targets)

    def host_done(self, *, required_write_ok=True):
        with self._lock:
            self.host_returned = True
            self.failed_write |= not required_write_ok
        self.reconcile()

    def reconcile(self):
        from contextlib import nullcontext

        with self._lock:
            if self.retired:
                return
            if self._reconciling:
                self._reconcile_pending = True
                return
            self._reconciling = True
        try:
            source = getattr(self.source, "agent_sources", None)
            with source.activate() if source is not None else nullcontext():
                while True:
                    self._reconcile_active()
                    with self._lock:
                        if self.retired or not self._reconcile_pending:
                            self._reconciling = False
                            return
                        self._reconcile_pending = False
        except BaseException:
            with self._lock:
                self._reconciling = False
                self._reconcile_pending = False
            raise

    def _reconcile_active(self):
        with self._lock:
            ready = [
                target
                for target in self.targets
                if target.returned
                and target.proven
                and self.host_returned
                and not self.failed_write
                and not target.releasing
            ]
            for target in ready:
                target.releasing = True
        for target in ready:
            target.operation.finish()
            with self._lock:
                target.released = True
        with self._lock:
            settle = self.returned and all(target.released for target in self.targets)
            settle = settle and not self.failed_write and not self.settling
            if settle:
                self.settling = True
        if settle:
            if self.claim is not None:
                self.claim.settle(durable=True)
                self.owner.finish_root()
            with self._lock:
                self.retired = True
        self.source.issuer.returned(self)

    def cancel_queued_if_requested(self):
        handoff = self.source.handoff
        if handoff is not None and handoff.cancelled:
            for target in self.targets:
                if target.future is not None and target.future.cancel():
                    target.cancel_before_start()

    def abandon_unstarted(self):
        for target in self.targets:
            target.cancel_before_start()
        self.host_done()


class _TargetCustody:
    def __init__(self, event, operation):
        self.event, self.operation = event, operation
        self.started = self.returned = self.releasing = self.released = False
        self.future = None
        self._callbacks = 0
        self.proven = True  # No process attempted; launch invalidates this first.

    def cancel_before_start(self):
        with self.event._lock:
            if not self.started:
                self.returned = True
        self.event.reconcile()

    def process_attempted(self):
        self.proven = False

    def process_terminal(self, proven):
        self.proven = proven
        if not proven:
            self.operation.mark_stopping()
            self.event.owner.mark_cleanup_unproven()

    def retain_callback(self):
        with self.event._lock:
            self._callbacks += 1
            self.returned = False

    def callback_returned(self):
        with self.event._lock:
            self._callbacks -= 1
            self.returned = not self._callbacks
        self.event.reconcile()

    @contextmanager
    def activate(self):
        with self.event._lock:
            if self.returned:
                raise RuntimeError("hook target cancelled before start")
            self.started = True
            self._callbacks += 1
        token = _target_operation.set(self)
        try:
            with (
                execution(sources=self.event.source.sources),
                self.event.owner.activate(),
            ):
                self.event.source.check()
                yield
        finally:
            _target_operation.reset(token)
            self.callback_returned()

    def invoke(self, callback, *args):
        with self.activate():
            return callback(*args)


class _PromptHandoff:
    def __init__(self, controller):
        self.controller = controller
        import asyncio

        self.loop = asyncio.get_running_loop()
        self.cancelled = self.closed = self.failed_write = self.deferred = False
        self.event = None
        self._lock = Lock()

    def attach(self, event):
        with self._lock:
            self.event = event
            closed, failed = self.closed, self.failed_write
        if closed:
            event.host_done(required_write_ok=not failed)

    def check(self):
        if self.event is not None and not self.event.source.current_memory(
            handoff=True
        ):
            raise RuntimeError("original hook outcome source changed")

    def discard(self):
        # Retirement is invoked under the existing store lock. Publish native
        # retirement on the original host loop only after that lock returns.
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.complete)

    def complete(self, *, required_write_ok=True):
        with self._lock:
            self.closed = True
            self.failed_write |= not required_write_ok
            event, failed = self.event, self.failed_write
        if event is not None:
            event.cancel_queued_if_requested()
            event.host_done(required_write_ok=not failed)


@contextmanager
def prompt_handoff(controller: Any) -> Iterator[None]:
    """The original submit coroutine owns outcome consumption and all discards."""
    handoff = _PromptHandoff(controller)
    token = _prompt_handoff.set(handoff)
    try:
        yield
    except BaseException:
        handoff.cancelled = True
        raise
    finally:
        if not handoff.deferred:
            handoff.complete()
        _prompt_handoff.reset(token)


def prompt_handed_off(*, required_write_ok: bool = True) -> None:
    handoff = _prompt_handoff.get()
    if handoff is not None:
        handoff.complete(required_write_ok=required_write_ok)


def pending_prompt_handoff() -> _PromptHandoff | None:
    handoff = _prompt_handoff.get()
    if handoff is not None and handoff.event is not None and not handoff.closed:
        return handoff
    return None


def check_prompt_handoff() -> None:
    handoff = _prompt_handoff.get()
    if handoff is not None:
        handoff.check()


class _CleanupSource:
    """Prelaunch originals used only by the already-owned process finalizer."""

    def __init__(self, target, selected):
        read = _cleanup_getattr_static
        self.target, self.selected = target, selected
        self.event = read(target, "event")
        self.operation = read(target, "operation")
        self.owner = read(self.event, "owner")
        self.operation_type, self.owner_type = OwnedOperation, ExecutionOwner
        self.definitions = _CLEANUP_DEFINITIONS
        self.source = read(self.event, "source")
        self.issuer = read(self.source, "issuer")
        self.runtime = read(self.issuer, "runtime")
        self.engine = read(self.issuer, "engine")
        self.permissions = read(self.issuer, "permissions")
        self.app = read(self.issuer, "app")
        self.store, self.controller = (
            read(self.source, "store"),
            read(self.source, "controller"),
        )
        self.session, self.session_id = (
            read(self.source, "session"),
            read(self.source, "session_id"),
        )
        self.session_identity = read(self.source, "session_identity")
        self.persistence, self.database = (
            read(self.source, "persistence"),
            read(self.source, "database"),
        )
        self.capacity, self.activity = (
            read(self.issuer, "capacity"),
            read(self.issuer, "activity"),
        )
        self.callbacks, self.sources = (
            read(self.issuer, "callbacks"),
            read(self.source, "sources"),
        )
        self.targets = read(self.source, "targets")

    def current(self, process):
        read = _cleanup_getattr_static
        target, event, source = self.target, self.event, self.source
        issuer, runtime, engine = self.issuer, self.runtime, self.engine
        registry = (
            read(self.store, "_sessions", None) if self.store is not None else None
        )
        identity = tuple(
            read(self.session, key, None)
            for key in (
                "incarnation_id",
                "conversation_binding_revision",
                "ephemeral",
                "workspace_id",
            )
        )
        if not (
            read(target, "event") is event
            and read(target, "operation") is self.operation
            and read(self.operation, "owner") is self.owner
            and read(target, "_cleanup_process", None) is process
            and read(event, "source") is source
            and read(event, "owner") is self.owner
            and any(item is target for item in read(event, "targets"))
            and read(source, "issuer") is issuer
            and read(source, "targets") is self.targets
            and any(item is self.selected for item in self.targets)
            and read(source, "sources") is self.sources
            and read(issuer, "runtime") is runtime
            and read(issuer, "engine") is engine
            and read(issuer, "permissions") is self.permissions
            and read(issuer, "app") is self.app
            and read(engine, "_native_source") is issuer
            and read(engine, "_native_attached", None) is True
            and read(runtime, "_legacy_hooks_source", None) is issuer
            and read(runtime, "_run_hooks_engine") is engine
            and read(runtime, "_hook_permissions") is self.permissions
            and read(runtime, "_app") is self.app
            and read(runtime, "_chat_store") is self.store
            and read(runtime, "_chat_controller") is self.controller
            and read(source, "store") is self.store
            and read(source, "controller") is self.controller
            and read(source, "session") is self.session
            and read(source, "session_id") is self.session_id
            and read(source, "session_identity") is self.session_identity
            and (
                self.store is None
                or type(registry) is dict
                and registry.get(self.session_id) is self.session
            )
            and all(
                type(value) in (str, int, bool, type(None))
                and type(value) is type(original)
                and value == original
                for value, original in zip(identity, self.session_identity, strict=True)
            )
            and read(source, "persistence") is self.persistence
            and read(source, "database") is self.database
            and (
                self.store is None
                or read(self.store, "persistence", None) is self.persistence
            )
            and (
                self.persistence is None
                or read(self.persistence, "db", None) is self.database
            )
            and (
                self.controller is None or read(self.controller, "store") is self.store
            )
            and read(issuer, "capacity") is self.capacity
            and read(issuer, "activity") is self.activity
            and (
                self.capacity is None
                or read(runtime, "_execution_capacity") is self.capacity
            )
            and (
                self.activity is None
                or read(runtime, "_worktree_activity") is self.activity
            )
            and read(issuer, "callbacks") is self.callbacks
            and all(
                original is current
                for original, current in zip(
                    self.callbacks,
                    (
                        read(engine, "_target_provider"),
                        read(engine, "_notification_targets"),
                        read(engine, "_launch_guard"),
                        read(engine, "_cwd_provider"),
                    ),
                    strict=True,
                )
            )
        ):
            return False
        return _guard.captured_sources(None) == self.sources

    def terminal(self, proven):
        """Retain the original operation if cleanup cannot be proven."""
        read = object.__getattribute__
        target, operation, owner = (
            read(self, "target"),
            read(self, "operation"),
            read(self, "owner"),
        )
        operation_type, owner_type = (
            read(self, "operation_type"),
            read(self, "owner_type"),
        )
        definitions, capacity = read(self, "definitions"), read(self, "capacity")
        object.__setattr__(target, "proven", proven)
        if proven:
            return
        for cls, name, receiver in (
            (operation_type, "mark_stopping", operation),
            (owner_type, "mark_cleanup_unproven", owner),
        ):
            record = next(
                record
                for recorded, key, record in definitions
                if recorded is cls and key == name
            )
            function = record[0]
            if (
                type(receiver) is cls
                and vars(cls).get(name) is function
                and function.__code__ is record[1]
                and function.__defaults__ is record[2]
                and function.__kwdefaults__ is record[3]
                and function.__closure__ is record[5]
                and (
                    cls is not operation_type
                    or object.__getattribute__(receiver, "__dict__").get("owner")
                    is owner
                )
                and (
                    cls is not owner_type
                    or object.__getattribute__(receiver, "__dict__").get("capacity")
                    is capacity
                )
            ):
                function(receiver)


def _cleanup_record(function):
    if type(function) is not FunctionType:
        raise RuntimeError("native hook cleanup definition is not original")
    return (
        function,
        function.__code__,
        function.__defaults__,
        function.__kwdefaults__,
        tuple((function.__kwdefaults__ or {}).items()),
        function.__closure__,
        tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
    )


import inspect as _cleanup_inspect

from tldw_chatbook.Backup_Recovery import admission_runtime as _cleanup_admission

from . import activation as _cleanup_activation

_CLEANUP_DEFINITIONS = tuple(
    (owner, name, _cleanup_record(vars(owner)[name]))
    for owner, name in (
        (_CleanupSource, "__init__"),
        (_CleanupSource, "current"),
        (_CleanupSource, "terminal"),
        (_EventSource, "check"),
        (_EventSource, "current_memory"),
        (_EventSource, "admit"),
        (_NativeLegacyHookSource, "capture"),
        (_NativeLegacyHookSource, "selected_ready"),
        (_NativeLegacyHookSource, "__init__"),
        (_EventCustody, "__init__"),
        (_EventCustody, "reconcile"),
        (_EventCustody, "_reconcile_active"),
        (_EventCustodyRegistry, "returned"),
        (vars(_NativeLegacyHookSource)["capture"], "__wrapped__"),
        (sys.modules[__name__], "_session_fenced"),
        (_EventSource, "_session_identity"),
        (_TargetCustody, "process_attempted"),
        (_TargetCustody, "process_terminal"),
        (OwnedOperation, "mark_stopping"),
        (ExecutionOwner, "mark_cleanup_unproven"),
        (type(_guard), "captured_sources"),
        (_cleanup_admission, "execution_identity"),
        (_cleanup_activation, "_sources"),
        (_cleanup_inspect, "getattr_static"),
    )
)
_CLEANUP_ORIGINALS = (
    _CleanupSource,
    _TargetCustody,
    _EventCustody,
    _EventSource,
    _NativeLegacyHookSource,
    _guard,
    _CLEANUP_DEFINITIONS,
    _cleanup_getattr_static,
    _EventCustodyRegistry,
    Condition,
    _CONDITION_LOCK_TYPE,
    _EVENT_LOCK_TYPE,
)
