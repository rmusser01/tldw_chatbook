"""Finite custody for actual app-managed legacy hook events (ADR-285/148)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Condition, Lock
from typing import Any

from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

from .activation import _guard, execution
from .execution_capacity import CapacityRefused, WorkOrigin, current_execution_owner

_prompt_handoff = ContextVar("legacy_prompt_hook_handoff", default=None)
_target_operation = ContextVar("legacy_hook_target_operation", default=None)


def current_target() -> _TargetCustody | None:
    return _target_operation.get()


class _NativeLegacyHookSource:
    """Sticky original runtime attachment; construction performs no config I/O."""

    def __init__(self, runtime, engine, permissions):
        self.runtime, self.engine, self.permissions = runtime, engine, permissions
        self.app = runtime._app
        self.capacity = runtime._execution_capacity
        self.activity = runtime._worktree_activity
        self.callbacks = (
            engine._target_provider,
            engine._notification_targets,
            engine._launch_guard,
            engine._cwd_provider,
        )
        self._condition = Condition()
        self._admissions = 0
        self._events = set()

    @contextmanager
    def capture(self, session_id):
        with self._condition:
            self._admissions += 1
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
                self._condition.notify_all()

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
        if not self.current_memory() or _guard.captured_sources(None) != self.sources:
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
        with activity.prelaunch(footprint=WorktreeFootprint(opaque=True)) as claim:
            if claim is None:
                raise CapacityRefused("hook_activity_busy")
            owner = capacity.begin_execution(
                origin=self.origin, conversation_id=self.conversation_id
            )
            operations = []
            try:
                for _target in targets:
                    operations.append(owner.reserve_tool())
                self.check()
                claim.bind_owner(owner)
            except BaseException:
                for operation in operations:
                    operation.finish()
                owner.finish_root()
                raise
            event = _EventCustody(self, claim, owner, operations)
            with issuer._condition:
                issuer._events.add(event)
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
        self.host_returned = False
        self.failed_write = False
        self.retired = self.settling = False

    @property
    def returned(self):
        return self.host_returned and all(target.returned for target in self.targets)

    def host_done(self, *, required_write_ok=True):
        with self._lock:
            self.host_returned = True
            self.failed_write |= not required_write_ok
        self.reconcile()

    def reconcile(self):
        with self._lock:
            ready = [
                target
                for target in self.targets
                if target.returned
                and target.proven
                and self.host_returned
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

    def invoke(self, callback, *args):
        with self.event._lock:
            if self.returned:
                raise RuntimeError("hook target cancelled before start")
            self.started = True
        token = _target_operation.set(self)
        try:
            with (
                execution(sources=self.event.source.sources),
                self.event.owner.activate(),
            ):
                self.event.source.check()
                return callback(*args)
        finally:
            _target_operation.reset(token)
            with self.event._lock:
                self.returned = True
            self.event.reconcile()


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
