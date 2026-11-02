"""Original app source and finite custody for complete native hook events."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from threading import Lock
from types import SimpleNamespace

from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

from ..activation import _guard, execution
from ..execution_capacity import CapacityRefused, WorkOrigin, current_execution_owner
from ..legacy_hook_ownership import _EventCustody, _EventCustodyRegistry
from .causality import current_chain
from .matching import UnsupportedEventField, matches_handler


class NativeCommandEvents(_EventCustodyRegistry):
    """Sticky metadata attachment; never constructs a Chat or loads capacity."""

    def __init__(self, runtime, engine, session_id):
        super().__init__()
        self.runtime, self.engine, self.session_id = runtime, engine, session_id
        self.app = runtime._app
        self.capacity = runtime._execution_capacity
        self.activity = runtime._worktree_activity

    def capture(self, event, *, teardown=False):
        if self.capacity is None:
            self.capacity = self.runtime._execution_capacity
        if self.activity is None:
            self.activity = self.runtime._worktree_activity
        from .engine import _current_delivery

        delivery = _current_delivery.get()
        if delivery is not None and delivery[0] is self.engine:
            _, parent, state, original, projected = delivery
            if state.native is not None:
                job = parent.job
                with self.engine._lock:
                    if (
                        parent not in self.engine._deliveries
                        or parent.cancel.is_set()
                        or state.closed
                        or state.issuer is not self.engine._scope_issuer
                        or state.native.issuer is not self
                        or state.native.event is not original
                        or getattr(job, "event", None) is not projected
                        or getattr(job, "execution", None) is not state
                        or getattr(job, "native_plan", None) is None
                        or not current_chain().visits
                        or current_chain().visits[-1][:2]
                        != (projected.event_id, job.handler.id)
                    ):
                        raise PermissionError(
                            "Original nested event source unavailable."
                        )
                    state.native.root._check_registration()
                    previous = job.native_plan.issued.get(event.event)
                    if previous is not None and previous.event is event:
                        return previous
                    return _NestedEvent(state.native.root, job, event)
        return _CommandEvent(self, event, teardown=teardown)


class _CommandCustody(_EventCustody):
    # This event adds no durable payload. Existing plugin settlement is required.
    required_writes = ()

    def reconcile(self):
        loop = self.source.issuer.engine.loop
        future = loop.run_in_executor(None, super().reconcile)
        future.add_done_callback(lambda done: done.exception())


class _CommandEvent:
    def __init__(self, issuer, event, *, teardown):
        self.issuer, self.event, self.teardown = issuer, event, teardown
        runtime, engine = issuer.runtime, issuer.engine
        self.store, self.controller = runtime._chat_store, runtime._chat_controller
        self.session = (
            self.store._sessions.get(issuer.session_id) if self.store else None
        )
        self.session_identity = self._session_identity()
        self.persistence = getattr(self.store, "persistence", None)
        self.database = getattr(self.persistence, "db", None)
        self.permissions = runtime._hook_permissions
        self.review = getattr(self.permissions, "_published", None)
        self.signature = runtime._hooks_v2_configured.get(issuer.session_id)
        self.lifecycle = getattr(engine, "lifecycle_owner", None)
        self.definitions = engine.definitions
        self.callbacks = self._callbacks()
        self.native = getattr(engine, "native_plugins", None)
        self.snapshots = getattr(self.native, "snapshots", ())
        self.service = getattr(self.native, "service", None)
        self.native_owners = getattr(self.native, "owners", None)
        self.owner_entries = (
            tuple(self.native_owners.items()) if self.native_owners is not None else ()
        )
        self.service_state = tuple(
            getattr(self.service, key, None)
            for key in ("_thread", "_loop", "_coordinator", "_admission", "fences")
        )
        self.sources = _guard.captured_sources(None)
        self.maximum = tuple(h for h in self.definitions if h.event == event.event)
        # Match fields cannot be rewritten by input transformations. Retain every
        # potential phase, including an unsupported matcher, before execution.
        self.match_values = tuple(
            (key, event.data.get(key))
            for key in sorted({key for h in self.maximum for key in h.match or ()})
        )
        matched = []
        for handler in self.maximum:
            try:
                if matches_handler(handler, event):
                    matched.append(handler)
            except UnsupportedEventField:
                pass
        self._coverage = (True, tuple(matched))
        self.root = self
        self.plans = {}
        self.all_plans = ()
        self.targets = {}
        self.custody = None
        self.closed = False
        self.hosts = 0
        self.admitted = False
        self.ready = False
        self.admission_failed = False
        self._lock = Lock()
        self.handoff = None
        self.origin = WorkOrigin.AUTOMATIC
        parent = current_execution_owner()
        if (
            parent is not None
            and parent.capacity is issuer.capacity
            and parent.is_admitted
        ):
            self.origin = parent.origin
        from tldw_chatbook.Chat.console_chat_models import ConsoleSubmissionOrigin
        from tldw_chatbook.Chat.console_plugin_hook_preparation import _submission

        bound = _submission.get()
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if bound is not None and (
            bound[0] is task
            and bound[1] is self.controller
            and bound[2] is self.session
            and type(bound[4]) is ConsoleSubmissionOrigin
        ):
            self.origin = (
                WorkOrigin.AUTOMATIC if bound[4].automatic else WorkOrigin.MANUAL
            )

    @property
    def covered(self):
        return self._coverage[0]

    @property
    def selected(self):
        return self._coverage[1]

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

    def _callbacks(self):
        engine = self.issuer.engine
        return (
            engine.authority_check,
            engine._effect_authority_check,
            engine._event_projector,
            engine._dependency_required,
            engine._environment,
            engine._host_environment,
            engine.processes,
            engine.processes.owner,
            engine.processes._launch_guard,
            engine.mcp_executor,
        )

    def check_memory(self, event=None):
        self._check_sources(event)
        for plan in self.all_plans:
            if plan.permission is not None:
                plan.permission.check_memory()

    def _check_sources(self, event=None):
        issuer, runtime = self.issuer, self.issuer.runtime
        engine = issuer.engine
        event = event or self.event
        terminal_scope = getattr(self.lifecycle, "_session_end_execution", None)
        terminal = bool(
            self.teardown
            and terminal_scope is not None
            and terminal_scope._state.native is self
            and getattr(self.lifecycle, "_session_end_event", None) is self.event
        )
        if (
            engine.definitions is not self.definitions
            or tuple((key, event.data.get(key)) for key, _ in self.match_values)
            != self.match_values
        ):
            raise RuntimeError("Command event maximum changed.")
        if not self.covered:
            return
        if (
            self.store is None
            or self.controller is None
            or self.session is None
            or engine._native_commands is not issuer
            or runtime._hooks_v2_engines.get(issuer.session_id) is not engine
            or event.runtime_session_id != issuer.session_id
            or runtime._app is not issuer.app
            or runtime._chat_store is not self.store
            or runtime._chat_controller is not self.controller
            or runtime._hook_permissions is not self.permissions
            or (
                self.store is not None
                and self.store._sessions.get(issuer.session_id) is not self.session
            )
            or self._session_identity() != self.session_identity
            or getattr(self.store, "persistence", None) is not self.persistence
            or getattr(self.persistence, "db", None) is not self.database
            or runtime._hooks_v2_configured.get(issuer.session_id) is not self.signature
            or (self.controller is not None and self.controller.store is not self.store)
            or getattr(engine, "lifecycle_owner", None) is not self.lifecycle
            or engine.definitions is not self.definitions
            or self._callbacks() != self.callbacks
            or getattr(engine, "native_plugins", None) is not self.native
            or (
                self.native is not None
                and (
                    self.native.snapshots is not self.snapshots
                    or self.native.service is not self.service
                    or self.native.owners is not self.native_owners
                    or tuple(self.native.owners.items()) != self.owner_entries
                    or tuple(
                        getattr(self.service, key, None)
                        for key in (
                            "_thread",
                            "_loop",
                            "_coordinator",
                            "_admission",
                            "fences",
                        )
                    )
                    != self.service_state
                )
            )
            or issuer.capacity is not None
            and runtime._execution_capacity is not issuer.capacity
            or issuer.activity is not None
            and runtime._worktree_activity is not issuer.activity
            or not terminal
            and (
                runtime._disposed
                or issuer.session_id in runtime._admission_fenced_sessions
            )
            or tuple((key, event.data.get(key)) for key, _ in self.match_values)
            != self.match_values
        ):
            raise RuntimeError("Original command event source changed.")
        if self.review is not None:
            current = self.permissions._published
            if (
                current is None
                or current.config.section_stamp != self.review.config.section_stamp
                or current.config.config_path != self.review.config.config_path
                or current.store_path != self.review.store_path
                or (str(current.store_path), str(current.config.config_path))
                in self.permissions._refresh_pending
                or not terminal
                and self.permissions._closed.is_set()
            ):
                raise PermissionError("Command event configuration changed.")
        for plan in self.all_plans:
            if plan.mcp is not None:
                plan.mcp.check_memory()
        # Accepted context outlives its input registration, never its sources.
        if not (
            self.custody is not None
            and self.custody.retired
            and self.custody.owner._drain_outcome is True
        ):
            self._check_registration()
        if self.custody is not None and not all(
            self._reachable(h) for h in self.selected if h.id in self.targets
        ):
            raise PermissionError("Command event target changed.")

    def _check_registration(self):
        for plan in self.all_plans:
            if plan.mcp is not None:
                plan.mcp.check_registration()

    def refresh_permissions(self):
        """Prepare passive context without reactivating the retired execution."""
        if not (
            self.custody is not None
            and self.custody.retired
            and self.custody.owner._drain_outcome is True
        ):
            return
        self._check_sources()
        if _guard.captured_sources(None) != self.sources:
            raise RuntimeError("Command event activation source changed.")
        for plan in self.all_plans:
            if plan.permission is not None:
                plan.permission.refresh()
        self.check_memory()

    def check(self):
        self.check_memory()
        if _guard.captured_sources(None) != self.sources:
            raise RuntimeError("Command event activation source changed.")

    def _reachable(self, handler):
        if self.native is not None and handler.id in self.native.owners:
            snapshot, component = self.native.owners[handler.id]
            from tldw_chatbook.Plugins.service import PluginService

            return bool(
                PluginService.published_current(self.service, snapshot)
                and component in snapshot.selection
            )
        if self.signature is None:
            return True  # Original host-injected resolver remains authoritative.
        return any(
            target.spec is handler or target.spec == handler
            for target in self.signature[1]
            if target in self.permissions._published_targets
            and (str(self.review.store_path), target.config_scope, target.key)
            not in self.permissions._sealed
        )

    def admit(self, handler_ids=None):
        self._check_registration()
        with self._lock:
            if self.closed:
                raise RuntimeError("Command event closed.")
            if self.admitted:
                if not self.ready or self.admission_failed:
                    raise RuntimeError("Command event admission unavailable.")
                return
            self.admitted = True
        issuer = self.issuer
        with issuer._condition:
            issuer._admissions += 1
        try:
            self.check()
            if self.covered and issuer.capacity is not None and issuer.capacity._closed:
                raise CapacityRefused("runtime_closed")
            if (
                not self.covered
                or not issuer.engine.enabled
                or issuer.engine.invalid_admissions
            ):
                return
            targets = tuple(
                h
                for h in self.selected
                if self._reachable(h) and (handler_ids is None or h.id in handler_ids)
            )
            if not targets:
                return
            with execution(sources=self.sources):
                if issuer.capacity is None:
                    issuer.capacity = issuer.runtime.execution_capacity
                if issuer.activity is None:
                    issuer.activity = issuer.runtime.worktree_activity
                self.check()
                with issuer.activity.prelaunch(
                    footprint=WorktreeFootprint(opaque=True)
                ) as claim:
                    if claim is None:
                        raise CapacityRefused("hook_activity_busy")
                    owner = issuer.capacity.begin_execution(
                        origin=self.origin,
                        conversation_id=(
                            self.session.persisted_conversation_id or self.session.id
                        )
                        if self.session
                        else "",
                    )
                    operations = []
                    try:
                        plans = []
                        context = None
                        if any(h.type == "mcp_tool" for h in targets):
                            try:
                                context = issuer.engine.mcp_executor._context(
                                    self.event
                                )
                            except PermissionError:
                                pass
                        for handler in targets:
                            plan = self._select(
                                handler, context, current_chain(), owner, operations
                            )
                            if plan is not None:
                                plans.append(plan)
                        self.plans = {p.handler.id: p for p in plans}
                        self.all_plans = tuple(p for plan in plans for p in plan.walk())
                        self.check()
                        if not all(self._reachable(h) for h in targets):
                            raise PermissionError("Command target revoked.")
                        if operations:
                            claim.bind_owner(owner)
                    except BaseException:
                        for operation in operations:
                            operation.finish()
                        owner.finish_root()
                        raise
                    if not operations:
                        owner.finish_root()
                        return
                    custody = _CommandCustody(self, claim, owner, operations)
                    self.custody = custody
                    by_operation = dict(zip(operations, custody.targets))
                    for plan in self.all_plans:
                        plan.target = by_operation[plan.operation]
                    self.targets = {
                        key: plan.target for key, plan in self.plans.items()
                    }
                    with issuer._condition:
                        issuer._events.add(custody)
                    if self.closed:
                        self.close()
        except BaseException:
            self.admission_failed = True
            raise
        finally:
            self.ready = True
            with issuer._condition:
                issuer._admissions -= 1
                issuer._condition.notify_all()

    def _select(self, handler, context, chain, owner, operations):
        if not self._reachable(handler):
            return None
        mcp = None
        if handler.type == "mcp_tool":
            if context is None:
                return None
            from .mcp_executor import MCPHookSelection

            try:
                mcp = MCPHookSelection.capture(context, handler, chain)
            except (ValueError, PermissionError):
                return None
        operation = owner.reserve_tool()
        operations.append(operation)
        children = {}
        if mcp is not None:
            for name in ("PreToolUse", "PostToolUse", "PostToolUseFailure"):
                selected = []
                for child in self.definitions:
                    reasons = (
                        ("execution_failed", "timeout", "cancelled")
                        if name == "PostToolUseFailure"
                        else (None,)
                    )
                    matched = False
                    for reason in reasons:
                        data = {
                            "tool_id": mcp.definition.tool_id,
                            "provider": mcp.definition.provider,
                        }
                        if reason is not None:
                            data["reason"] = reason
                        try:
                            matched |= matches_handler(
                                child, SimpleNamespace(event=name, data=data)
                            )
                        except UnsupportedEventField:
                            pass
                    if matched:
                        node = self._select(
                            child, context, mcp.chain, owner, operations
                        )
                        if node is not None:
                            selected.append(node)
                children[name] = tuple(selected)
        return _HeldHandler(handler, operation, mcp, children)

    def retain_host(self):
        with self._lock:
            self.hosts += 1

    def host_done(self):
        with self._lock:
            self.hosts -= 1
            done = not self.hosts and self.closed
        if done and self.custody is not None:
            self.custody.host_done()

    def close(self):
        with self._lock:
            self.closed = True
            done = not self.hosts
        if self.custody is not None:
            for plan in self.plans.values():
                plan.discard()
            if done:
                self.custody.host_done()


@dataclass(eq=False)
class _HeldHandler:
    handler: object
    operation: object
    mcp: object | None
    children: dict
    target: object | None = None
    issued: dict = field(default_factory=dict)
    permission: object | None = None

    def walk(self):
        yield self
        for candidates in self.children.values():
            for child in candidates:
                yield from child.walk()

    def discard(self):
        if self.target is not None and not self.target.started:
            self.target.cancel_before_start()
            self.producer_returned()

    def producer_returned(self):
        for phase, candidates in self.children.items():
            if phase not in self.issued:
                for child in candidates:
                    child.discard()


class _NestedEvent:
    covered = True

    def __init__(self, root, job, event):
        self.root, self.job, self.event = root, job, event
        self.issuer = root.issuer
        plan, hook_run = job.native_plan, job.hook_run
        if (
            hook_run is None
            or hook_run._event_definitions.get(event.event_id) != plan.mcp.definition
            or event.run_id != hook_run.run_id
            or event.event not in plan.children
            or current_chain().visits != job.chain.visits
        ):
            raise PermissionError("Original nested event identity changed.")
        previous = plan.issued.get(event.event)
        if previous is not None and previous.event is not event:
            raise PermissionError("Nested event already issued.")
        self.plans = {p.handler.id: p for p in plan.children[event.event]}
        self.targets = {key: p.target for key, p in self.plans.items()}
        self.match_values = tuple(
            (key, event.data.get(key))
            for key in sorted(
                {k for p in self.plans.values() for k in p.handler.match or ()}
            )
        )
        self.closed, self.hosts = False, 0
        self._lock = Lock()
        self.check_memory(event)
        root.retain_host()
        plan.issued[event.event] = self

    @property
    def selected(self):
        return tuple(p.handler for p in self.plans.values())

    def check_memory(self, event=None):
        self.root.check_memory()
        event = event or self.event
        if (
            tuple((key, event.data.get(key)) for key, _ in self.match_values)
            != self.match_values
        ):
            raise PermissionError("Nested matcher source changed.")
        if not all(self.root._reachable(p.handler) for p in self.plans.values()):
            raise PermissionError("Nested target revoked.")

    def check(self):
        self.root.check()
        self.check_memory()

    def admit(self, handler_ids=None):
        self.root._check_registration()
        self.check()

    def retain_host(self):
        with self._lock:
            self.hosts += 1

    def host_done(self):
        with self._lock:
            self.hosts -= 1
            done = self.closed and not self.hosts
        if done:
            self.root.host_done()

    def close(self):
        with self._lock:
            if self.closed:
                return
            self.closed = True
            done = not self.hosts
        for plan in self.plans.values():
            plan.discard()
        if done:
            self.root.host_done()
