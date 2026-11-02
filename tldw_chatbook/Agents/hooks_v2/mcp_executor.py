"""MCP hook adapters composed with the existing tool and resource owners."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from uuid import uuid4

from ..agent_models import ToolCall, ToolResult
from ..mcp_tool_provider import (
    MCPInvocationPolicy,
    MCPToolProvider,
    capture_mcp_result,
    current_mcp_invocation_policies,
    restrict_mcp_invocation,
)
from ..run_context import use_run_id
from .causality import current_chain
from .mcp_results import normalize_hook_result
from .models import HookEvent, HookHandler
from .validation import INPUT_BYTES, TEMPLATE_RE, _json_tree


def expand_input(event: HookEvent, handler: HookHandler) -> dict:
    """Expand documented typed template values once, without recursive evaluation."""
    source = {
        key: value for key, value in event.model_dump().items() if value is not None
    }

    def resolve(path):
        value = source
        for key in path.split("."):
            if not isinstance(value, Mapping) or key not in value:
                raise ValueError("hook_mcp_template_missing")
            value = value[key]
        return value

    def scalar(match):
        value = resolve(match.group(1))
        if isinstance(value, (Mapping, list, tuple)):
            raise ValueError("hook_mcp_template_not_scalar")  # noqa: TRY004
        return value if isinstance(value, str) else json.dumps(value, allow_nan=False)

    def expand(value):
        match = TEMPLATE_RE.fullmatch(value)
        if match:
            return resolve(match.group(1))
        chunks, size, offset = [], 0, 0
        for match in TEMPLATE_RE.finditer(value):
            chunk = value[offset : match.start()] + scalar(match)
            size += len(chunk.encode("utf-8"))
            if size > INPUT_BYTES:
                raise ValueError("hook_mcp_template_overflow")
            chunks.append(chunk)
            offset = match.end()
        chunks.append(value[offset:])
        return "".join(chunks)

    return _json_tree(handler.input or {}, string_transform=expand, allow_frozen=True)


@dataclass(frozen=True)
class MCPHookContext:
    """A private host invocation view contained by its exact live parent."""

    registry: object
    lifecycle: object
    parent_scope: str
    run_id: str
    session_id: str
    allowed_names: frozenset[str]
    current: Callable[[], bool]
    required_handler_ids: Callable
    workspace_id: str | None = None
    turn_id: str | None = None
    parent_run_id: str | None = None
    input_scope: str | None = None
    definition_resolver: Callable | None = None
    runtime_current: Callable[[], bool] | None = None
    budget_run_id: str | None = None
    captured_readiness: tuple | None = None
    source_scope: str | None = None

    def check(self) -> None:
        if not self.current() or not self.lifecycle.checkpoints.is_current(
            self.parent_scope
        ):
            raise PermissionError("hook_mcp_parent_refused")

    def definition(self, call):
        self.check()
        if call.name not in self.allowed_names:
            raise PermissionError("hook_mcp_tool_not_permitted")
        definition = (
            self.definition_resolver(call)
            if self.definition_resolver is not None
            else self.registry.snapshot_for_hook(call.name)
        )
        required = self.required_handler_ids(definition)
        if self.captured_readiness is not None:
            if required is None or not any(
                captured == definition and requirements == required
                for captured, requirements in self.captured_readiness
            ):
                raise PermissionError("hook_mcp_teardown_readiness_changed")
        else:
            self.lifecycle.checkpoints.assert_continuation_dependencies(
                self.parent_scope, required_handler_ids=required
            )
        return definition

    def resolve(self, handler):
        """Resolve exact server key/tool name from the ACTUAL registry owners."""
        matches = []
        for entry in self.registry.list_catalog():
            if entry.name not in self.allowed_names:
                continue
            owner = self.registry.resolve_owner_for_name(entry.name)
            if owner is None:
                continue
            provider = owner[1]
            # Existing Console collision wrapper delegates to this exact owner.
            provider = getattr(provider, "_provider", provider)
            from tldw_chatbook.Plugins.mcp_provider import MCPProviderGroup

            if isinstance(provider, MCPProviderGroup):
                provider = provider.provider_for(entry.name)
            if (
                isinstance(provider, MCPToolProvider)
                and provider.hook_target(handler.server, handler.tool) == entry.name
            ):
                matches.append((entry.name, provider))
        if len(matches) != 1:
            raise PermissionError("hook_mcp_tool_unavailable")
        name, provider = matches[0]
        self.definition(ToolCall(name, {}, ""))
        return name, provider


@dataclass(frozen=True)
class MCPHookSelection:
    """Original metadata ceiling; admission callbacks remain in the real worker."""

    context: MCPHookContext
    handler: HookHandler
    name: str
    provider: object
    definition: object
    entry: tuple
    service: object
    generation: int
    chain: tuple
    context_key: tuple
    source_bindings: tuple
    maximum: tuple
    plugin_mapping: str | None
    schema: str

    @classmethod
    def capture(cls, context, handler, chain):
        name, provider = context.resolve(handler)
        definition = context.definition(ToolCall(name, {}, ""))
        ancestry = (
            tuple((h, t) for _, h, t in chain.visits)
            if hasattr(chain, "visits")
            else chain
        )
        if len(ancestry) >= 4 or any(
            h == handler.id or t == definition.tool_id for h, t in ancestry
        ):
            raise ValueError("hook_dependency_cycle")
        entry = provider._entry_by_llm_name[name]
        if entry[1].state == "deny":
            raise PermissionError("hook_mcp_tool_not_permitted")
        executor = context.lifecycle.engine.mcp_executor
        with executor._context_lock:
            keys = tuple(
                (kind, key)
                for kind in ("_contexts", "_runtime_contexts", "_teardown_contexts")
                for key, value in getattr(executor, kind).items()
                if value is context
            )
        if not keys:
            raise PermissionError("hook_mcp_original_context_unavailable")
        # Equal replacement snapshots cannot inherit original event admission.
        bindings = []
        for source, names in (
            (
                provider,
                (
                    "_main_loop",
                    "_persona_policy_provider",
                    "_profile_id",
                    "_runtime_source_provider",
                    "_approval_callback",
                    "plugins",
                    "snapshot",
                    "ownership",
                ),
            ),
            (provider._service, ("local_service",)),
            (getattr(provider._service, "local_service", None), ("client", "store")),
            (
                getattr(provider, "plugins", None),
                ("_thread", "_loop", "_coordinator", "_admission", "fences"),
            ),
        ):
            bindings.extend(
                (source, name, getattr(source, name, None)) for name in names
            )
        mappings = getattr(provider, "_tool_mappings", None)
        return cls(
            context,
            handler,
            name,
            provider,
            definition,
            entry,
            provider._service,
            context.registry._catalog_generation,
            ancestry + ((handler.id, definition.tool_id),),
            keys,
            tuple(bindings),
            cls._maximum(provider),
            json.dumps(mappings, sort_keys=True) if mappings is not None else None,
            json.dumps(entry[0].input_schema, sort_keys=True),
        )

    @staticmethod
    def _maximum(provider):
        return (
            provider._maximum_tool_ids,
            None
            if provider._maximum_definition_hashes is None
            else frozenset(provider._maximum_definition_hashes.items()),
            provider._owned_profile_ids,
            provider._builtin_raw_name_exclusions,
        )

    def check_registration(self):
        executor = self.context.lifecycle.engine.mcp_executor
        if not any(
            getattr(executor, kind).get(key) is self.context
            for kind, key in self.context_key
        ):
            raise PermissionError("hook_mcp_original_registration_retired")

    def check_memory(self):
        if (
            self.provider._service is not self.service
            or any(
                getattr(source, name, None) is not original
                for source, name, original in self.source_bindings
            )
            or self._maximum(self.provider) != self.maximum
            or (
                self.plugin_mapping is not None
                and json.dumps(self.provider._tool_mappings, sort_keys=True)
                != self.plugin_mapping
            )
            or self.provider._entry_by_llm_name.get(self.name) is not self.entry
            or self.context.registry._catalog_generation != self.generation
            or json.dumps(self.entry[0].input_schema, sort_keys=True) != self.schema
        ):
            raise PermissionError("hook_mcp_original_selection_changed")
        if self.plugin_mapping is not None:
            from tldw_chatbook.Plugins.service import PluginService

            if not PluginService.published_current(
                self.provider.plugins, self.provider.snapshot
            ):
                raise PermissionError("hook_mcp_original_plugin_revoked")

    def check(self):
        self.check_registration()
        self.check_memory()
        if self.context.definition(ToolCall(self.name, {}, "")) != self.definition:
            raise PermissionError("hook_mcp_original_definition_changed")


@dataclass(eq=False)
class _MCPDelivery:
    executor: object
    event: HookEvent
    handler: HookHandler
    ticket: object
    execution: object
    cancellation: asyncio.Event
    context: MCPHookContext
    inherited_policies: tuple[MCPInvocationPolicy, ...] = ()
    cancel_event: threading.Event = field(default_factory=threading.Event)
    bridges: list = field(default_factory=list)
    requests: list = field(default_factory=list)
    task: asyncio.Task | None = None
    worker: asyncio.Task | None = None
    result_check: Callable[[], bool] | None = None
    active_started: float | None = None
    active_seconds: float = 0.0
    approval_end: float | None = None
    stopped: bool = False
    pending_release: bool = False
    wait_depth: int = 0
    approval_depth: int = 0
    native_target: object | None = None
    native_plan: object | None = None
    hook_run: object | None = None
    chain: object | None = None
    worker_returned: threading.Event = field(default_factory=threading.Event)

    @property
    def deadline(self):
        inherited = [policy.deadline for policy in self.inherited_policies]
        return min([self.execution.deadline, *inherited])

    @property
    def active_used(self):
        return self.active_seconds + (
            time.monotonic() - self.active_started
            if self.active_started is not None
            else 0
        )

    def current(self):
        self.context.check()
        if self.native_plan is not None:
            self.native_plan.mcp.check_registration()
            self.native_plan.mcp.check_memory()
            self.native_target.event.source.check()
        for policy in self.inherited_policies:
            policy.check()
        engine = self.executor.engine
        try:
            on_owner_loop = asyncio.get_running_loop() is engine.loop
        except RuntimeError:
            on_owner_loop = False
        authority = (
            engine._effect_authority_check if on_owner_loop else engine.authority_check
        )
        return (
            not self.stopped
            and not self.execution.closed
            and not self.cancellation.is_set()
            and self.active_used < self.handler.timeout_seconds
            and self.execution.active_seconds + self.active_used < 60
            and authority(self.handler, self.event, "launch")
        )

    def stop(self):
        self.stopped = True
        self.cancel_event.set()
        for bridge in tuple(self.bridges):
            bridge.future.cancel()

    def bridge(self, observation):
        self.bridges.append(observation)
        if self.cancel_event.is_set():
            observation.future.cancel()

    def own_request(self, plugins, token):
        rows = [row for row in plugins.live_runs() if row.lease_token == token]
        if len(rows) != 1:
            raise PermissionError("hook_mcp_custody_unavailable")
        self.requests.append(rows[0])

    def terminal(self):
        return (
            self.worker is not None
            and (self.native_target is None or self.worker_returned.is_set())
            and self.worker.done()
            and not self.worker.cancelled()
            and all(
                bridge.completed.is_set()
                and bridge.dispatch.state in {"settled", "not_started"}
                for bridge in self.bridges
            )
            and all(row.completed.is_set() for row in self.requests)
        )

    async def suspend(self):
        if self.ticket.observation:
            return
        if self.active_started is not None:
            self.active_seconds += time.monotonic() - self.active_started
            self.active_started = None
        self.ticket.suspend()

    async def resume(self):
        if self.ticket.observation:
            return
        if self.cancel_event.is_set() or time.monotonic() >= self.deadline:
            raise PermissionError("hook_mcp_cancelled")
        acquire = asyncio.create_task(self.ticket.acquire(retain_on_cancel=True))
        try:
            while not acquire.done():
                if self.cancel_event.is_set() or time.monotonic() >= self.deadline:
                    raise PermissionError("hook_mcp_cancelled")
                await asyncio.wait(
                    {acquire},
                    timeout=min(0.05, max(0, self.deadline - time.monotonic())),
                )
            await acquire
            self.active_started = time.monotonic()
        finally:
            if not acquire.done():
                acquire.cancel()
                await asyncio.gather(acquire, return_exceptions=True)

    async def enter_wait(self, kind):
        self.wait_depth += 1
        if kind == "approval":
            self.approval_depth += 1
            if self.approval_end is None:
                self.approval_end = min(self.deadline, time.monotonic() + 120)
        if self.wait_depth == 1:
            await self.suspend()

    async def leave_wait(self, kind):
        if kind == "approval":
            self.approval_depth -= 1
        self.wait_depth -= 1
        if self.wait_depth == 0:
            await self.resume()

    @contextlib.contextmanager
    def wait_scope(self, kind):
        self.executor.engine._sync(self.enter_wait(kind))
        try:
            yield
        finally:
            self.executor.engine._sync(self.leave_wait(kind))


class MCPHookExecutor:
    """Orchestrate existing ToolHookRun, provider, checkpoint and ticket owners."""

    def __init__(self, engine):
        self.engine = engine
        self._contexts = {}
        self._runtime_contexts = {}
        self._teardown_contexts = {}
        self._context_lock = threading.RLock()
        self._jobs = set()
        self._reaper = None

    def bind_context(self, context: MCPHookContext, *, run_id: str | None = None):
        context.check()
        with self._context_lock:
            self._contexts[run_id or ("session:" + context.session_id)] = context

    def retire_context(self, run_id):
        with self._context_lock:
            self._contexts.pop(run_id, None)

    def retain_runtime(self, session_id: str) -> None:
        """Retain a successful session's ceiling without its pending input scope."""
        key = "session:" + session_id
        with self._context_lock:
            context = self._contexts.get(key)
            previous = self._runtime_contexts.get(session_id)
        if context is None or context.runtime_current is None:
            return
        context.check()
        if previous is not None and previous.current():
            # Preserve the same registry policy and original cap counters.
            return
        if not context.lifecycle.live:
            raise PermissionError("hook_mcp_session_not_live")
        runtime = replace(
            context,
            parent_scope=context.lifecycle.scope_id,
            run_id=context.lifecycle.scope_id,
            turn_id=None,
            input_scope=None,
            current=context.runtime_current,
            budget_run_id=context.budget_run_id or context.run_id,
        )
        runtime.check()
        with self._context_lock:
            if (
                self._contexts.get(key) is context
                and self._runtime_contexts.get(session_id) is previous
            ):
                self._runtime_contexts[session_id] = runtime

    def retire_scope(self, scope_id: str) -> None:
        """Remove only the exact retired binding, never a newer pending input."""
        with self._context_lock:
            for key, context in tuple(self._contexts.items()):
                if context.parent_scope == scope_id:
                    self._contexts.pop(key, None)

    def capture_teardown(self, lifecycle, scope_id: str):
        """Capture positively checked capability provenance before ordinary seal."""
        with self._context_lock:
            context = self._runtime_contexts.get(lifecycle.session_id)
        if context is None:
            return False
        try:
            context.check()
            readiness = []
            for name in context.allowed_names:
                try:
                    definition = context.definition(ToolCall(name, {}, ""))
                    requirements = context.required_handler_ids(definition)
                    lifecycle.checkpoints.assert_continuation_dependencies(
                        context.parent_scope, required_handler_ids=requirements
                    )
                    readiness.append((definition, tuple(requirements)))
                except Exception:  # noqa: BLE001, S112 -- unknown readiness refuses
                    continue
            captured = replace(
                context,
                parent_scope=scope_id,
                run_id=scope_id,
                source_scope=context.parent_scope,
                captured_readiness=tuple(readiness),
            )
            with self._context_lock:
                self._teardown_contexts[lifecycle.session_id] = captured
            return True
        except Exception:  # noqa: BLE001 -- best effort notification
            return False

    def _context(self, event):
        with self._context_lock:
            if event.run_id:
                context = self._contexts.get(event.run_id)
            elif event.event in {"Interrupt", "SessionEnd"}:
                context = self._teardown_contexts.get(event.runtime_session_id)
                if context is None and event.event == "Interrupt":
                    context = self._runtime_contexts.get(event.runtime_session_id)
            else:
                context = self._contexts.get("session:" + event.runtime_session_id)
                if context is None:
                    context = self._runtime_contexts.get(event.runtime_session_id)
        if context is None:
            raise PermissionError("hook_mcp_context_unavailable")
        context.check()
        return context

    @property
    def cleanup_pending(self):
        return bool(self._jobs)

    def start(self, event, handler, ticket, execution, cancellation, *, context=None):
        from ..legacy_hook_ownership import current_target

        target = current_target()
        plan = execution.native.plans.get(handler.id) if target is not None else None
        if target is not None:
            if plan is None or plan.target is not target or plan.mcp is None:
                raise PermissionError("hook_mcp_original_target_unavailable")
            plan.mcp.check()
            context = plan.mcp.context
        context = context or self._context(event)
        context.check()
        if event.event in {"Interrupt", "SessionEnd"}:
            context = replace(context, input_scope=None, turn_id=None)
        job = _MCPDelivery(
            self,
            event,
            handler,
            ticket,
            execution,
            cancellation,
            context,
            current_mcp_invocation_policies(),
            native_target=target,
            native_plan=plan,
        )
        job.active_started = time.monotonic()
        if target is not None:
            target.process_attempted()
            target.retain_callback()
        self._jobs.add(job)
        job.task = asyncio.create_task(self._run(job))
        if target is not None:

            def returned(done):
                target.proven = job.terminal()
                if not target.proven:
                    # The original reaper still owns proof; do not retire uncertain custody.
                    target.operation.mark_stopping()
                if job.worker is None:
                    plan.producer_returned()
                # Pending bridge/request proof belongs to the original reaper,
                # even after the logical job callback has returned.
                if not job.pending_release:
                    target.callback_returned()

            job.task.add_done_callback(returned)
        return job

    async def invoke(self, event: HookEvent, handler: HookHandler):
        """Use the owning engine so callers cannot bypass its lifetime admission."""
        scope = self.engine.begin_event(event)
        try:
            return await self.engine.fire_handler_async(scope, handler.id, event)
        finally:
            scope.close()

    def _preflight(self, context, hook_run, handler, chain, visited=None):
        """Traverse known required guard edges before any MCP dispatch."""
        visited = set() if visited is None else visited
        name, _provider = context.resolve(handler)
        definition = context.definition(ToolCall(name, {}, ""))
        next_chain = chain.enter(uuid4().hex, handler.id, definition.tool_id)
        identity = (handler.id, definition.tool_id)
        if identity in visited:
            return
        visited.add(identity)
        event, definition = hook_run.preview_call(ToolCall(name, {}, ""))
        from .matching import UnsupportedEventField, matches_handler
        from .validation import handler_phase

        for child in self.engine.definitions:
            if child.event != event.event:
                continue
            try:
                if not matches_handler(child, event):
                    continue
            except UnsupportedEventField:
                pass
            dependency, _failure = self.engine._dependency_status(child, event)
            if (
                handler_phase(child, dependency_required=dependency) != "observe"
                and child.type == "mcp_tool"
            ):
                self._preflight(context, hook_run, child, next_chain, visited)

    def _invoke_worker_bound(self, job):
        with self.engine._continuation_scope(job.execution):
            return self._invoke_worker(job)

    def _invoke_worker(self, job):
        from .tool_pipeline import ToolHookRun

        context = job.context
        name, provider = context.resolve(job.handler)
        call = ToolCall(name, expand_input(job.event, job.handler), call_id=uuid4().hex)
        definition = context.definition(call)
        chain = current_chain().enter(
            job.event.event_id, job.handler.id, definition.tool_id
        )
        job.chain = chain
        operation_id = "hook-operation:" + uuid4().hex
        hook_run = job.hook_run = ToolHookRun(
            self.engine,
            run_id=operation_id,
            session_id=context.session_id,
            turn_id=context.turn_id or operation_id,
            resolve_definition=context.definition,
            should_cancel=job.cancel_event.is_set,
            required_handler_ids=lambda: (),
            workspace_id=context.workspace_id,
            parent_run_id=context.run_id,
            lifecycle=context.lifecycle,
            parent_scope=None,
            context_owner=context.input_scope,
            containing_hook=(job.event, job.handler.id),
        )
        self.bind_context(context, run_id=operation_id)
        policy = MCPInvocationPolicy(
            current=job.current,
            deadline=job.deadline,
            cancel_event=job.cancel_event,
            allow_approval=(
                not job.ticket.observation
                and job.event.event
                not in {"ApprovalRequested", "Interrupt", "SessionEnd"}
            ),
            wait_scope=job.wait_scope,
            approval_deadline=lambda: job.approval_end or job.deadline,
            on_bridge=job.bridge,
            on_owned_request=job.own_request,
        )
        result = ToolResult.blocked("hook_mcp_not_dispatched")
        settled = False
        try:
            # This owner is a continuation operation in the SAME store. Its
            # actual parent dependency query remains in context.definition.
            self._preflight(context, hook_run, job.handler, current_chain())
            with (
                chain.scope(),
                restrict_mcp_invocation(policy),
                use_run_id(context.budget_run_id or context.run_id),
            ):
                with job.wait_scope("nested"):
                    call = hook_run.prepare_call(call)
                    hook_run.validate_dispatch(call)
                policy.check()
                with capture_mcp_result(provider) as capture:
                    expected = hook_run.validate_dispatch(call, accept_context=False)
                    result = context.registry.invoke_by_name(
                        call.name, call.args, expected_definition=expected
                    )
                    evidence = capture.consume_authorized(result)
                with job.wait_scope("nested"):
                    hook_run.install_result(call, result)
                    settled = True
                    hook_run.settle()
                policy.check()
                if evidence is None or result.dispatch_state != "settled":
                    raise ValueError("hook_mcp_execution_failed")
                raw, owner_current = evidence

                def result_current():
                    policy.check()
                    if context.definition(call) != expected:
                        raise PermissionError("hook_mcp_result_definition_changed")
                    owner_current()
                    # The normal owned authority port can block. Its completion
                    # is not permission to outlive this original event/scope.
                    policy.check()
                    if context.definition(call) != expected:
                        raise PermissionError("hook_mcp_result_definition_changed")
                    return True

                job.result_check = result_current
                result_current()
                return normalize_hook_result(raw, job.handler)
        finally:
            self.retire_context(operation_id)
            # Refused preparation/checkpoint acceptance still owns its original
            # host scopes; use the normal ToolHookRun discard/terminal join.
            try:
                if not settled:
                    hook_run.settle()
            finally:
                context.lifecycle.close_scope(operation_id)

    async def _wait_worker(self, job):
        """Join sequential work under this same ticket and original deadline."""
        failure = None
        accepted = None
        while not job.worker.done():
            if job.cancellation.is_set() or job.execution.closed:
                failure = "cancelled"
                break
            if time.monotonic() >= job.deadline:
                failure = "event_deadline"
                break
            if (
                job.approval_depth
                and job.approval_end is not None
                and time.monotonic() >= job.approval_end
            ):
                failure = "approval_deadline"
                break
            if job.active_used >= min(
                job.handler.timeout_seconds, 60 - job.execution.active_seconds
            ):
                failure = "handler_deadline"
                break
            try:
                job.context.check()
                for policy in job.inherited_policies:
                    policy.check()
            except Exception:  # noqa: BLE001 -- refuse
                failure = "authority_refused"
                break
            await asyncio.wait({job.worker}, timeout=0.025)
        if failure:
            job.stop()
            await job.suspend()
            await asyncio.wait({job.worker}, timeout=5)
        if job.worker.done():
            try:
                accepted = job.worker.result()
            except Exception:  # noqa: BLE001 -- refuse
                failure = failure or "hook_mcp_execution_failed"
        return accepted, failure

    def _native_worker(self, job, callback, *args):
        from ..activation import _guard, execution

        try:
            with (
                _guard.worker_isolation(),
                execution(sources=job.native_target.event.source.sources),
                job.native_target.event.owner.activate(),
            ):
                job.native_target.event.source.check()
                return callback(*args)
        finally:
            if job.native_plan is not None and callback == self._invoke_worker_bound:
                job.native_plan.producer_returned()
            job.worker_returned.set()
            if job.native_target is not None:
                job.native_target.callback_returned()

    def _worker(self, job, callback, *args):
        if job.native_target is None:
            return asyncio.create_task(asyncio.to_thread(callback, *args))
        job.worker_returned.clear()
        job.native_target.retain_callback()
        return asyncio.create_task(
            asyncio.to_thread(self._native_worker, job, callback, *args)
        )

    async def _run(self, job):
        from .engine import HookEventOutcome

        failure = None
        accepted = None
        try:
            job.worker = self._worker(job, self._invoke_worker_bound, job)
            accepted, failure = await self._wait_worker(job)
            if accepted is not None:
                current, code = await self.engine._authority_status(
                    job.handler, job.event, "accept"
                )
                if (
                    not current
                    or job.cancel_event.is_set()
                    or job.execution.closed
                    or time.monotonic() >= job.deadline
                ):
                    accepted = None
                    failure = code or "authority_refused"
                else:
                    # Sequential continuation, never another MCP request. Keep
                    # this actual check in the same worker/custody join so a
                    # cancelled awaiter cannot release a running owner check.
                    job.worker = self._worker(job, job.result_check)
                    checked, failure = await self._wait_worker(job)
                    if (
                        not checked
                        or failure
                        or job.cancel_event.is_set()
                        or job.execution.closed
                        or time.monotonic() >= job.deadline
                    ):
                        accepted = None
                        failure = failure or "authority_refused"
            await job.suspend()
            job.execution.active_seconds += job.active_seconds
            if job.terminal():
                job.ticket.release()
                self._jobs.discard(job)
            else:
                accepted = None
                failure = failure or "hook_mcp_cleanup_pending"
                job.pending_release = True
                if self._reaper is None or self._reaper.done():
                    self._reaper = asyncio.create_task(self._reap())
            if failure:
                return HookEventOutcome(
                    failures=(
                        self.engine._failure(
                            job.handler,
                            job.event,
                            failure,
                            dependency=job.execution.dependencies.get(
                                job.handler.id, (None,)
                            )[0],
                        ),
                    ),
                    outstanding_cleanup=(job.handler.id,) if not job.terminal() else (),
                )
            return HookEventOutcome(accepted=((job.handler.id, accepted),))
        except BaseException:
            job.stop()
            await job.suspend()
            job.pending_release = True
            if self._reaper is None or self._reaper.done():
                self._reaper = asyncio.create_task(self._reap())
            raise
        finally:
            job.result_check = None

    async def _reap(self):
        while self._jobs:
            for job in tuple(self._jobs):
                if job.pending_release and job.task.done() and job.terminal():
                    if job.worker is not None and not job.worker.cancelled():
                        job.worker.exception()
                    job.ticket.release()
                    self._jobs.discard(job)
                    if job.native_target is not None:
                        job.native_target.process_terminal(True)
                        job.native_target.callback_returned()
            if self._jobs:
                await asyncio.sleep(0.1)

    def retire_teardown(self):
        with self._context_lock:
            contexts = tuple(self._teardown_contexts.values())
            self._teardown_contexts.clear()
            self._runtime_contexts.clear()
            self._contexts.clear()
        for context in contexts:
            context.lifecycle.close_scope(context.parent_scope)

    async def close(self, *, deadline):
        for job in tuple(self._jobs):
            job.stop()
        tasks = {job.task for job in self._jobs if job.task is not None}
        if tasks:
            await asyncio.wait(tasks, timeout=max(0, deadline - time.monotonic()))
