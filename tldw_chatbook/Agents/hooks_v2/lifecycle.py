"""Live-session initialization and host-owned lifecycle checkpoint composition."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ..agent_models import AgentConfig, RunBudget, RunOutcome
from .checkpoints import HookCheckpointError, HookCheckpointStore
from .context import ContextLedger
from .engine import HookEngine, HookEventOutcome
from .models import HookEvent
from .validation import parse_event


def _json_projection(value):
    if isinstance(value, Mapping):
        return {key: _json_projection(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_projection(item) for item in value]
    return value


def lifecycle_event(
    name: str,
    session_id: str,
    *,
    data: dict[str, Any] | None = None,
    run_id: str | None = None,
    turn_id: str | None = None,
    parent_run_id: str | None = None,
    workspace_id: str | None = None,
    initiator: str = "manual",
) -> HookEvent:
    value = {
        "protocol_version": 2,
        "event_id": uuid4().hex,
        "event": name,
        "timestamp": datetime.now(UTC).isoformat(),
        "runtime_session_id": session_id,
        "initiator": initiator,
        "origin": "host_lifecycle",
        "causal_chain_id": session_id,
        "causal_depth": 0,
        "data": data or {},
    }
    for key, item in (
        ("run_id", run_id),
        ("turn_id", turn_id),
        ("parent_run_id", parent_run_id),
        ("workspace_id", workspace_id),
    ):
        if item is not None:
            value[key] = item
    return parse_event(value)


def narrow_child(
    config: AgentConfig, outcomes: HookEventOutcome, tool_ids: Mapping[str, str]
) -> tuple[AgentConfig, frozenset[str]]:
    """Intersect fully inherited authority, including zero-as-unlimited caps."""
    selected = frozenset(tool_ids.values())
    budget = config.budget
    unlimited = {"max_tool_result_chars", "max_total_tokens", "max_tool_call_seconds"}
    for _handler, result in outcomes.accepted:
        if result.child_limits is None:
            continue
        limits = result.child_limits
        if "tool_ids" in limits:
            proposed = frozenset(limits["tool_ids"])
            if not proposed <= selected:
                raise HookCheckpointError("hook_child_tools_widened")
            selected = proposed
        values = {}
        for key, value in limits.get("budget_caps", {}).items():
            inherited = getattr(budget, key)
            if key in unlimited and inherited == 0:
                values[key] = value
            elif key in unlimited and value == 0:
                values[key] = inherited
            else:
                values[key] = min(inherited, value)
        budget = replace(budget, **values)
    allowed = tuple(
        name for name in config.allowed_tools if tool_ids.get(name) in selected
    )
    return replace(config, allowed_tools=allowed, budget=budget), selected


class HookSessionLifecycle:
    """One shared coordinator; operation IDs never masquerade as AgentRuns IDs."""

    def __init__(
        self,
        engine: HookEngine,
        session_id: str,
        *,
        current: Callable[[], bool] = lambda: True,
    ) -> None:
        self.engine = engine
        self.session_id = session_id
        self.current = current
        self.scope_id = "session:" + uuid4().hex
        self.checkpoints = HookCheckpointStore(accept_current=engine.effects_current)
        self.checkpoints.bind_owner(self.scope_id)
        self.context = ContextLedger(
            owners=self.checkpoints.owners,
            effects_current=engine.effects_current,
            lock=self.checkpoints._condition,
        )
        self.live = False
        self._sealed = False
        self._reservations = {}
        self._executions = {}
        self.turn_scope = None
        self._pending = set()
        self._handoff = None
        self.diagnostics = {"late_context": 0}
        self.terminal_budgets = {}
        self.terminal_budget_times = {}
        self.inherited_budgets = {}

    def inherit_root_budget(self, turn_id: str, config: AgentConfig) -> AgentConfig:
        """Narrow a new root to the actual settled parent's remaining budget."""
        inherited = self.inherited_budgets.pop(turn_id, None)
        if inherited is None:
            return config
        from dataclasses import fields

        inherited, admitted_at = inherited
        wall = inherited.max_wall_seconds - (time.monotonic() - admitted_at)
        if wall <= 0:
            raise HookCheckpointError("hook_continuation_budget_exhausted")
        inherited = replace(inherited, max_wall_seconds=wall)
        unlimited = {
            "max_total_tokens",
            "max_tool_call_seconds",
            "max_tool_result_chars",
        }
        values = {}
        for field in fields(inherited):
            old = getattr(inherited, field.name)
            current = getattr(config.budget, field.name)
            if field.name in unlimited and (old == 0 or current == 0):
                values[field.name] = max(old, current)
            else:
                values[field.name] = min(old, current)
        return replace(config, budget=replace(config.budget, **values))

    def record_root_budget(
        self,
        turn_id: str,
        outcome: RunOutcome,
        budget: RunBudget,
        elapsed_seconds: float,
    ) -> None:
        """Capture actual usage before terminal scope retirement loses the run."""
        self.terminal_budget_times[turn_id] = time.monotonic()
        steps = budget.max_steps - len(outcome.steps)
        turns = budget.max_model_turns - sum(
            step.kind == "model" for step in outcome.steps
        )
        wall = budget.max_wall_seconds - elapsed_seconds
        tokens = (
            budget.max_total_tokens - outcome.total_tokens
            if budget.max_total_tokens
            else 0
        )
        if (
            outcome.status != "done"
            or min(steps, turns, wall) <= 0
            or budget.max_total_tokens
            and tokens <= 0
        ):
            self.terminal_budgets[turn_id] = False
            return
        self.terminal_budgets[turn_id] = replace(
            budget,
            max_steps=steps,
            max_model_turns=turns,
            max_wall_seconds=wall,
            max_total_tokens=tokens,
            max_subagents=max(0, budget.max_subagents - outcome.subagents_spawned),
        )

    def event(self, name: str, **kwargs: Any) -> HookEvent:
        key = getattr(self, "context_key", None)
        if key is not None:
            kwargs.setdefault("workspace_id", key[0])
        return lifecycle_event(name, self.session_id, **kwargs)

    def reserve(self, event: HookEvent) -> str:
        if event.event != "SessionStart" or not self.current():
            raise HookCheckpointError("invalid session initialization")
        token = self.begin(event, self.scope_id)
        self._reservations[token] = (event, None)
        return token

    async def initialize(self, token: str) -> HookEventOutcome:
        event, _ = self._reservations[token]
        result = await self._execute(token)
        if token not in self._reservations or not self.current():
            raise HookCheckpointError("session initialization cancelled")
        self._reservations[token] = (event, result)
        return result

    def publish(self, token: str) -> None:
        _event, result = self._reservations.pop(token)
        if (
            result is None
            or not self.current()
            or not result.allowed
            or result.outstanding_cleanup
        ):
            raise HookCheckpointError("required session initialization failed")
        self.checkpoints.accept(token, result)
        self.checkpoints.assert_next_input_allowed(self.scope_id)
        self.live = True

    def cancel(self, token: str) -> None:
        self._reservations.pop(token, None)
        execution = self._executions.pop(token, None)
        if execution is not None:
            execution[1].close()

    def open_scope(self, *, parent: str | None = None) -> str:
        scope = "operation:" + uuid4().hex
        self.checkpoints.bind_owner(scope, parent or self.scope_id)
        return scope

    def begin(
        self,
        event: HookEvent,
        owner: str,
        *,
        current: Callable[[], bool] = lambda: True,
        publish: bool = True,
        context_owner: str | None = None,
    ) -> str:
        scope = self.engine.begin_event(event)
        try:
            plans = self.engine.plan_handlers(scope, event)
            required = tuple(
                p.handler_id
                for p in plans
                if p.explicit_required
                or (event.event == "SubagentStart" and p.phase == "validate")
            )
            dependent = tuple(p.handler_id for p in plans if p.dependency_required)
            required += tuple(
                f"config:{r.index}"
                for r in self.engine.invalid_admissions
                if r.policy in {"explicit_required", "event_control"}
                and r.event in {None, event.event}
            )
        except BaseException:
            scope.close()
            raise
        probe = lambda: (
            self.current() and self.checkpoints.is_current(owner) and current()
        )
        self.context.bind(context_owner or owner, event, current=probe)
        token = self.checkpoints.begin(
            event,
            required,
            dependency_requirements=dependent,
            owner_id=owner,
            current=lambda _event: probe(),
            stage_context=self.context.accept if publish else lambda *_: None,
            retain_context=False,
            retirement_owner_id=context_owner or owner,
        )
        self._executions[token] = (event, scope, plans)
        return token

    async def _execute(self, token):
        event, scope, plans = self._executions.pop(token)
        try:
            outcomes = []
            for plan in plans:
                if plan.phase != "observe":
                    outcomes.append(
                        await self.engine.fire_handler_async(
                            scope, plan.handler_id, event
                        )
                    )
            if not plans:
                outcomes.append(await self.engine.fire_async(event))
            self.engine.notify_planned(scope, event)
            return self.engine._merge(outcomes)
        finally:
            scope.close()

    async def fire(
        self,
        event: HookEvent,
        owner: str,
        *,
        current: Callable[[], bool] = lambda: True,
        publish: bool = True,
    ) -> HookEventOutcome:
        token = self.begin(event, owner, current=current, publish=publish)
        try:
            result = await self._execute(token)
            self.checkpoints.accept(token, result)
            self.checkpoints.assert_next_input_allowed(owner)
            if (
                not self.current()
                or not current()
                or not result.allowed
                or result.outstanding_cleanup
            ):
                raise HookCheckpointError("controlling lifecycle hook failed")
            return result
        except BaseException:
            # accept() can already have settled the requirement.
            try:
                self.checkpoints.fail(token, "lifecycle event failed")
            except HookCheckpointError:
                pass
            raise

    def install(
        self,
        event: HookEvent,
        owner: str,
        *,
        current: Callable[[], bool] = lambda: True,
        context_owner: str | None = None,
    ) -> Future | None:
        """Install synchronously before publishing a durable completion."""
        if not self.checkpoints.is_current(owner):
            self.diagnostics["late_context"] += 1
            if event.event == "SubagentStop":
                self.engine.notify(event)
            return None
        token = self.begin(event, owner, current=current, context_owner=context_owner)

        async def complete():
            try:
                result = await self._execute(token)
                if not current() or not self.checkpoints.is_current(
                    context_owner or owner
                ):
                    self.diagnostics["late_context"] += 1
                self.checkpoints.accept(token, result)
                return result
            except BaseException:
                try:
                    self.checkpoints.fail(token, "lifecycle completion failed")
                except HookCheckpointError:
                    pass
                raise

        future = asyncio.run_coroutine_threadsafe(complete(), self.engine.loop)
        self._pending.add(future)

        def settled(done):
            self._pending.discard(done)
            if not done.cancelled():
                done.exception()  # Custody stays here even if the operation was cancelled.

        future.add_done_callback(settled)
        return future

    async def wait(
        self, owner: str, *, required_handler_ids: tuple[str, ...] | None = ()
    ) -> None:
        await asyncio.to_thread(
            self.checkpoints.wait,
            owner,
            required_handler_ids=required_handler_ids,
            should_cancel=lambda: not self.current(),
        )

    def close_scope(self, owner: str) -> None:
        if self.engine.mcp_executor is not None:
            self.engine.mcp_executor.retire_scope(owner)
        self.context.close(owner)
        self.checkpoints.retire_owner(owner)

    def claim_handoff(self, owner: str) -> tuple[dict, ...]:
        """Claim only at accepted-turn publication under the admission lock."""
        with self.checkpoints._condition:
            if self._handoff is None:
                return ()
            source, current = self._handoff
            self._handoff = None
            if not current() or not self.checkpoints.is_current(source):
                self.diagnostics["late_context"] += 1
                self.close_scope(source)
                return ()
            self.context.transfer(source, owner, current=current)
            self.close_scope(source)
            return self.context.blocks(owner, "model")

    def seal(self) -> None:
        if self._sealed:
            return
        self._sealed = True
        self.engine.begin_close()
        teardown_scope = None
        if self.live and self.engine.mcp_executor is not None:
            # Same owner/store, independent bounded scope; retain the exact
            # source readiness before closing ordinary scopes and dependencies.
            candidate = "teardown:" + uuid4().hex
            with self.checkpoints._condition:
                self.checkpoints.bind_owner(candidate)
                if self.engine.mcp_executor.capture_teardown(self, candidate):
                    teardown_scope = candidate
                else:
                    self.close_scope(candidate)
        self.checkpoints.close_owner(self.scope_id)
        self._reservations.clear()
        for _event, execution, _plans in self._executions.values():
            execution.close()
        self._executions.clear()
        self._handoff = None
        self.terminal_budgets.clear()
        self.terminal_budget_times.clear()
        self.inherited_budgets.clear()
        for owner in reversed(tuple(self.checkpoints._parents)):
            if owner != teardown_scope:
                self.close_scope(owner)
        if self.live:
            self.live = False
            self.engine.notify_teardown(
                self.event("SessionEnd", initiator="host_cleanup")
            )
        self.engine.begin_close()


class CompactionHooks:
    """Exact pending auxiliary candidate and commit-adjacent post requirement."""

    def __init__(
        self,
        lifecycle: HookSessionLifecycle,
        owner: str,
        *,
        reason: str,
        prepare: Callable,
        current: Callable[[], bool],
        memory_current: Callable,
    ) -> None:
        self.lifecycle = lifecycle
        self.owner = owner
        self.reason = reason
        self.prepare = prepare
        self.current = current
        self.memory_current = memory_current
        self.post_future = None
        self.memory = None

    async def before(
        self, candidate: tuple[Mapping[str, Any], ...], output_cap: int
    ) -> tuple[Mapping[str, Any], ...]:
        if not self.current():
            raise HookCheckpointError("compaction candidate stale")
        event = self.lifecycle.event(
            "PreCompact",
            data={"candidate": _json_projection(candidate), "reason": self.reason},
        )
        result = await self.lifecycle.fire(
            event,
            self.owner,
            current=self.current,
            publish=False,
        )
        # Candidate-only ledger: runtime blocks never enter the summary model.
        ledger = ContextLedger()
        ledger.bind(self.owner, event, current=self.current)
        ledger.accept(event, result)
        messages = tuple(candidate) + ledger.blocks(self.owner, "candidate")
        from ..agent_models import check_host_context

        check_host_context(messages, strip=False)
        prepared = self.prepare(messages, output_cap)
        if prepared.known_overflow or not self.current():
            raise HookCheckpointError("hook compaction input exceeds current capacity")
        return messages

    def committed(self, memory: Any) -> None:
        self.memory = memory
        current = lambda: self.memory_current(memory)
        owner = self.lifecycle.scope_id if self.reason == "manual" else self.owner
        self.post_future = self.lifecycle.install(
            self.lifecycle.event(
                "PostCompact",
                data={
                    "reason": self.reason,
                    "memory_id": memory.memory_id,
                    "summarized_prefix_digest": memory.summarized_prefix_digest,
                },
            ),
            owner,
            current=lambda: (
                self.lifecycle.checkpoints.is_current(self.owner) and current()
            ),
            context_owner=self.owner,
        )

    async def finish(self) -> None:
        if self.post_future is not None:
            result = await asyncio.wrap_future(self.post_future)
            if self.reason == "manual" and (
                not result.allowed or result.outstanding_cleanup
            ):
                # The session keeps compact failure evidence; this failed
                # operation has no handoff and must release its closures.
                self.lifecycle.close_scope(self.owner)
                return
        if (
            self.reason == "manual"
            and self.memory is not None
            and self.lifecycle.checkpoints.is_current(self.owner)
            and self.memory_current(self.memory)
        ):
            # A single pending handoff replaces, rather than queues behind, an old one.
            old = self.lifecycle._handoff
            if old is not None:
                self.lifecycle.close_scope(old[0])
            self.lifecycle._handoff = (
                self.owner,
                lambda: self.memory_current(self.memory),
            )
