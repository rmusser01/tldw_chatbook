"""App-owned bounded native goal continuation using service and ledger authority."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from loguru import logger

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    RunOutcome,
    RunTerminationReason,
)
from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused, GoalAttempt
from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkContext
from tldw_chatbook.Agents.goal_models import (
    GoalIterationResult,
    GoalProviderRef,
    GoalRequest,
    GoalScriptEvidence,
    GoalScriptInvocation,
    GoalSnapshot,
    GoalToolObservation,
)
from tldw_chatbook.Agents.run_log import _setting
from tldw_chatbook.Chat.console_provider_endpoints import (
    normalize_generic_endpoint_for_compare,
)
from tldw_chatbook.Skills_Interop.skill_script_runner import ScriptRunResult

if TYPE_CHECKING:
    from tldw_chatbook.Agents.goal_run_service import GoalRunService
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnExecutionContext


def goal_provider_ref(resolution: ConsoleProviderResolution) -> GoalProviderRef:
    """Pin resolved execution and credential identity without retaining credentials."""
    return GoalProviderRef(
        provider=resolution.provider,
        model=resolution.model,
        config_ref="api_settings."
        + (resolution.readiness_key or resolution.execution_key or resolution.provider),
        authority_ref=resolution.api_key_source or "unauthenticated",
        endpoint_ref=normalize_generic_endpoint_for_compare(resolution.base_url)
        or "provider-default",
    )


_KEY = object()


class GoalIterationAuthorization:
    """One coordinator-issued capability bound to exact durable authority."""

    def __init__(
        self,
        coordinator: ConsoleGoalCoordinator,
        goal: GoalSnapshot,
        attempt: GoalAttempt,
        *,
        _key: object,
    ) -> None:
        if _key is not _KEY:
            raise PermissionError("goal authority is coordinator-internal")
        from tldw_chatbook.Chat.console_chat_controller import (
            _capture_project_root_identity,
        )

        self._source_root_identities = tuple(
            (Path(b.locator), _capture_project_root_identity(Path(b.locator)))
            for b in goal.request.source_bindings
        )
        self._coordinator = coordinator
        self.goal = goal
        self.attempt = attempt
        self.session_id = attempt.session_id
        self.conversation_id = goal.conversation_id
        self.work_chain_id = goal.chain_id
        self.owner_id = attempt.owner_id
        self.attempt_id = attempt.id
        self.acceptance_started = False
        self.accepted = False
        self.preflight_refused = False
        self.context = AutomaticWorkContext(
            coordinator.ledger,
            goal.chain_id,
            attempt.owner_id,
            attempt.id,
            attempt_kind="goal_iteration",
            goal=self,
        )
        self.native_run_id = None
        self.outcome = None
        self.records = []
        self.observations = []
        self.evidence_limit = coordinator.service.db.goal_runs.result_record_capacity(
            attempt.id
        )
        self.started = None
        self._script_invocations = {}
        self.registry = None
        self.project_state = None

    def __repr__(self):
        return "GoalIterationAuthorization(authority=<redacted>)"

    def check_binding(self) -> None:
        from tldw_chatbook.Chat.console_chat_controller import (
            _project_root_identity_matches,
        )

        self._coordinator._check_admission_open()
        if any(
            identity is None or not _project_root_identity_matches(root, identity)
            for root, identity in self._source_root_identities
        ):
            raise AutomaticWorkRefused("binding_changed")
        reason = self._coordinator.service._binding_reason(self.goal.request)
        if reason:
            raise AutomaticWorkRefused(reason)
        if not self._coordinator.authorizes(self, self.session_id):
            raise AutomaticWorkRefused("goal_authority_expired")
        if self.project_state is not None:
            session = next(
                (
                    s
                    for s in self._coordinator.controller.store.sessions()
                    if s.id == self.session_id
                ),
                None,
            )
            if (
                session is None
                or session.workspace_id != self.goal.request.binding.workspace_id
            ):
                raise AutomaticWorkRefused("binding_changed")
            state = session.project_instruction_state
            if (
                state.project_instructions_enabled
                != self.project_state.project_instructions_enabled
                or state.working_folder_binding_id
                != self.project_state.working_folder_binding_id
                or state.working_folder_locator_fingerprint
                != self.project_state.working_folder_locator_fingerprint
            ):
                raise AutomaticWorkRefused("binding_changed")
        if self._coordinator._stop_requested:
            raise AutomaticWorkRefused("cancelled")
        if self.started is not None and self.remaining_seconds() <= 0:
            raise AutomaticWorkRefused("iteration_wall_budget")

    def remaining_seconds(self) -> float:
        snapshot = self.context.ledger.snapshot(self.work_chain_id)
        iteration = self.goal.policy.iteration_wall_seconds
        if self.started is not None:
            iteration -= time.monotonic() - self.started
        chain = (
            snapshot.deadline_at - time.time()
            if snapshot.deadline_at
            else self.goal.policy.wall_seconds
        )
        return max(0.0, min(iteration, chain))

    def narrow_config(
        self, config: AgentConfig, registry: ToolCatalogRegistry
    ) -> AgentConfig:
        self.registry = registry
        scope = self.goal.request.tool_scope
        mcp_ids = {binding.tool_id for binding in scope.mcp_bindings}
        allowed = tuple(
            entry.name
            for entry in registry.list_catalog()
            if entry.id in scope.catalog_tools
            and (entry.source != "mcp" or entry.id in mcp_ids)
        )
        policy = self.goal.policy
        return replace(
            config,
            allowed_tools=allowed,
            budget=replace(
                config.budget,
                max_subagents=0,
                max_steps=min(config.budget.max_steps, policy.iteration_steps),
                max_model_turns=min(
                    config.budget.max_model_turns, policy.iteration_model_turns
                ),
                max_wall_seconds=min(
                    config.budget.max_wall_seconds, self.remaining_seconds()
                ),
            ),
        )

    def permits_runtime(self, name: str) -> bool:
        scope = self.goal.request.tool_scope.runtime_tools
        return name != "spawn_subagent" and (
            name in scope or "runtime:" + name in scope
        )

    def permits_call(self, name: str) -> bool:
        if self.permits_runtime(name):
            return True
        return (
            self.registry is not None
            and self.registry.resolve_name(name)
            in self.goal.request.tool_scope.catalog_tools
        )

    def begin_script(
        self, path: str | Path, args: Sequence[str], skill_name: str, trust_digest: str
    ) -> GoalScriptInvocation:
        """Validate the actual resolved execution; extendable pre-execution seam."""
        from tldw_chatbook.Agents.run_context import current_run_id

        self.context.check()
        if len(self.records) + len(self.observations) >= self.evidence_limit:
            raise AutomaticWorkRefused("goal_evidence_capacity")
        if not self.permits_runtime("run_skill_script"):
            raise AutomaticWorkRefused("goal_tool_scope")
        from tldw_chatbook.Agents.goal_iteration import (
            capture_manifest,
            verifier_digest,
        )

        path = str(Path(path).resolve())
        digest = verifier_digest(path)
        try:
            selected = self.goal.request.select_script_verifier(
                path, digest, tuple(args), trust_digest
            )
        except ValueError:
            raise AutomaticWorkRefused("ambiguous_verifier_invocation") from None
        if selected is None:
            raise AutomaticWorkRefused("verifier_binding_changed")
        run_id = current_run_id()
        if not run_id:
            raise AutomaticWorkRefused("goal_run_missing")
        invocation = GoalScriptInvocation(
            uuid4().hex,
            self.goal.id,
            self.attempt_id,
            self.attempt.ordinal,
            run_id,
            path,
            digest,
            skill_name,
            trust_digest,
            tuple(args),
            capture_manifest(
                self.goal.request.binding.locator,
                selected,
                seconds=self.remaining_seconds(),
            ),
            verifier_id=selected.id,
        )
        self._script_invocations[invocation.id] = invocation
        return invocation

    def finish_script(
        self, invocation: GoalScriptInvocation, result: ScriptRunResult
    ) -> None:
        """Capture typed process results even when a deadline passed during cleanup."""
        from tldw_chatbook.Agents.run_context import current_run_id

        if (
            self._script_invocations.get(invocation.id) is not invocation
            or invocation.run_id != current_run_id()
            or not isinstance(result, ScriptRunResult)
        ):
            raise ValueError("stale goal evidence callback")
        attempt = self.context.ledger.read_goal_attempt(
            self.attempt_id, owner_id=self.owner_id
        )
        if attempt.state != "accepted":
            raise ValueError("goal evidence attempt no longer accepted")
        from tldw_chatbook.Agents.goal_iteration import capture_manifest

        spec = self.goal.request.select_script_verifier(
            invocation.verifier_path,
            invocation.verifier_sha256,
            invocation.arguments,
            invocation.skill_trust_ref,
            verifier_id=invocation.verifier_id,
        )
        if spec is None:
            raise ValueError("stale goal verifier callback")
        manifest = capture_manifest(
            self.goal.request.binding.locator, spec, seconds=self.remaining_seconds()
        )
        self.records.append(GoalScriptEvidence(invocation, result, manifest))
        del self._script_invocations[invocation.id]

    def observe_tool_result(self, tool: str, arguments: dict, content: str) -> str:
        """Own bounded generic observations, with no objective verification verdict."""
        import json

        from tldw_chatbook.Agents.goal_iteration import digest
        from tldw_chatbook.Agents.run_context import current_run_id

        if (
            tool == "run_skill_script"
            or len(self.records) + len(self.observations) >= self.evidence_limit
        ):
            return content
        run_id = current_run_id()
        if not run_id or not self.permits_call(tool):
            return content
        encoded = content.encode("utf-8")
        retained = encoded[: 48 * 1024].decode("utf-8", "ignore")
        item = GoalToolObservation(
            uuid4().hex,
            self.goal.id,
            self.attempt_id,
            run_id,
            tool,
            digest(json.dumps(arguments, sort_keys=True, default=str)),
            retained,
            len(encoded) <= 48 * 1024,
        )
        self.observations.append(item)
        return f"goal_evidence_id: {item.id}\n" + content

    def record_outcome(self, run_id: str, outcome: RunOutcome) -> None:
        if self.native_run_id is not None and self.native_run_id != run_id:
            raise ValueError("goal native run identity conflict")
        self.native_run_id, self.outcome = run_id, outcome
        with self.context.ledger.transaction() as conn:
            row = conn.execute(
                "SELECT conversation_id, work_chain_id FROM agent_runs WHERE id=?",
                (run_id,),
            ).fetchone()
            if (
                row is None
                or row["conversation_id"] != self.conversation_id
                or row["work_chain_id"] != self.work_chain_id
            ):
                raise ValueError("goal native run ownership mismatch")
            conn.execute(
                "UPDATE goal_iterations SET run_id=?, revision=revision+1 WHERE attempt_id=? AND (run_id IS NULL OR run_id=?)",
                (run_id, self.attempt_id, run_id),
            )


class ConsoleGoalCoordinator:
    """Schedule settled iterations using the runtime’s shared audit and controller."""

    def __init__(self, controller: ConsoleChatController, service: GoalRunService):
        self.controller, self.service = controller, service
        self.ledger = service.db.automatic_work
        self._active = None
        self._dispatching = False
        self._stop_requested = False
        self._reserved_conversation_id: str | None = None
        self._owner_id = controller.fleet_wake._owner_id
        self._runner = None
        self._running_goal = None
        self._closed = False
        self._projected_audit = None
        self._wake = asyncio.Event()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._dispatch_goal_id: str | None = None
        self._observers: set[Callable[[str], None]] = set()
        self._launch_task: asyncio.Task[GoalSnapshot] | None = None
        self._launch_id: str | None = None
        self._resume_goal_id: str | None = None
        self.service._control_listener = self._control_changed

    def subscribe(self, callback: Callable[[str], None]) -> Callable[[], None]:
        """Attach a body-free view observer on the runtime loop; return its detach."""
        self._event_loop = asyncio.get_running_loop()
        self._observers.add(callback)
        return lambda: self._observers.discard(callback)

    def notify_goal_changed(self, goal_id: str) -> None:
        """Publish identity only, never selected private reports or evidence."""
        loop = self._event_loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._deliver_goal_changed, goal_id)

    def _deliver_goal_changed(self, goal_id: str) -> None:
        for callback in tuple(self._observers):
            try:
                callback(goal_id)
            except Exception:  # noqa: BLE001 - observer failure cannot change execution
                # View failure cannot alter admission or durable accounting.
                self._observers.discard(callback)

    @property
    def primary_reserved(self) -> bool:
        return self._reserved_conversation_id is not None

    @property
    def active_goal_id(self) -> str | None:
        """Return the goal owning active execution, approval or a bounded wait."""
        return self._running_goal or self._dispatch_goal_id or self._resume_goal_id

    async def recover(self) -> tuple[GoalSnapshot, ...]:
        """Observe/project the one shared startup audit without admitting work."""
        if not await self.controller.fleet_wake.wait_for_recovery():
            raise RuntimeError("history_unavailable")
        audit = self.controller.fleet_wake.recovery_result
        if audit is None:
            return ()
        result = await asyncio.to_thread(self.service.project_recovery, audit)
        self._projected_audit = audit
        return result

    async def restore_session(
        self, goal_id: str, *, app: Any, settings: ConsoleSessionSettings | None = None
    ) -> ConsoleChatSession:
        """Use normal persisted-tree hydration with stable identity and no navigation."""
        from dataclasses import replace

        from tldw_chatbook.Chat.console_conversation_hydration import (
            apply_resume_settings_overrides,
            hydrate_console_session,
            load_console_conversation_tree,
        )
        from tldw_chatbook.Chat.console_session_settings import (
            default_console_session_settings,
        )

        goal = self.service.get(goal_id)
        if goal.request is None:
            raise ValueError("goal_payload_removed")
        existing = next(
            (
                s
                for s in self.controller.store.sessions()
                if s.id == goal.conversation_id
            ),
            None,
        )
        if existing is not None:
            if existing.persisted_conversation_id != goal.conversation_id:
                raise ValueError("conversation_identity_conflict")
            return existing
        tree = await load_console_conversation_tree(app, goal.conversation_id)
        if tree is None:
            raise ValueError("goal_conversation_missing")
        settings = settings or default_console_session_settings(
            getattr(app, "app_config", {})
        )
        settings = replace(
            settings,
            provider=goal.request.provider.provider,
            model=goal.request.provider.model,
        )
        settings = apply_resume_settings_overrides(settings, tree["conversation"])
        return hydrate_console_session(
            app=app,
            store=self.controller.store,
            conversation_id=goal.conversation_id,
            tree=tree,
            settings=settings,
            target_workspace_id=goal.request.binding.workspace_id,
            session_id=goal.conversation_id,
            activate=False,
        )

    def _control_changed(self, snapshot):
        """The service may run on a DB thread; controller mutation stays on its loop."""
        loop = self._event_loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._apply_control, snapshot)

    def _apply_control(self, snapshot):
        self.notify_goal_changed(snapshot.id)
        self._wake.set()
        if snapshot.id == self.active_goal_id and snapshot.status in {
            "stopping",
            "stopped",
        }:
            self._stop_requested = True
            self._signal_active_stop()

    def _signal_active_stop(self) -> None:
        if self._active is not None:
            self.controller._signal_stop(session_id=self._active.session_id)

    def launch(
        self, request: GoalRequest, launch_id: str, *, app: Any
    ) -> asyncio.Task[GoalSnapshot]:
        """Own one bounded provisioning/hydration task independently of a view."""
        if self._closed or self.controller._disposed:
            raise RuntimeError("runtime_unavailable")
        if self._launch_task is not None and not self._launch_task.done():
            if self._resume_goal_id is None and self._launch_id == launch_id:
                return self._launch_task
            raise ValueError("goal_setup_active")
        if self.active_goal_id is not None:
            raise ValueError("goal_active")
        self._launch_id = launch_id
        self._launch_task = asyncio.create_task(self._launch(request, launch_id, app))
        return self._launch_task

    async def _launch(
        self, request: GoalRequest, launch_id: str, app: Any
    ) -> GoalSnapshot:
        goal = await asyncio.to_thread(
            self.service.create, request, launch_id=launch_id
        )
        if (
            not self._closed
            and not self.controller._disposed
            and goal.status == "ready"
        ):
            await self.restore_session(goal.id, app=app)
            if not self._closed and not self.controller._disposed:
                self.start(goal.id)
        self.notify_goal_changed(goal.id)
        return goal

    def resume(
        self, goal_id: str, *, expected_revision: int, app: Any
    ) -> asyncio.Task[GoalSnapshot]:
        """Hydrate and resume in the one runtime-owned activation slot."""
        if self._closed or self.controller._disposed:
            raise RuntimeError("runtime_unavailable")
        if self._launch_task is not None and not self._launch_task.done():
            if self._resume_goal_id == goal_id:
                return self._launch_task
            raise ValueError("goal_setup_active")
        if self.active_goal_id is not None:
            raise ValueError("goal_active")
        self._event_loop = asyncio.get_running_loop()
        self._launch_id = None
        self._resume_goal_id = goal_id
        self._launch_task = asyncio.create_task(
            self._resume(goal_id, expected_revision=expected_revision, app=app)
        )
        self.notify_goal_changed(goal_id)
        return self._launch_task

    async def _resume(
        self, goal_id: str, *, expected_revision: int, app: Any
    ) -> GoalSnapshot:
        try:
            goal = await asyncio.to_thread(self.service.get, goal_id)
            if goal.revision != expected_revision:
                raise ValueError("revision_conflict")
            if goal.status not in {"paused", "ready"}:
                raise ValueError("resume_requires_clean_checkpoint")
            if self._closed or self.controller._disposed:
                return await asyncio.to_thread(self.service.get, goal_id)
            hydration = asyncio.create_task(self.restore_session(goal_id, app=app))
            try:
                await asyncio.shield(hydration)
            except asyncio.CancelledError:
                # A cancelled view never reaches here; explicit runtime/hydration
                # cancellation still drains the existing read before stores close.
                await asyncio.gather(hydration, return_exceptions=True)
                return await asyncio.to_thread(self.service.get, goal_id)
            if self._closed or self.controller._disposed:
                return await asyncio.to_thread(self.service.get, goal_id)
            if goal.status == "paused":
                goal = await asyncio.to_thread(
                    self.service.resume, goal_id, expected_revision=expected_revision
                )
            else:
                goal = await asyncio.to_thread(self.service.get, goal_id)
            if (
                not self._closed
                and not self.controller._disposed
                and goal.status == "ready"
            ):
                self.start(goal_id)
            return goal
        finally:
            self._resume_goal_id = None
            self.notify_goal_changed(goal_id)

    def start(self, goal_id: str) -> asyncio.Task[GoalSnapshot]:
        """Start one runtime-owned chain; await the task to observe its next rest state."""
        if self._closed or self.controller._disposed:
            raise RuntimeError("runtime_unavailable")
        if (
            self._launch_task is not None
            and not self._launch_task.done()
            and asyncio.current_task() is not self._launch_task
        ):
            raise ValueError("goal_setup_active")
        if self._runner is not None and not self._runner.done():
            if self._running_goal != goal_id:
                raise ValueError("goal_active")
            return self._runner
        self._event_loop = asyncio.get_running_loop()
        self._running_goal = goal_id
        self._runner = asyncio.create_task(self._continue(goal_id))
        self.notify_goal_changed(goal_id)
        return self._runner

    async def _continue(self, goal_id: str):
        try:
            while not self._closed:
                # Clear before the awaited snapshot: a control notification arriving
                # during that read must remain observable even if the snapshot is old.
                self._wake.clear()
                goal = await asyncio.to_thread(self.service.get, goal_id)
                if goal.status != "ready":
                    return goal
                if goal.retry_at is not None:
                    try:
                        await asyncio.wait_for(
                            self._wake.wait(), max(0.01, goal.retry_at - time.time())
                        )
                    except TimeoutError:
                        pass
                    if self._closed or self.service.get(goal_id).status != "ready":
                        continue
                    # Capacity notifications do not bypass a provider backoff.
                    if (
                        goal.retry_reason == "pre_effect_rate_limit"
                        and time.time() < goal.retry_at
                    ):
                        continue
                    self.service.clear_wait(goal_id)
                result = await self.dispatch_once(goal_id)
                if result.reason_code == "primary_capacity":
                    self.service.defer(goal_id, reason="primary_capacity", delay=1)
                    continue
                if result.attempt_id:
                    attempt = await asyncio.to_thread(
                        self.ledger.read_goal_attempt,
                        result.attempt_id,
                        owner_id=self._owner_id,
                    )
                    if attempt.state in {"accepted", "review_required"}:
                        try:
                            await asyncio.to_thread(self.service.checkpoint, result)
                        except Exception:  # noqa: BLE001 - never replay to replace a lost checkpoint
                            return await asyncio.to_thread(
                                self.service.checkpoint_failed, goal_id
                            )
                    elif result.reason_code:
                        self.service._state(
                            self.service.get(goal_id), "paused", result.reason_code
                        )
                else:
                    current = self.service.get(goal_id)
                    if current.status == "ready":
                        self.service._state(
                            current,
                            "paused",
                            result.reason_code or "goal_preflight_refused",
                        )
                goal = await asyncio.to_thread(
                    self.service.db.goal_runs.settle_control, goal_id
                )
                self.notify_goal_changed(goal_id)
                if goal.status != "ready":
                    return goal
                if (
                    result.termination_reason
                    == RunTerminationReason.PRE_EFFECT_RATE_LIMIT
                ):
                    self.service.defer(
                        goal_id,
                        reason="pre_effect_rate_limit",
                        delay=min(30, 2 ** (goal.iteration_count - 1)),
                    )
                elif (
                    result.termination_reason == RunTerminationReason.PREFLIGHT_REFUSED
                ):
                    return self.service.pause(goal_id)
            return self.service.get(goal_id)
        finally:
            self._running_goal = None
            self.notify_goal_changed(goal_id)

    def notify_capacity(self) -> None:
        """Wake the single runtime waiter; provider retry time remains authoritative."""
        loop = self._event_loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._wake.set)

    def close_admission(self) -> None:
        """Synchronously fence continuation before controller shutdown drains workers."""
        self._closed = True
        self._stop_requested = True
        self.notify_capacity()
        loop = self._event_loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._signal_active_stop)
        # Durable Stop is best-effort at teardown. Neither the in-memory fence
        # nor physical draining may depend on database availability.
        if self.active_goal_id is not None:
            try:
                self.service.stop(self.active_goal_id)
            except Exception:  # noqa: BLE001 - shutdown must still drain its owner
                logger.warning("Goal Stop persistence failed during shutdown.")

    async def shutdown(self) -> None:
        self.close_admission()
        if self._launch_task is not None:
            await asyncio.shield(
                asyncio.gather(self._launch_task, return_exceptions=True)
            )
        if self._runner is not None:
            await asyncio.shield(self._runner)

    def manual_send(self, conversation_id: str | None) -> None:
        """An explicit manual instruction pauses only this conversation’s successors."""
        with self.service.db.connection() as conn:
            row = conn.execute(
                "SELECT id FROM goal_runs WHERE conversation_id=? AND status IN ('ready','pause_requested')",
                (conversation_id,),
            ).fetchone()
        if row:
            self.service.pause(row[0])

    def _manual_priority(self, conversation_id: str) -> bool:
        probe = getattr(self.controller, "wake_user_priority_probe", None)
        return any(
            self.controller.prompt_queue_coordinator.controls_generation(session.id)
            or (callable(probe) and probe(session.id))
            for session in self.controller.store.sessions()
            if session.persisted_conversation_id == conversation_id
        )

    def owns_conversation(self, conversation_id: str | None) -> bool:
        """Protect the owning conversation even through a second hydrated UI alias."""
        return bool(
            conversation_id and conversation_id == self._reserved_conversation_id
        )

    def _check_admission_open(self) -> None:
        """Read the monotonic runtime fence, including from the ledger worker."""
        if self._closed or self.controller._disposed:
            raise AutomaticWorkRefused("runtime_unavailable")
        if self._stop_requested:
            raise AutomaticWorkRefused("cancelled")

    def authorizes(self, authorization: object, session_id: str | None) -> bool:
        return (
            isinstance(authorization, GoalIterationAuthorization)
            and authorization is self._active
            and authorization._coordinator is self
            and authorization.session_id == session_id
            and authorization.owner_id == self._owner_id
        )

    def validate_resolution(
        self,
        authorization: GoalIterationAuthorization,
        resolution: ConsoleProviderResolution,
        turn_context: ConsoleTurnExecutionContext,
    ) -> None:
        authorization.check_binding()
        if goal_provider_ref(resolution) != authorization.goal.request.provider:
            raise AutomaticWorkRefused("provider_binding_changed")
        if (
            self.controller._agent_bridge is None
            or not turn_context.tool_configuration.get(
                "agent_runtime_enabled", self.controller._agent_runtime_enabled
            )
            or not turn_context.tool_configuration.get(
                "native_tool_calls_enabled", True
            )
        ):
            raise AutomaticWorkRefused("native_goal_required")

    async def accept(
        self, authorization: GoalIterationAuthorization, session_id: str
    ) -> bool:
        self._check_admission_open()
        if not self.authorizes(authorization, session_id):
            raise PermissionError("goal authority is no longer live")
        authorization.check_binding()
        if self._manual_priority(authorization.goal.conversation_id):
            raise AutomaticWorkRefused("primary_capacity")
        authorization.acceptance_started = True
        accepted = await asyncio.to_thread(
            self.ledger.accept_goal_iteration,
            authorization.attempt_id,
            owner_id=self._owner_id,
            admission_guard=self._check_admission_open,
        )
        if not accepted:
            raise AutomaticWorkRefused("attempt_not_prepared")
        await asyncio.to_thread(authorization.context.mark_accepted)
        authorization.accepted = True
        authorization.started = time.monotonic()
        self._check_admission_open()
        return True

    async def dispatch_once(self, goal_id: str) -> GoalIterationResult:
        """Cancellation requests Stop and waits for the owning cleanup path."""
        if self._closed or self.controller._disposed:
            return GoalIterationResult(
                goal_id, 0, None, None, None, "runtime_unavailable"
            )
        if self._dispatching or (
            self._launch_task is not None and not self._launch_task.done()
        ):
            return GoalIterationResult(goal_id, 0, None, None, None, "goal_active")
        self._event_loop = asyncio.get_running_loop()
        self._dispatching = True
        self._dispatch_goal_id = goal_id
        self._stop_requested = False
        task = asyncio.create_task(self._dispatch_owned(goal_id))
        try:
            while True:
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    if task.done():
                        return task.result()
                    self._stop_requested = True
                    if self._active is not None:
                        self.controller._signal_stop(session_id=self._active.session_id)
        finally:
            self._dispatching = False
            self._dispatch_goal_id = None

    async def _dispatch_owned(self, goal_id: str) -> GoalIterationResult:
        from tldw_chatbook.Agents.agent_service import _coerce_autowake_enabled
        from tldw_chatbook.Chat.console_chat_models import (
            ConsoleRunState,
            ConsoleRunStatus,
            ConsoleSubmissionOrigin,
        )

        if self._closed or self.controller._disposed:
            return GoalIterationResult(
                goal_id, 0, None, None, None, "runtime_unavailable"
            )
        goal = await asyncio.to_thread(self.service.get, goal_id)
        if self._closed or self.controller._disposed:
            return GoalIterationResult(
                goal_id, 0, None, None, None, "runtime_unavailable"
            )
        if goal.status != "ready" or goal.request is None:
            return GoalIterationResult(goal_id, 0, None, None, None, "goal_not_ready")
        if not _coerce_autowake_enabled(_setting("goal_runs_enabled", False)):
            return GoalIterationResult(
                goal_id, 0, None, None, None, "goal_runs_disabled"
            )
        if self._active is not None:
            return GoalIterationResult(goal_id, 0, None, None, None, "goal_active")
        if not await self.controller.fleet_wake.wait_for_recovery():
            return GoalIterationResult(
                goal_id, 0, None, None, None, "history_unavailable"
            )
        audit = self.controller.fleet_wake.recovery_result
        if audit is not None and audit is not self._projected_audit:
            await asyncio.to_thread(self.service.project_recovery, audit)
            self._projected_audit = audit
            goal = await asyncio.to_thread(self.service.get, goal_id)
            if goal.status != "ready":
                return GoalIterationResult(
                    goal_id, 0, None, None, None, "goal_not_ready"
                )
        if self._closed or self.controller._disposed:
            return GoalIterationResult(
                goal_id, 0, None, None, None, "runtime_unavailable"
            )
        session = next(
            (
                s
                for s in self.controller.store.sessions()
                if s.id == goal.conversation_id
                and s.persisted_conversation_id == goal.conversation_id
            ),
            None,
        )
        if session is None:
            return GoalIterationResult(
                goal_id, 0, None, None, None, "goal_session_missing"
            )
        if session.assistant_kind == "character":
            return GoalIterationResult(
                goal_id, 0, None, None, None, "native_goal_required"
            )
        if session.workspace_id != goal.request.binding.workspace_id:
            return GoalIterationResult(goal_id, 0, None, None, None, "binding_changed")
        busy = set(self.controller._live_busy_session_ids()) | set(
            self.controller.fleet_wake.delivering_session_ids()
        )
        if (
            any(
                s.id in busy and s.persisted_conversation_id == goal.conversation_id
                for s in self.controller.store.sessions()
            )
            or self.controller.fleet_wake.automatic_primary_count
            >= self.controller.fleet_wake.MAX_AUTOMATIC_PRIMARIES
            or len(busy) >= max(0, self.controller.max_parallel_runs - 1)
            or self._manual_priority(goal.conversation_id)
        ):
            return GoalIterationResult(goal_id, 0, None, None, None, "primary_capacity")
        self._reserved_conversation_id = goal.conversation_id
        # Occupancy is reserved synchronously before the first admission await.
        self.controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.VALIDATING, "Preparing goal iteration."),
            session_id=session.id,
        )
        attempt = None
        authorization = None
        reason = RunTerminationReason.PREFLIGHT_REFUSED
        reason_code = None
        try:
            attempt = await asyncio.to_thread(
                self.ledger.prepare_goal_iteration, goal_id, owner_id=self._owner_id
            )
            self._check_admission_open()
            authorization = GoalIterationAuthorization(self, goal, attempt, _key=_KEY)
            self._active = authorization
            from tldw_chatbook.Chat.console_project_instructions import (
                ProjectInstructionControlState,
                fingerprint_canonical_locator,
            )

            reference = goal.request.binding
            previous = session.project_instruction_state
            fingerprint = fingerprint_canonical_locator(reference.locator)
            state = ProjectInstructionControlState(
                project_instructions_enabled=previous.project_instructions_enabled,
                working_folder_binding_id=reference.binding_id,
                working_folder_locator_fingerprint=fingerprint,
                project_instruction_notice_key=(
                    previous.project_instruction_notice_key
                    if previous.working_folder_binding_id == reference.binding_id
                    and previous.working_folder_locator_fingerprint == fingerprint
                    else None
                ),
            )
            self.controller.store.set_session_project_instruction_state(
                session.id, state
            )
            authorization.project_state = state
            from tldw_chatbook.Agents.goal_iteration import build_goal_handoff

            invocations = []
            for verifier in goal.request.verifiers:
                skills = self.controller._agent_bridge._skills_service
                if skills is None:
                    raise AutomaticWorkRefused("goal_verifier_mapping_missing")
                from tldw_chatbook.runtime_policy.types import PolicyDeniedError
                from tldw_chatbook.Skills_Interop.skill_trust_models import (
                    SkillTrustBlockedError,
                )

                try:
                    invocations.append(await skills.goal_verifier_invocation(verifier))
                except (ValueError, SkillTrustBlockedError, PolicyDeniedError):
                    raise AutomaticWorkRefused("verifier_binding_changed") from None
                authorization.check_binding()
            try:
                handoff = build_goal_handoff(
                    goal, goal.checkpoints, verifier_invocations=invocations
                )
            except ValueError:
                raise AutomaticWorkRefused("goal_launch_context_capacity") from None
            with authorization.context.scope():
                await self.controller.submit_draft(
                    handoff,
                    session_id=session.id,
                    origin=ConsoleSubmissionOrigin.GOAL_ITERATION,
                    goal_authorization=authorization,
                )
            reason = (
                authorization.outcome.termination_reason or authorization.outcome.status
                if authorization.outcome
                else RunTerminationReason.UNKNOWN_EFFECT
                if authorization.accepted
                else RunTerminationReason.PREFLIGHT_REFUSED
            )
        except AutomaticWorkRefused as exc:
            reason = exc.termination_reason
            reason_code = exc.reason
        except asyncio.CancelledError:
            reason = "cancelled"
        except Exception:  # noqa: BLE001 - uncertain execution never authorizes replay
            reason = RunTerminationReason.UNKNOWN_EFFECT
        finally:
            if attempt is not None:
                try:
                    await asyncio.to_thread(
                        self.ledger.abort_goal_iteration,
                        attempt.id,
                        owner_id=self._owner_id,
                    )
                except Exception:  # noqa: BLE001 - uncertainty cannot release an owned executor
                    reason = RunTerminationReason.UNKNOWN_EFFECT
                    reason_code = "goal_cleanup_ledger_unavailable"
            capacity = getattr(self.controller._agent_bridge, "runtime_capacity", None)
            if capacity is not None:
                while any(
                    entry.conversation_id == goal.conversation_id
                    for entry in capacity.snapshot().executions
                ):
                    await asyncio.sleep(0.05)
            self._active = None
            self._reserved_conversation_id = None
            self.controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.IDLE, "Goal iteration settled."),
                session_id=session.id,
            )
        return GoalIterationResult(
            goal_id,
            attempt.ordinal if attempt else 0,
            attempt.id if attempt else None,
            authorization.native_run_id if authorization else None,
            authorization.outcome if authorization else None,
            reason,
            tuple(authorization.records) if authorization else (),
            reason_code,
            tuple(authorization.observations) if authorization else (),
        )
