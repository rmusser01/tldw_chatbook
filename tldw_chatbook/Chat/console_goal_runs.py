"""App-owned single native goal iteration; no repetition or completion inference."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

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
    GoalScriptEvidence,
    GoalScriptInvocation,
    GoalSnapshot,
    GoalToolObservation,
)
from tldw_chatbook.Agents.native_tools import provider_supports_native_tools
from tldw_chatbook.Agents.run_log import _setting
from tldw_chatbook.Chat.console_provider_endpoints import (
    normalize_generic_endpoint_for_compare,
)
from tldw_chatbook.Skills_Interop.skill_script_runner import ScriptRunResult

if TYPE_CHECKING:
    from tldw_chatbook.Agents.goal_run_service import GoalRunService
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
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
    """Dispatch one iteration using the runtime's shared audit and controller."""

    def __init__(self, controller: ConsoleChatController, service: GoalRunService):
        self.controller, self.service = controller, service
        self.ledger = service.db.automatic_work
        self._active = None
        self._dispatching = False
        self._stop_requested = False
        self._reserved_conversation_id: str | None = None
        self._owner_id = controller.fleet_wake._owner_id

    def owns_conversation(self, conversation_id: str | None) -> bool:
        """Protect the owning conversation even through a second hydrated UI alias."""
        return bool(
            conversation_id and conversation_id == self._reserved_conversation_id
        )

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
            or not provider_supports_native_tools(
                resolution.execution_key or resolution.provider
            )
        ):
            raise AutomaticWorkRefused("native_goal_required")

    async def accept(
        self, authorization: GoalIterationAuthorization, session_id: str
    ) -> bool:
        if not self.authorizes(authorization, session_id):
            raise PermissionError("goal authority is no longer live")
        authorization.check_binding()
        authorization.acceptance_started = True
        accepted = await asyncio.to_thread(
            self.ledger.accept_goal_iteration,
            authorization.attempt_id,
            owner_id=self._owner_id,
        )
        if not accepted:
            raise AutomaticWorkRefused("attempt_not_prepared")
        await asyncio.to_thread(authorization.context.mark_accepted)
        authorization.accepted = True
        authorization.started = time.monotonic()
        return True

    async def dispatch_once(self, goal_id: str) -> GoalIterationResult:
        """Cancellation requests Stop and waits for the owning cleanup path."""
        if self._dispatching:
            return GoalIterationResult(goal_id, 0, None, None, None, "goal_active")
        self._dispatching = True
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

    async def _dispatch_owned(self, goal_id: str) -> GoalIterationResult:
        from tldw_chatbook.Agents.agent_service import _coerce_autowake_enabled
        from tldw_chatbook.Chat.console_chat_models import (
            ConsoleRunState,
            ConsoleRunStatus,
            ConsoleSubmissionOrigin,
        )

        if self.controller._disposed:
            return GoalIterationResult(
                goal_id, 0, None, None, None, "runtime_unavailable"
            )
        goal = await asyncio.to_thread(self.service.get, goal_id)
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
            or len(busy) >= max(0, self.controller.max_parallel_runs - 1)
            or self.controller.prompt_queue_coordinator.controls_generation(session.id)
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

            handoff = build_goal_handoff(goal, goal.checkpoints)
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
