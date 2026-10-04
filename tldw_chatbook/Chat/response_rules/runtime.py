"""App-owned learning and the successful text-generation settlement boundary."""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from threading import RLock
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from loguru import logger

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunStatus
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest

from .builder import ResponseRuleBuilder
from .corrections import NativeCorrectionProposal
from .evaluator import (
    ResponseRuleEvaluator,
    _unverified,
    aggregate_checks,
    check_deterministic,
)
from .evidence import capture_rule_input
from .models import (
    MAX_HELPER_SECONDS,
    RuleAssessment,
    RuleBinding,
    RuleCandidate,
    RuleLearningResult,
    RuleInput,
    RuleRevision,
    RuleRuntimeState,
    RuleScope,
    RuleSource,
    digest_payload,
)
from .resources import RuleHelperPool
from .store import ResponseRuleStore

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_prompt_queue_coordinator import (
        ConsolePromptQueueCoordinator,
    )
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnConfigurationSnapshot


def checked_generation(
    method: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Check only real primary generation, after its existing cleanup finishes."""

    @functools.wraps(method)
    async def wrapped(controller, *args, **kwargs):
        service = getattr(controller, "response_rules", None)
        if service is None:
            return await method(controller, *args, **kwargs)
        message_id = kwargs["assistant_message_id"]
        boundary = service.begin_generation(
            message_id,
            kwargs["resolution"],
            configuration=getattr(kwargs.get("turn_context"), "configuration", None),
            direct_plain=True,
        )
        if boundary is None:
            return await method(controller, *args, **kwargs)
        session_id = boundary
        try:
            result = await method(controller, *args, **kwargs)
            message = controller.store.get_message(message_id)
            if message.status == "complete" and not message.generation_metadata:
                hook_owner = getattr(controller, "_hooks_v2_submissions", {}).get(
                    asyncio.current_task()
                )
                if hook_owner is not None and hook_owner[0] is not None:
                    await service.queue.settle_hook_parent(session_id, hook_owner[1])
                await service.finish_generation(session_id, message_id)
        except BaseException:
            service.discard_direct_repair(session_id)
            raise
        finally:
            service.release_generation(session_id, message_id)
        await service.finish_direct_repair(session_id)
        return result

    return wrapped


class ResponseRuleRuntime:
    """Retain domain state without retaining a Console view or granting tools."""

    def __init__(
        self,
        *,
        store: ResponseRuleStore,
        controller: ConsoleChatController | None,
        queue: ConsolePromptQueueCoordinator | None,
        builder: ResponseRuleBuilder,
        evaluator: ResponseRuleEvaluator,
        helpers: RuleHelperPool,
        chat_store: ConsoleChatStore | None = None,
        profile_current: Callable[[], bool] = lambda: True,
    ) -> None:
        self.store = store
        self._controller, self._queue = controller, queue
        if chat_store is None:
            if controller is None:
                raise ValueError("rule_store_owner_required")
            chat_store = controller.store
        self.chat_store = chat_store
        self._profile_current = profile_current
        self.builder, self.evaluator, self.helpers = builder, evaluator, helpers
        database = store.repository.db
        self.profile_id = digest_payload({"profile": str(database.db_path)})
        self._lock = RLock()
        self._epochs: dict[str, int] = {}
        self._scope_owners: dict[str, tuple[RuleScope, RuleScope | None, RuleScope]] = (
            {}
        )
        self._sources: dict[str, RuleSource] = {}
        self._pinned: dict[RuleSource, tuple[RuleRevision, ...]] = {}
        self._assessments: dict[str, RuleAssessment] = {}
        self._states: dict[str, RuleRuntimeState] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._boundaries: dict[str, tuple[str, Any, Any]] = {}
        self._direct_requests: dict[str, ConsoleTurnCustodyRequest] = {}
        self._direct_proposals: dict[str, NativeCorrectionProposal] = {}
        self._direct_budgets: dict[str, tuple[Any, float]] = {}
        self._closed = False
        self._committing: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._unsubscribe = store.add_invalidation_listener(self._binding_changed)
        self._unsubscribe_source = self.chat_store.subscribe_response_rule_invalidation(
            self.invalidate
        )
        if controller is not None:
            self.bind_controller(controller)

    @property
    def controller(self) -> ConsoleChatController:
        """Return the real Console owner only after execution is assembled."""
        if self._controller is None:
            raise RuntimeError("active_chat_required")
        return self._controller

    @property
    def queue(self) -> ConsolePromptQueueCoordinator:
        if self._queue is None:
            raise RuntimeError("active_chat_required")
        return self._queue

    def profile_current(self) -> bool:
        """Fence profile-only management independently of any active Chat."""
        return not self._closed and self._profile_current()

    def bind_controller(self, controller: ConsoleChatController) -> None:
        """Attach later Console execution to the same profile management service."""
        if controller.store is not self.chat_store:
            raise ValueError("rule_store_owner_changed")
        self._controller = controller
        self._queue = controller.prompt_queue_coordinator
        self.queue.bind_native_assessment_lookup(
            self.current_assessment,
            rules=self.pinned_rules,
            limit_reached=self.correction_limit_reached,
        )
        controller.response_rules = self

    def state(self, session_id: str) -> RuleRuntimeState:
        """Return retained status for a mounted or detached view."""
        return self._states.get(session_id, RuleRuntimeState("idle", None, None, ""))

    def correction_limit_reached(self, source: RuleSource) -> None:
        """Project only a still-current owner's exhausted correction chain."""
        if self.acceptance_current(source):
            self._project(source.session_id, phase="idle", reason="correction_limit")

    def scopes(self, session_id: str) -> tuple[RuleScope, RuleScope | None, RuleScope]:
        """Resolve local ownership through the actual selected Chat."""
        session = next(s for s in self.chat_store.sessions() if s.id == session_id)
        scopes = (
            RuleScope("chat", session.persisted_conversation_id or session.id),
            (
                RuleScope("workspace", session.workspace_id)
                if session.workspace_id
                else None
            ),
            RuleScope("global", self.profile_id),
        )
        with self._lock:
            self._scope_owners[session_id] = scopes
        self.store.register_context(*scopes)
        return scopes

    def effective_rules(self, session_id: str) -> tuple[RuleRevision, ...]:
        return self.store.effective_rules(*self.scopes(session_id))

    def acceptance_current(self, source: RuleSource) -> bool:
        """Pure cached epoch check, also safe inside worker-owned admission."""
        with self._lock:
            return not self._closed and self._sources.get(source.session_id) == source

    def pinned_rules(self, source: RuleSource) -> tuple[RuleRevision, ...]:
        with self._lock:
            return (
                self._pinned.get(source, ()) if self.acceptance_current(source) else ()
            )

    def current_assessment(
        self, source: RuleSource, assessment_id: str
    ) -> RuleAssessment | None:
        with self._lock:
            assessment = self._assessments.get(assessment_id)
            return (
                assessment
                if self.acceptance_current(source)
                and assessment is not None
                and assessment.source == source
                else None
            )

    def _project(self, session_id: str, **changes) -> None:
        self._states[session_id] = replace(self.state(session_id), **changes)
        callback = getattr(self._controller, "response_rules_changed", None)
        if callable(callback):
            try:
                callback(session_id)
            except Exception:  # noqa: BLE001 -- disposable projection is not custody.
                logger.warning("response_rule_projection_unavailable")

    def _binding_changed(self, scope: RuleScope, _rule_id: str) -> None:
        # Binding writes may execute off-loop; never inspect mutable UI owners.
        with self._lock:
            owners = tuple(
                owner for owner, scopes in self._scope_owners.items() if scope in scopes
            )
        for owner in owners:
            self.invalidate(owner, "rules_changed")

    def invalidate(self, session_id: str, reason: str) -> None:
        """Revoke acceptance before cancellation; retained physical work settles."""
        with self._lock:
            self._epochs[session_id] = self._epochs.get(session_id, 0) + 1
            source = self._sources.pop(session_id, None)
            if source is not None:
                self._pinned.pop(source, None)
                self._assessments = {
                    key: value
                    for key, value in self._assessments.items()
                    if value.source != source
                }
            task = self._tasks.get(session_id)
            committing = session_id in self._committing

        def cancel_and_project():
            self.helpers.cancel_session(session_id, reason)
            if task is not None and not committing and not task.done():
                task.cancel()
            if source is not None or task is not None or session_id in self._states:
                self._project(
                    session_id,
                    phase="idle",
                    assessment=None,
                    learning=None,
                    reason=reason,
                )

        try:
            on_loop = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            on_loop = False
        if on_loop:
            cancel_and_project()
        elif self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(cancel_and_project)
        else:
            cancel_and_project()

    def cancel(self, session_id: str, reason: str) -> None:
        self.invalidate(session_id, reason)

    def dispose(self) -> None:
        """Seal result authority before the app's bounded transport cleanup."""
        self._closed = True
        for session in self.chat_store.sessions():
            self.invalidate(session.id, "shutdown")
        self._unsubscribe()
        self._unsubscribe_source()

    def defer_terminal(self, session_id: str, run_state: Any) -> bool:
        boundary = self._boundaries.get(session_id)
        if boundary is None or run_state.status is not ConsoleRunStatus.COMPLETED:
            return False
        self._boundaries[session_id] = (boundary[0], boundary[1], run_state)
        return True

    def begin_generation(
        self,
        message_id: str,
        resolution: ConsoleProviderResolution,
        *,
        configuration: ConsoleTurnConfigurationSnapshot | None = None,
        direct_plain: bool = False,
    ) -> str | None:
        chats = self.controller.store
        session_id = chats.session_id_for_message(message_id)
        if self._closed:
            return None
        try:
            effective = self.effective_rules(session_id)
        except Exception:  # noqa: BLE001 -- rules cannot block requested work.
            logger.warning("response_rule_lookup_unavailable")
            self._project(
                session_id,
                phase="idle",
                assessment=None,
                learning=None,
                reason="rules_unavailable",
            )
            return None
        if not effective:
            return None
        self._loop = asyncio.get_running_loop()
        existing = self._boundaries.get(session_id)
        if existing is not None:
            if existing[0] == message_id:
                # The shared send owner delegates into the real agent owner.
                # Its outer boundary alone publishes completion and drains.
                return None
            raise RuntimeError("response_rule_generation_owner_conflict")
        self._boundaries[session_id] = (message_id, resolution, None)
        chats.hold_response_rule_completion(message_id)
        # Machine feedback is an extension of its parent task, not a new task root.
        chain = self.queue._chains.get(session_id)
        if chain is None and direct_plain and configuration is not None:
            self._direct_requests[session_id] = ConsoleTurnCustodyRequest(
                turn_id=str(uuid4()),
                session_id=session_id,
                draft="",
                configuration=configuration,
            )
        receipt = chain.machine_receipt if chain is not None else None
        if receipt is not None:
            parent = next(
                (
                    m
                    for m in chats.read_only_messages_for_session(session_id)
                    if receipt.parent_assistant_message_id
                    in (m.id, m.persisted_message_id)
                ),
                None,
            )
            if parent is not None:
                root = chats.response_rule_task_root(parent.id)
                if root is None:
                    messages = {
                        m.id: m
                        for m in chats.read_only_messages_for_session(session_id)
                    }
                    path = chats.active_path_message_ids(session_id)
                    prefix = (
                        path[: path.index(parent.id) + 1] if parent.id in path else ()
                    )
                    root = next(
                        (
                            node
                            for node in reversed(prefix)
                            if messages[node].role is ConsoleMessageRole.USER
                        ),
                        None,
                    )
                if root is not None:
                    chats.bind_response_rule_task_root(message_id, root)
        return session_id

    def _source(
        self,
        session_id: str,
        *,
        message_id: str | None = None,
        parent_turn_id: str | None = None,
    ) -> RuleSource:
        chats = self.controller.store
        session = next(s for s in chats.sessions() if s.id == session_id)
        path = chats.active_path_message_ids(session_id)
        messages = {m.id: m for m in chats.read_only_messages_for_session(session_id)}
        target = (
            messages[message_id]
            if message_id
            else next(
                m
                for node in reversed(path)
                if (m := messages[node]).role is ConsoleMessageRole.ASSISTANT
                and m.status == "complete"
                and not m.generation_metadata
            )
        )
        if (
            session.persisted_conversation_id is not None
            and target.persisted_message_id is None
        ):
            raise ValueError("rule_source_not_saved")
        pinned = self.effective_rules(session_id)
        chain = self.queue._chains.get(session_id)
        receipt = chain.machine_receipt if chain is not None else None
        parent_turn_id = parent_turn_id or (
            chain.request.turn_id
            if chain is not None and chain.request is not None
            else str(uuid4())
        )
        version = chats.response_rule_source_version(target.id)
        with self._lock:
            old = self._sources.get(session_id)
            if old is not None:
                self._pinned.pop(old, None)
                self._assessments = {
                    key: value
                    for key, value in self._assessments.items()
                    if value.source != old
                }
            source = RuleSource(
                self.profile_id,
                session_id,
                session.persisted_conversation_id,
                path[-1],
                target.persisted_message_id or target.id,
                version,
                parent_turn_id,
                receipt.operation_id if receipt is not None else str(uuid4()),
                str(uuid4()),
                digest_payload(
                    [
                        (r.rule_id, r.revision, r.candidate.candidate_digest())
                        for r in pinned
                    ]
                ),
                self._epochs.get(session_id, 0),
            )
            self._sources[session_id] = source
            self._pinned[source] = pinned
        capture_rule_input(chats, source)
        return source

    async def _owned(self, session_id: str, work):
        self._loop = asyncio.get_running_loop()
        task = asyncio.create_task(work)
        self._tasks[session_id] = task
        try:
            return await task
        finally:
            if self._tasks.get(session_id) is task:
                self._tasks.pop(session_id, None)

    async def finish_generation(self, session_id: str, message_id: str) -> None:
        try:
            direct = self._direct_requests.get(session_id)
            source = self._source(
                session_id,
                message_id=message_id,
                parent_turn_id=direct.turn_id if direct is not None else None,
            )
            assessment = await self._owned(session_id, self.assess_completed(source))
            if assessment is not None and assessment.outcome == "violation":
                proposal = NativeCorrectionProposal(source, assessment.assessment_id)
                if (
                    not self.queue.offer_native_correction(session_id, proposal)
                    and direct is not None
                ):
                    self._direct_proposals[session_id] = proposal
        except asyncio.CancelledError:
            owner = asyncio.current_task()
            if owner is not None and owner.cancelling():
                raise
        except (KeyError, ValueError, StopIteration):
            self._project(session_id, phase="idle", reason="source_changed")
        except Exception:  # noqa: BLE001 -- a rule failure cannot discard saved work.
            logger.warning("response_rule_assessment_unavailable")
            self._project(session_id, phase="idle", reason="assessment_unavailable")

    def release_generation(self, session_id: str, message_id: str) -> None:
        boundary = self._boundaries.pop(session_id, None)
        self.controller.store.release_response_rule_completion(message_id)
        if boundary is not None and boundary[2] is not None:
            self.controller._set_run_state(boundary[2], session_id=session_id)

    async def finish_direct_repair(self, session_id: str) -> None:
        """Drain direct plain-text repairs only after releasing their parent."""
        request = self._direct_requests.pop(session_id, None)
        proposal = self._direct_proposals.pop(session_id, None)
        budget = self._direct_budgets.pop(session_id, None)
        if (
            request is not None
            and proposal is not None
            and self.acceptance_current(proposal.source)
        ):
            await self.queue.start_native_repair(
                request, proposal, parent_budget=budget
            )

    def record_generation_budget(
        self, message_id: str, turn_id: str | None, budget: Any, recorded_at: float
    ) -> None:
        """Accept only the owning primary's actual settled budget on the app loop."""
        session_id = self.controller.store.session_id_for_message(message_id)
        if turn_id is not None:
            self.queue.record_native_budget(session_id, turn_id, budget, recorded_at)
        elif (
            session_id in self._direct_requests
            and self._boundaries.get(session_id, (None,))[0] == message_id
        ):
            self._direct_budgets[session_id] = (budget, recorded_at)

    def discard_direct_repair(self, session_id: str) -> None:
        self._direct_requests.pop(session_id, None)
        self._direct_proposals.pop(session_id, None)
        self._direct_budgets.pop(session_id, None)

    async def assess_completed(self, source: RuleSource) -> RuleAssessment | None:
        if not self.acceptance_current(source):
            return None
        inputs = capture_rule_input(self.controller.store, source)
        rules = self.pinned_rules(source)
        resolution = self._boundaries[source.session_id][1]
        self._project(source.session_id, phase="checking", reason="")
        semantic = any(check_deterministic(rule, inputs) is None for rule in rules)
        waiting = self.queue.registry.snapshot(source.session_id).waiting_count
        deadline = self.queue.native_helper_deadline(
            source.session_id, source.parent_turn_id
        )
        direct_budget = self._direct_budgets.get(source.session_id)
        if direct_budget is not None:
            remaining, recorded_at = direct_budget
            if remaining is False or remaining.max_total_tokens > 0:
                deadline = None
            elif deadline is not None:
                deadline = min(deadline, recorded_at + remaining.max_wall_seconds)
                if deadline <= time.monotonic():
                    deadline = None
        lease = (
            self.helpers.try_acquire(source, purpose="checking", deadline=deadline)
            if semantic and not waiting and deadline is not None
            else None
        )
        try:
            assessment = await self.evaluator.assess(
                source, inputs, rules, resolution=resolution, lease=lease
            )
        except asyncio.CancelledError:
            if lease is not None:
                lease.cancel_acceptance("cancelled")
                lease.release_unused()
            incomplete = replace(
                aggregate_checks(
                    source,
                    tuple(_unverified(rule, "cancelled") for rule in rules),
                    inputs=inputs,
                ),
                state="cancelled",
            )
            try:
                await asyncio.to_thread(self.store.save_assessment, incomplete)
            except (
                Exception
            ):  # noqa: BLE001 -- history is best effort after revocation.
                logger.warning("response_rule_cancelled_history_unavailable")
            self._project(source.session_id, phase="idle", assessment=incomplete)
            raise
        if not self.acceptance_current(source):
            return None
        capture_rule_input(self.controller.store, source)
        await asyncio.to_thread(
            self.store.save_assessment,
            assessment,
            current=lambda: self.acceptance_current(source),
            admission_lock=self._lock,
        )
        with self._lock:
            if not self.acceptance_current(source):
                return None
            self._assessments[assessment.assessment_id] = assessment
        self._project(source.session_id, phase="idle", assessment=assessment, reason="")
        return assessment

    async def learn(self, session_id: str, complaint: str) -> RuleLearningResult:
        def failed(reason, state="inactive"):
            return RuleLearningResult(state, None, None, {}, reason)

        if (
            self._closed
            or self.state(session_id).phase != "idle"
            or self.controller.run_state_for(session_id).is_stop_allowed
            or session_id in self._boundaries
        ):
            return failed("active_run")
        try:
            source = self._source(session_id)
            inputs = capture_rule_input(self.controller.store, source)
        except (KeyError, ValueError, StopIteration):
            return failed("no_eligible_response")
        self._project(session_id, phase="drafting", reason="")
        scope = self.scopes(session_id)[0]

        async def work():
            resolution = await self.controller.provider_gateway.resolve_for_send(
                self.controller._provider_selection_for_session(session_id)
            )
            return await self.builder.learn(
                source,
                complaint,
                inputs,
                resolution=resolution,
                current=lambda: self.acceptance_current(source),
                progress=lambda phase: (
                    self._project(session_id, phase=phase)
                    if self.acceptance_current(source)
                    else None
                ),
            )

        try:
            result = await self._owned(session_id, work())
            if not self.acceptance_current(source):
                result = replace(result, state="stale", reason="source_changed")
            else:
                await asyncio.to_thread(
                    self.store.save_draft,
                    scope,
                    source,
                    result,
                    complaint=complaint,
                    current=lambda: self.acceptance_current(source),
                    admission_lock=self._lock,
                )
                if (
                    result.rule is not None
                    and result.validation is not None
                    and result.reason == "tested"
                ):
                    self._committing.add(session_id)
                    try:
                        await asyncio.to_thread(
                            self.store.activate,
                            result.rule,
                            result.validation,
                            scope,
                            expected_binding_revision=0,
                            current=lambda: self.acceptance_current(source),
                            admission_lock=self._lock,
                        )
                        result = replace(result, state="active")
                    finally:
                        self._committing.discard(session_id)
            self._project(
                session_id, phase="idle", learning=result, reason=result.reason
            )
            if result.state == "active":
                try:
                    await self._initial_repair(session_id, result, inputs)
                except asyncio.CancelledError:
                    result = replace(result, reason="initial_repair_cancelled")
                    self._project(
                        session_id, phase="idle", learning=result, reason=result.reason
                    )
                    owner = asyncio.current_task()
                    if owner is not None and owner.cancelling():
                        raise
                except Exception:  # noqa: BLE001 -- activation already committed.
                    logger.warning("response_rule_initial_repair_unavailable")
                    result = replace(result, reason="initial_repair_unavailable")
                    self._project(
                        session_id, phase="idle", learning=result, reason=result.reason
                    )
            return result
        except asyncio.CancelledError:
            owner = asyncio.current_task()
            if owner is not None and owner.cancelling():
                raise
            result = failed(self.state(session_id).reason or "cancelled", "cancelled")
            self._project(session_id, phase="idle", learning=result)
            return result
        except ValueError as exc:
            result = failed(
                "too_many_effective_rules"
                if str(exc) == "too_many_effective_rules"
                else "save_or_learning_unavailable"
            )
            self._project(
                session_id, phase="idle", learning=result, reason=result.reason
            )
            return result
        except Exception:  # noqa: BLE001 -- leave the original answer recoverable.
            logger.warning("response_rule_learning_unavailable")
            result = failed("save_or_learning_unavailable")
            self._project(
                session_id, phase="idle", learning=result, reason=result.reason
            )
            return result

    async def _initial_repair(
        self, session_id: str, result: RuleLearningResult, inputs: RuleInput
    ) -> None:
        if result.rule is None or result.validation is None:
            return
        request = ConsoleTurnCustodyRequest(
            turn_id=str(uuid4()),
            session_id=session_id,
            draft="",
            configuration=self.controller.resolve_turn_configuration_snapshot(
                session_id
            ),
        )
        source = self._source(session_id, parent_turn_id=request.turn_id)
        current_input = capture_rule_input(self.controller.store, source)
        if (
            current_input != inputs
            or result.validation.original_input_digest != inputs.evidence_digest()
        ):
            return
        checks = []
        for rule in self.pinned_rules(source):
            if (rule.rule_id, rule.revision) == (
                result.rule.rule_id,
                result.rule.revision,
            ):
                checks.append(
                    next(
                        case.check
                        for case in result.validation.case_results
                        if case.case_type == "recorded_violation"
                    )
                )
            else:
                checks.append(
                    check_deterministic(rule, inputs)
                    or _unverified(rule, "not_assessed")
                )
        assessment = aggregate_checks(source, tuple(checks), inputs=inputs)
        await asyncio.to_thread(
            self.store.save_assessment,
            assessment,
            current=lambda: self.acceptance_current(source),
            admission_lock=self._lock,
        )
        self._assessments[assessment.assessment_id] = assessment
        self._project(session_id, phase="repairing", assessment=assessment)
        await self.queue.start_native_repair(
            request, NativeCorrectionProposal(source, assessment.assessment_id)
        )
        self._project(session_id, phase="idle")

    async def activate_tested(
        self,
        result: RuleLearningResult,
        scope: RuleScope,
        *,
        expected_binding_revision: int,
    ) -> RuleBinding:
        """Commit a reviewed editor result under the same live epoch lock."""
        if (
            result.reason not in {"tested", "validation_reused"}
            or result.rule is None
            or result.validation is None
        ):
            raise ValueError("validated_editor_result_required")
        source = result.validation.source
        if scope not in self.scopes(source.session_id):
            raise ValueError("rule_scope_changed")
        return await asyncio.to_thread(
            self.store.activate,
            result.rule,
            result.validation,
            scope,
            expected_binding_revision=expected_binding_revision,
            current=lambda: self.acceptance_current(source),
            admission_lock=self._lock,
        )

    async def test_edit(
        self,
        session_id: str,
        scope: RuleScope,
        rule_id: str,
        revision: int,
        candidate: RuleCandidate,
        *,
        example_message_id: str | None = None,
    ) -> RuleLearningResult:
        """Test a reviewed definition without activating or rewriting it."""
        if (
            self.state(session_id).phase != "idle"
            or self.controller.run_state_for(session_id).is_stop_allowed
        ):
            return RuleLearningResult("inactive", None, None, {}, "active_run")
        rule = self.store.get_revision(rule_id, revision)
        prior = self.store.get_validation(rule_id, revision)
        try:
            native = example_message_id or next(
                m.id
                for m in self.controller.store.read_only_messages_for_session(
                    session_id
                )
                if rule.origin.message_id in (m.id, m.persisted_message_id)
            )
            source = self._source(session_id, message_id=native)
            inputs = capture_rule_input(self.controller.store, source)
        except (KeyError, ValueError, StopIteration):
            return RuleLearningResult(
                "inactive", None, None, {}, "original_evidence_unavailable"
            )
        old = next(
            (
                d
                for d in reversed(self.store.list_drafts(scope))
                if d.rule is not None
                and (d.rule.rule_id, d.rule.revision) == (rule_id, revision)
            ),
            None,
        )
        cases = dict(old.fixtures) if old is not None else None
        if example_message_id is not None:
            cases, prior = None, None
        if cases is not None and prior is not None:
            for case in prior.case_results:
                if case.case_type == "recorded_violation":
                    cases[case.case_id] = inputs
        self._project(session_id, phase="testing", reason="")
        try:
            resolution = await self.controller.provider_gateway.resolve_for_send(
                self.controller._provider_selection_for_session(session_id)
            )
            result = await self._owned(
                session_id,
                self.builder.validate_edit(
                    source,
                    candidate,
                    candidate.title,
                    inputs,
                    prior,
                    resolution=resolution,
                    current=lambda: self.acceptance_current(source),
                    cases=cases,
                ),
            )
            if result.rule is not None:
                next_revision = await asyncio.to_thread(
                    self.store.next_revision, rule_id
                )
                updated = replace(result.rule, rule_id=rule_id, revision=next_revision)
                validation = (
                    replace(
                        result.validation,
                        case_results=tuple(
                            replace(
                                case,
                                check=replace(
                                    case.check, rule_id=rule_id, revision=next_revision
                                ),
                            )
                            for case in result.validation.case_results
                        ),
                    )
                    if result.validation is not None
                    else None
                )
                result = replace(result, rule=updated, validation=validation)
                if self.acceptance_current(source):
                    await asyncio.to_thread(
                        self.store.save_draft,
                        scope,
                        source,
                        result,
                        complaint="",
                        current=lambda: self.acceptance_current(source),
                        admission_lock=self._lock,
                    )
            self._project(
                session_id, phase="idle", learning=result, reason=result.reason
            )
            return result
        except asyncio.CancelledError:
            owner = asyncio.current_task()
            if owner is not None and owner.cancelling():
                raise
            return RuleLearningResult(
                "cancelled", None, None, {}, self.state(session_id).reason
            )
        finally:
            if self.state(session_id).phase == "testing":
                self._project(session_id, phase="idle")
