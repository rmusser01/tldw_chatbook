"""Controller-side authority for sequential Console prompt queue drains."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from threading import Event
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleControllerActivity,
    ConsoleQueuedAcceptanceEvent,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
)
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchCheckpointState,
)
from tldw_chatbook.Chat.console_prompt_queue import (
    ConsolePromptQueueRegistry,
    PromptQueueMode,
    PromptQueueMutationResult,
    PromptQueuePauseReason,
    PromptQueueReservation,
    PromptQueueSnapshot,
    QueuedPrompt,
    QueueMutationStatus,
    QueueThreadViolation,
)
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest

if TYPE_CHECKING:
    from tldw_chatbook.Agents.agent_models import PluginContextText
    from tldw_chatbook.Agents.hooks_v2.continuations import (
        ContinuationAdmission,
        ContinuationReceipt,
    )
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
    from tldw_chatbook.Agents.hooks_v2.models import HookResult
    from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult
    from tldw_chatbook.Chat.console_received_turn import ConsoleReceivedTurnClaim
    from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryPolicySnapshot


_AUTHORIZATION_KEY = object()

# Shown when a press that would run the queue stops at a context review
# instead (TASK-33621.19). It names no cause and no time, because the change
# need not be the user's or happen during the pause: an edit, a delete, a
# compaction, or a failed regeneration the queue itself ran all move the
# context epoch.
CONTEXT_CHANGED_REVIEW_NOTICE = (
    "The conversation has changed. Review it before the queue continues."
)


class QueueGenerationAuthorization:
    """Opaque, coordinator-issued authority to cross a queue-owned send gate."""

    __slots__ = ("_coordinator", "entry_id", "session_id")

    def __init__(self, coordinator: object, session_id: str, *, _key: object) -> None:
        if _key is not _AUTHORIZATION_KEY:
            raise PermissionError("queue generation authority is coordinator-internal")
        self._coordinator = coordinator
        self.session_id = session_id
        chain = coordinator._chains.get(session_id)
        self.entry_id = chain.current_entry_id if chain else None

    def __repr__(self) -> str:
        return (
            "QueueGenerationAuthorization("
            f"session_id={self.session_id!r}, authority=<redacted>)"
        )


class QueuedTurnSubmitter(Protocol):
    """Production-shaped callback used to submit one claimed queue entry."""

    def __call__(
        self,
        prompt: QueuedPrompt,
        *,
        session_id: str,
        entry_id: str,
        authorization: QueueGenerationAuthorization,
    ) -> Awaitable["ConsoleSubmitResult"]: ...


@dataclass(slots=True)
class _PromptChain:
    accepted_live_turn: bool = False
    current_entry_id: str | None = None
    last_terminal_status: ConsoleRunStatus | None = None
    logical_outcome_id: str | None = None
    request: ConsoleTurnCustodyRequest | None = None
    hook_parent: object | None = None
    pending_stop_key: tuple[str, str] | None = None
    hook_cancel_event: Event | None = None
    pending_stop_task: asyncio.Task | None = None
    pending_stop_cancelled: bool = False
    continuation: object | None = None
    continuation_started: float | None = None
    initiator: str = "manual"
    rollback_epoch: tuple[int, int] | None = None
    machine_entry_id: str | None = None


@dataclass(frozen=True, slots=True)
class _RecoveredQueueOwner:
    """Body-free exact owner fencing later queued work."""

    queue_entry_id: str
    preparation_id: str
    checkpoint_state: ConsoleDispatchCheckpointState


class ConsolePromptQueueCoordinator:
    """Own queue admission, accepted claims, drain progression, and recovery."""

    _SUCCESS = frozenset({ConsoleRunStatus.COMPLETED})
    _TERMINAL = frozenset(
        {
            ConsoleRunStatus.BLOCKED,
            ConsoleRunStatus.COMPLETED,
            ConsoleRunStatus.FAILED,
            ConsoleRunStatus.STOPPED,
        }
    )

    def __init__(
        self,
        *,
        registry: ConsolePromptQueueRegistry,
        context_epoch: Callable[[str], int],
        run_status: Callable[[str], ConsoleRunStatus],
        submit_queued: QueuedTurnSubmitter,
        has_staged_rider: Callable[[str], bool] | None = None,
        needs_approval: Callable[[str], bool] | None = None,
        can_reacquire_slot: Callable[[str], bool] | None = None,
        on_queued_accepted: Callable[[ConsoleQueuedAcceptanceEvent], None]
        | None = None,
        on_activity_changed: Callable[[str], None] | None = None,
        on_chain_terminal: Callable[[str, ConsoleRunStatus, str | None], None]
        | None = None,
        received_turn_for_session: Callable[[str], ConsoleReceivedTurnClaim | None]
        | None = None,
    ) -> None:
        self.registry = registry
        self._context_epoch = context_epoch
        self._run_status = run_status
        self._submit_queued = submit_queued
        self._received_turn_for_session = received_turn_for_session
        self._has_staged_rider = has_staged_rider or (lambda _session_id: False)
        self._needs_approval = needs_approval or (lambda _session_id: False)
        self._can_reacquire_slot = can_reacquire_slot or (lambda _session_id: True)
        self.on_queued_accepted = on_queued_accepted
        self.on_activity_changed = on_activity_changed
        self.on_chain_terminal = on_chain_terminal
        self._chains: dict[str, _PromptChain] = {}
        self._queue_snapshots: dict[str, PromptQueueSnapshot] = {}
        self._dispatch_recoveries: dict[str, _RecoveredQueueOwner] = {}
        self._recovered_logical_outcomes: dict[str, str] = {}
        self._settled_dispatch_recoveries: OrderedDict[tuple[str, str, str], None] = (
            OrderedDict()
        )
        self._shutting_down = False
        self._maintenance_paused = False
        self._maintenance_suspended: dict[str, int] = {}
        self._continuation_keys: set[tuple[str, str]] = set()
        self._stop_parents = {}
        self._machine_entries = {}
        self._stop_outcomes = {}
        self._sealed_continuations: set[str] = set()
        self._continuation_admission_current = lambda _request: not self._maintenance_paused

    def maintenance_close_admission(self) -> None:
        """Stop admissions and next claims, leaving the accepted turn intact."""
        self._maintenance_paused = True

    def maintenance_resume(self) -> tuple[tuple[str, int], ...]:
        """Return only maintenance-suspended queues for guarded redispatch."""
        self._maintenance_paused = False
        return tuple(self._maintenance_suspended.items())

    def _maintenance_refusal(self, session_id: str) -> PromptQueueMutationResult:
        return PromptQueueMutationResult(
            QueueMutationStatus.INVALID, self.registry.snapshot(session_id),
            detail="Console generation is paused for backup maintenance.",
        )

    async def resume_after_maintenance(
        self, session_id: str, revision: int
    ) -> PromptQueueMutationResult:
        """Resume only an unchanged maintenance pause, never a user edit."""
        if self._maintenance_paused:
            return self._maintenance_refusal(session_id)
        if self._maintenance_suspended.get(session_id) != revision:
            return self._maintenance_refusal(session_id)
        self._maintenance_suspended.pop(session_id, None)
        if self.registry.snapshot(session_id).revision != revision:
            return self._maintenance_refusal(session_id)
        return await self.resume_and_drain(session_id)

    def bind_turn_request(
        self, request: ConsoleTurnCustodyRequest, *, origin: ConsoleSubmissionOrigin
    ) -> None:
        """Capture the exact runtime-owned inputs while its chain is active."""
        chain = self._chains.get(request.session_id)
        if chain is not None:
            chain.request = request
            chain.initiator = (
                "manual" if origin is ConsoleSubmissionOrigin.MANUAL else "scheduled"
            )

    def capture_hook_parent(
        self,
        session_id: str,
        lifecycle: HookSessionLifecycle,
        scope: str,
        assistant_id: str,
        *,
        accepted_policy: ConsoleLibraryPolicySnapshot | None = None,
    ) -> None:
        """Pin an accepted root before H4 retires its operation scope."""
        chain = self._chains.get(session_id)
        if chain is None or not chain.accepted_live_turn or chain.request is None:
            return
        request = chain.request
        if accepted_policy is not None:
            ceiling = request.configuration.library_policy_maximum
            if (
                ceiling is None
                or ceiling.source != "new_session"
                or ceiling.policy_revision is not None
                or accepted_policy.source != "durable"
                or accepted_policy.policy_revision != 1
                or replace(
                    accepted_policy,
                    source=ceiling.source,
                    policy_revision=ceiling.policy_revision,
                )
                != ceiling
            ):
                return
            request = replace(
                request,
                configuration=replace(
                    request.configuration, library_policy_maximum=accepted_policy
                ),
            )
        if chain.continuation_started is None:
            chain.continuation_started = time.monotonic()
        event = lifecycle.event(
            "Stop",
            turn_id=request.turn_id,
            initiator="continuation" if chain.continuation else chain.initiator,
        )
        chain.hook_parent = (lifecycle, scope, event, assistant_id, request)

    async def settle_hook_parent(self, session_id: str, scope: str) -> None:
        """Join exact required gates before the provisional scope retires."""
        chain = self._chains.get(session_id)
        parent = chain.hook_parent if chain else None
        if parent is None or parent[1] != scope:
            return
        try:
            await parent[0].wait(scope)
        except BaseException:
            chain.hook_parent = None
            raise

    async def _stop_then_enqueue(self, session_id: str) -> None:
        chain = self._chains.get(session_id)
        parent = chain.hook_parent if chain else None
        if parent is None:
            return
        chain.hook_parent = None
        lifecycle, _scope, event, _assistant_id, request = parent
        # Stop has no context effect: execute under the live host session,
        # never resurrect an H4 retired operation/run owner.
        key = (request.turn_id, event.event_id)
        self._stop_parents[key] = (session_id, parent)
        chain.pending_stop_key = key
        self._changed(session_id)
        try:
            if not lifecycle.live or not lifecycle.current():
                return
            stop_task = asyncio.create_task(lifecycle.engine.fire_async(event))
            chain.pending_stop_task = stop_task
            try:
                outcome = await stop_task
            except asyncio.CancelledError:
                owner = asyncio.current_task()
                if not (
                    chain.pending_stop_cancelled
                    and stop_task.cancelled()
                    and owner is not None
                    and not owner.cancelling()
                ):
                    raise
                return
            if (
                outcome.allowed
                and not outcome.outstanding_cleanup
                and lifecycle.engine.effects_current(event, outcome)
            ):
                self._stop_outcomes[key] = outcome
                await self.schedule_continuation(
                    request.turn_id,
                    event.event_id,
                    tuple(result for _, result in outcome.accepted),
                )
        finally:
            chain.pending_stop_key = None
            chain.hook_cancel_event = None
            chain.pending_stop_task = None
            chain.pending_stop_cancelled = False
            self._stop_parents.pop(key, None)
            self._stop_outcomes.pop(key, None)
            lifecycle.terminal_budgets.pop(request.turn_id, None)
            lifecycle.terminal_budget_times.pop(request.turn_id, None)
            self._changed(session_id)

    async def schedule_continuation(
        self, parent_turn_id: str, event_id: str, proposals: tuple[HookResult, ...]
    ) -> str | None:
        """Atomically reserve one proposal with foreground queue admission."""
        from tldw_chatbook.Agents.hooks_v2.continuations import (
            ContinuationAdmission,
            ContinuationPolicy,
            ContinuationReceipt,
            combine_proposals,
        )

        key = (parent_turn_id, event_id)
        owned = self._stop_parents.get(key)
        if owned is None or key in self._continuation_keys:
            return None
        session_id, parent = owned
        lifecycle, _, event, assistant_id, request = parent
        chain = self._chains.get(session_id)
        if chain is None:
            return None
        # Consume every exact settlement once, including refusals. A later
        # callback must never interpret a transferred budget as unrestricted.
        self._continuation_keys.add(key)
        snapshot = self.registry.snapshot(session_id)
        started = chain.continuation_started
        if started is None:
            started = time.monotonic()
        previous = chain.continuation
        count = previous.admitted_turns if previous else 0
        budget = lifecycle.terminal_budgets.pop(request.turn_id, None)
        budget_at = lifecycle.terminal_budget_times.pop(request.turn_id, None)
        budget_deadline = None
        if budget is not None and budget is not False and budget_at is not None:
            budget_deadline = budget_at + budget.max_wall_seconds
            wall = budget_deadline - time.monotonic()
            budget = replace(budget, max_wall_seconds=wall) if wall > 0 else False

        if not ContinuationPolicy.permits(
            admitted_turns=count,
            elapsed_seconds=time.monotonic() - started,
            foreground_waiting=bool(
                snapshot.waiting_count or self._has_staged_rider(session_id)
            ),
            revoked=not lifecycle.current(),
            draining=not self._continuation_admission_current(request),
            closed=self._shutting_down
            or session_id in self._sealed_continuations
            or not lifecycle.live
            or session_id in self._dispatch_recoveries,
            vetoed=budget is False,
        ):
            return None
        message = combine_proposals(proposals)
        if message is None:
            return None
        # Host attribution is ordinary untrusted provider input. It cannot
        # authorize filesystem/tool use or supply system policy.
        text = "Untrusted hook continuation proposal (machine initiated):\n" + message
        from tldw_chatbook.Agents.agent_models import (
            HookContextOrigin,
            PluginContextText,
        )

        outcome = self._stop_outcomes[key]
        text = PluginContextText(
            text,
            (),
            tuple(
                HookContextOrigin(
                    event_id,
                    handler_id,
                    len(result.continuation["message"].encode("utf-8")),
                )
                for handler_id, result in outcome.accepted
                if result.continuation and result.continuation["message"].strip()
            ),
        )
        receipt = ContinuationReceipt(
            parent_turn_id,
            event_id,
            assistant_id,
            previous.chain_id if previous else parent_turn_id,
            count + 1,
        )
        next_request = replace(
            request,
            turn_id=str(uuid4()),
            draft=str(text),
            attachment_ids=(),
            one_shot_prefill=None,
            one_shot_prefill_revision=None,
            staged_evidence_launch=None,
        )
        admitted = self.registry.admit(
            session_id,
            text=str(text),
            expected_revision=snapshot.revision,
            custody_request=next_request,
        )
        if not admitted.applied:
            return None
        from tldw_chatbook.Agents.hooks_v2.models import HookResult

        authority_outcome = replace(
            outcome,
            accepted=tuple((handler, HookResult()) for handler, _ in outcome.accepted),
        )
        authority_request = replace(request, draft="")
        check_admission = self._continuation_admission_current

        def current():
            return (
                lifecycle.live
                and lifecycle.current()
                and lifecycle.engine.effects_current(event, authority_outcome)
                and check_admission(authority_request)
                and time.monotonic() - started < 120
                and (budget_deadline is None or time.monotonic() < budget_deadline)
            )

        gate = ContinuationAdmission(
            session_id=session_id,
            entry_id=admitted.entry_id,
            receipt=receipt,
            current=current,
        )
        self._machine_entries[admitted.entry_id] = (
            receipt,
            lifecycle,
            event,
            authority_outcome,
            text,
            gate,
            next_request.turn_id,
        )
        chain.continuation = receipt
        chain.continuation_started = started
        if budget is not None:
            lifecycle.inherited_budgets[next_request.turn_id] = (
                budget,
                time.monotonic(),
            )
        self._changed(session_id)
        return next_request.turn_id

    def bind_hook_parent_cancellation(
        self, session_id: str, assistant_id: str, cancellation: Event
    ) -> None:
        """Retain only the accepted parent's genuine provider cancellation identity."""
        chain = self._chains.get(session_id)
        parent = chain.hook_parent if chain is not None else None
        if (
            chain is not None
            and chain.accepted_live_turn
            and chain.request is not None
            and parent is not None
            and parent[3] == assistant_id
            and parent[4].session_id == session_id
            and parent[4].turn_id == chain.request.turn_id
        ):
            chain.hook_cancel_event = cancellation

    def pending_stop_cancellation(self, session_id: str) -> Event | None:
        """Return the original Event only during its exact pending settlement."""
        chain = self._chains.get(session_id)
        if (
            chain is not None
            and chain.request is not None
            and chain.pending_stop_key is not None
            and chain.pending_stop_key[0] == chain.request.turn_id
            and chain.pending_stop_key in self._stop_parents
        ):
            return chain.hook_cancel_event
        return None

    def cancel_pending_stop(self, session_id: str, cancellation: Event) -> None:
        """Cancel only the joined Stop event task after sealing its original owner."""
        if (
            self.pending_stop_cancellation(session_id) is cancellation
            and cancellation.is_set()
            and session_id in self._sealed_continuations
        ):
            chain = self._chains[session_id]
            if (
                chain.pending_stop_task is not None
                and not chain.pending_stop_task.done()
            ):
                chain.pending_stop_cancelled = True
                chain.pending_stop_task.cancel()

    def interrupt_parent(
        self, session_id: str
    ) -> tuple[HookSessionLifecycle, str] | None:
        """Return the exact accepted live root or its pending Stop settlement."""
        chain = self._chains.get(session_id)
        if chain is None:
            return None
        parent = chain.hook_parent if chain.accepted_live_turn else None
        if parent is None and self.pending_stop_cancellation(session_id) is not None:
            parent = self._stop_parents[chain.pending_stop_key][1]
        return (parent[0], parent[4].turn_id) if parent is not None else None

    def continuation_receipt(
        self, session_id: str, entry_id: str | None
    ) -> ContinuationReceipt | None:
        """Expose only the currently claimed machine admission to persistence."""
        chain = self._chains.get(session_id)
        if chain is None or entry_id != chain.current_entry_id:
            return None
        issued = self._machine_entries.get(entry_id)
        return issued[0] if issued is not None else None

    def acknowledge_machine_rollback(
        self, session_id: str, entry_id: str | None, before: int, after: int
    ) -> None:
        """Retain only the epoch transition caused by this claim's local echo cleanup."""
        chain = self._chains.get(session_id)
        if (
            chain is not None
            and self.continuation_receipt(session_id, entry_id) is not None
            and self.registry.snapshot(session_id).expected_context_epoch == before
        ):
            chain.rollback_epoch = (before, after)

    def continuation_contribution(
        self, session_id: str, entry_id: str | None
    ) -> ContinuationAdmission | None:
        """Return the one gate for this exact coordinator-minted live claim."""
        if self.continuation_receipt(session_id, entry_id) is None:
            return None
        return self._machine_entries[entry_id][5]

    def _invalidate_machine_admission(self, session_id: str) -> None:
        chain = self._chains.get(session_id)
        if chain is not None:
            issued = self._machine_entries.get(chain.current_entry_id)
            if issued is not None:
                issued[5].invalidate()

    def continuation_input(
        self, session_id: str, entry_id: str | None, text: str
    ) -> PluginContextText | None:
        """Restore only host-issued, whole live input, never parsed metadata."""
        if self.continuation_receipt(session_id, entry_id) is None:
            return None
        source = self._machine_entries[entry_id][4]
        if str(text) != str(source):
            raise ValueError("hook_continuation_input_changed")
        return source

    def continuation_current(self, session_id: str, entry_id: str | None) -> bool:
        """Recheck machine admission immediately before normal acceptance."""
        issued = self._machine_entries.get(entry_id)
        if issued is None:
            return True
        _, lifecycle, event, outcome, _, _gate, _turn_id = issued
        snapshot = self.registry.snapshot(session_id)
        chain = self._chains.get(session_id)
        return bool(
            chain is not None
            and chain.current_entry_id == entry_id
            and not self._shutting_down
            and not snapshot.closing
            and session_id not in self._sealed_continuations
            and snapshot.waiting_count == 0
            and lifecycle.live
            and lifecycle.current()
            and chain.request is not None
            and self._continuation_admission_current(chain.request)
            and lifecycle.engine.effects_current(event, outcome)
            and chain.continuation_started is not None
            and time.monotonic() - chain.continuation_started < 120
        )

    def authorizes(
        self,
        authorization: QueueGenerationAuthorization | None,
        session_id: str,
    ) -> bool:
        """Return whether ``authorization`` is this coordinator's session token."""

        return bool(
            authorization is not None
            and authorization._coordinator is self
            and authorization.session_id == session_id
            and session_id in self._chains
            and not self._shutting_down
            and self.continuation_current(
                session_id, self._chains[session_id].current_entry_id
            )
        )

    def owns_machine_claim(
        self, authorization, session_id: str, entry_id: str | None
    ) -> bool:
        """Recognize a minted claim even after its permission has been revoked."""
        chain = self._chains.get(session_id)
        return bool(
            authorization is not None
            and authorization._coordinator is self
            and authorization.session_id == session_id
            and chain is not None
            and entry_id == authorization.entry_id == chain.current_entry_id
            and entry_id in self._machine_entries
        )

    def reuses_claimed_slot(
        self,
        authorization: QueueGenerationAuthorization | None,
        session_id: str,
        *,
        recovery: bool = False,
    ) -> bool:
        """Reuse held capacity only for a current claim or explicit recovery."""
        chain = self._chains.get(session_id)
        snapshot = self.registry.snapshot(session_id)
        return bool(
            self.authorizes(authorization, session_id)
            and chain.current_entry_id == authorization.entry_id
            and (
                (authorization.entry_id is not None and snapshot.claimed_count == 1)
                or (
                    recovery
                    and authorization.entry_id is None
                    and snapshot.claimed_count == 0
                )
            )
            and not chain.accepted_live_turn
            and snapshot.reservation is PromptQueueReservation.HELD
            and self._run_status(session_id) in self._TERMINAL
            and session_id not in self._dispatch_recoveries
        )

    def bind_continuation_admission(
        self, check: Callable[[ConsoleTurnCustodyRequest], bool]
    ) -> None:
        """Use the runtime's current plugin admission fences without new authority."""
        self._continuation_admission_current = (
            lambda request: not self._maintenance_paused and check(request)
        )

    def bind_runtime_submitter(self, submit_queued: QueuedTurnSubmitter) -> None:
        """Route future claimed entries through the app-owned runtime."""

        self._submit_queued = submit_queued

    def _terminal_status(
        self, session_id: str, result: "ConsoleSubmitResult"
    ) -> ConsoleRunStatus:
        return result.terminal_status or self._run_status(session_id)

    def activity(self, session_id: str) -> ConsoleControllerActivity:
        """Derive the sole fleet-visible activity projection for a session."""

        if not session_id:
            status = self._run_status(session_id)
            return ConsoleControllerActivity(
                session_id=session_id,
                occupies_slot=False,
                preparing_before_acceptance=False,
                accepted_live_turn=False,
                needs_approval=False,
                queued_count=0,
                queue_paused=False,
                terminal_notification_eligible=status in self._TERMINAL,
            )
        try:
            snapshot = self.registry.snapshot(session_id)
        except QueueThreadViolation:
            # Fleet-summary diagnostics can read from a worker thread. Queue
            # writes remain owner-thread confined; those writes publish this
            # immutable cache before any UI callback observes the revision.
            snapshot = self._queue_snapshots.get(session_id)
        if snapshot is None:
            queued_count = 0
            queue_paused = False
            reservation = PromptQueueReservation.RELEASED
        else:
            queued_count = snapshot.total_count
            queue_paused = snapshot.mode is PromptQueueMode.PAUSED
            reservation = snapshot.reservation
        chain = self._chains.get(session_id)
        status = self._run_status(session_id)
        accepted_live = bool(
            (chain and chain.accepted_live_turn)
            or session_id in self._dispatch_recoveries
        )
        # TASK-33620.4 review: "preparing before acceptance" promises that a
        # queue opens once this turn is accepted, which only a prompt-chain
        # turn ever is. A chainless run (regenerate / continue / agent wake /
        # manual summary) in the same status occupies the slot but never
        # opens a queue, so it is not "preparing".
        preparing = (
            chain is not None
            and status
            in {
                ConsoleRunStatus.VALIDATING,
                ConsoleRunStatus.RETRYING,
            }
            and not accepted_live
        )
        occupies_slot = reservation is PromptQueueReservation.HELD or status in {
            ConsoleRunStatus.VALIDATING,
            ConsoleRunStatus.STREAMING,
            ConsoleRunStatus.CHECKING_CITATIONS,
            ConsoleRunStatus.RETRYING,
        }
        received = (
            self._received_turn_for_session(session_id)
            if self._received_turn_for_session is not None
            else None
        )
        if received is not None:
            occupies_slot = True
            preparing = preparing or (
                not accepted_live
                and received.origin
                in {ConsoleSubmissionOrigin.MANUAL, ConsoleSubmissionOrigin.QUEUED}
            )
        return ConsoleControllerActivity(
            session_id=session_id,
            occupies_slot=occupies_slot,
            preparing_before_acceptance=preparing,
            accepted_live_turn=accepted_live,
            needs_approval=self._needs_approval(session_id),
            queued_count=queued_count,
            queue_paused=queue_paused,
            terminal_notification_eligible=(
                status in self._TERMINAL and not occupies_slot and not accepted_live
            ),
        )

    def controls_generation(self, session_id: str) -> bool:
        """Return whether older queue-owned work controls the next generation."""

        if not session_id:
            return False
        snapshot = self.registry.snapshot(session_id)
        return bool(
            snapshot.total_count > 0
            or snapshot.expected_context_epoch is not None
            or session_id in self._dispatch_recoveries
        )

    def hydrate_dispatch_recovery(
        self,
        session_id: str,
        *,
        queue_entry_id: str,
        preparation_id: str,
        checkpoint_state: ConsoleDispatchCheckpointState,
    ) -> bool:
        """Fence one already-accepted entry before any restored queue wake."""

        if (
            not session_id
            or not queue_entry_id
            or not preparation_id
            or not isinstance(checkpoint_state, ConsoleDispatchCheckpointState)
        ):
            return False
        owner = _RecoveredQueueOwner(
            queue_entry_id,
            preparation_id,
            checkpoint_state,
        )
        current = self._dispatch_recoveries.get(session_id)
        if current is not None and current != owner:
            return False
        self._dispatch_recoveries[session_id] = owner
        self._recovered_logical_outcomes[session_id] = f"queue-chain:{preparation_id}"
        snapshot = self.registry.snapshot(session_id)
        if snapshot.total_count and snapshot.mode is not PromptQueueMode.PAUSED:
            paused = self.registry.pause(
                session_id,
                reason=PromptQueuePauseReason.FAILED,
                expected_revision=snapshot.revision,
            )
            if paused.status not in {
                QueueMutationStatus.APPLIED,
                QueueMutationStatus.UNCHANGED,
                QueueMutationStatus.LOCKED,
            }:
                self._dispatch_recoveries.pop(session_id, None)
                return False
        self._changed(session_id)
        return True

    def dispatch_recovery_blocks_queue(self, session_id: str) -> bool:
        """Return whether a durable owner prevents automatic advancement."""

        return session_id in self._dispatch_recoveries

    def clear_dispatch_recovery(
        self,
        session_id: str,
        *,
        queue_entry_id: str,
        preparation_id: str,
    ) -> bool:
        """Clear only the exact hydrated queue owner."""

        current = self._dispatch_recoveries.get(session_id)
        if current is None or (
            current.queue_entry_id != queue_entry_id
            or current.preparation_id != preparation_id
        ):
            return False
        self._dispatch_recoveries.pop(session_id, None)
        self._changed(session_id)
        return True

    async def settle_dispatch_recovery_and_drain(
        self,
        session_id: str,
        *,
        queue_entry_id: str,
        preparation_id: str,
        terminal_status: ConsoleRunStatus,
    ) -> PromptQueueMutationResult:
        """Release one settled owner and advance later work at most once."""

        key = (session_id, queue_entry_id, preparation_id)
        snapshot = self.registry.snapshot(session_id)
        if key in self._settled_dispatch_recoveries:
            return PromptQueueMutationResult(QueueMutationStatus.UNCHANGED, snapshot)
        if not self.clear_dispatch_recovery(
            session_id,
            queue_entry_id=queue_entry_id,
            preparation_id=preparation_id,
        ):
            return PromptQueueMutationResult(
                QueueMutationStatus.INVALID,
                snapshot,
                detail="Dispatch recovery queue owner changed.",
            )
        self._settled_dispatch_recoveries[key] = None
        while len(self._settled_dispatch_recoveries) > 256:
            self._settled_dispatch_recoveries.popitem(last=False)
        snapshot = self.registry.snapshot(session_id)
        if snapshot.total_count == 0:
            return PromptQueueMutationResult(QueueMutationStatus.UNCHANGED, snapshot)
        resumed = self.resume(session_id)
        # Same guard as resume_and_drain: a context change while the response
        # was pending makes resume() re-pause as CONTEXT_CHANGED (APPLIED, no
        # chain). Stop at that review; its detail carries the notice for the
        # Retry or Discard press that settled the owner (TASK-33621.19).
        if not resumed.applied or resumed.snapshot.mode is PromptQueueMode.PAUSED:
            return resumed
        await self._drain_waiting(session_id, terminal_status)
        return resumed

    def pending_continuation_stop_available(self, session_id: str) -> bool:
        """Read exact unsealed pending ownership without changing admission."""
        chain = self._chains.get(session_id)
        return bool(
            chain is not None
            and not chain.accepted_live_turn
            and not self._shutting_down
            and session_id not in self._sealed_continuations
            and (
                chain.request is not None
                and chain.pending_stop_key is not None
                and chain.pending_stop_key[0] == chain.request.turn_id
                or chain.current_entry_id in self._machine_entries
            )
        )

    def stop_pending_continuation(self, session_id: str) -> bool:
        """Seal this turn's pending Stop proposal or claimed machine preparation."""
        if not self.pending_continuation_stop_available(session_id):
            return False
        self.pause_for_stop(session_id)
        return True

    def continuation_cancelled(self, session_id: str) -> bool:
        """Carry a precommit Stop into the accepted turn's normal cancellation owner."""
        chain = self._chains.get(session_id)
        return bool(
            chain is not None
            and chain.continuation is not None
            and session_id in self._sealed_continuations
        )

    def pause_for_stop(self, session_id: str) -> PromptQueueMutationResult:
        """Release a chain reservation as soon as Stop targets its live turn."""

        self._sealed_continuations.add(session_id)
        self._invalidate_machine_admission(session_id)
        snapshot = self.registry.snapshot(session_id)
        if snapshot.total_count == 0:
            self._changed(session_id)
            return PromptQueueMutationResult(QueueMutationStatus.UNCHANGED, snapshot)
        result = self.registry.pause(
            session_id,
            reason=PromptQueuePauseReason.STOPPED,
            expected_revision=snapshot.revision,
        )
        if result.status in {
            QueueMutationStatus.APPLIED,
            QueueMutationStatus.UNCHANGED,
        }:
            self._changed(session_id)
        return result

    def admit(
        self,
        session_id: str,
        *,
        text: str,
        expected_revision: int,
        custody_request: ConsoleTurnCustodyRequest | None = None,
    ) -> PromptQueueMutationResult:
        """Admit text only behind an accepted turn or an existing queue."""
        if self._maintenance_paused:
            return self._maintenance_refusal(session_id)

        if self._has_staged_rider(session_id):
            snapshot = self.registry.snapshot(session_id)
            return PromptQueueMutationResult(
                status=QueueMutationStatus.INVALID,
                snapshot=snapshot,
                detail="Remove attachments or staged evidence before queueing.",
            )
        result = self.registry.admit(
            session_id,
            text=text,
            expected_revision=expected_revision,
            custody_request=custody_request,
        )
        if result.applied:
            self._invalidate_machine_admission(session_id)
            self._changed(session_id)
        return result

    def request_pause_after_turn(
        self, session_id: str, *, expected_revision: int
    ) -> PromptQueueMutationResult:
        """Request a pause and refresh the immutable activity cache."""

        result = self.registry.request_pause_after_turn(
            session_id, expected_revision=expected_revision
        )
        if result.applied:
            self._changed(session_id)
        return result

    def keep_draining(
        self, session_id: str, *, expected_revision: int
    ) -> PromptQueueMutationResult:
        """Cancel pause-after-turn and refresh the activity cache."""
        if self._maintenance_paused:
            return self._maintenance_refusal(session_id)

        result = self.registry.keep_draining(
            session_id, expected_revision=expected_revision
        )
        if result.applied:
            self._changed(session_id)
        return result

    async def run_prompt_chain(
        self,
        session_id: str,
        initial_turn: Callable[[], Awaitable["ConsoleSubmitResult"]],
    ) -> "ConsoleSubmitResult":
        """Run one manual turn and sequentially drain accepted queued turns."""
        if self._maintenance_paused:
            from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult

            return ConsoleSubmitResult(
                False, False, "Console generation is paused for backup maintenance."
            )

        if session_id in self._chains:
            return await initial_turn()
        self._sealed_continuations.discard(session_id)
        self._chains[session_id] = _PromptChain()
        self._changed(session_id)
        try:
            result = await initial_turn()
            await self._after_turn(session_id, result)
            return result
        except BaseException:
            if session_id in self._chains:
                self._pause_after_exception(session_id)
            raise
        finally:
            chain = self._chains.get(session_id)
            if chain is not None and not chain.accepted_live_turn:
                snapshot = self.registry.snapshot(session_id)
                if snapshot.expected_context_epoch is None:
                    self._chains.pop(session_id, None)
                    self._changed(session_id)

    def turn_accepted(
        self,
        session_id: str,
        *,
        origin: ConsoleSubmissionOrigin,
        context_epoch: int,
        entry_id: str | None = None,
        preparation_id: str | None = None,
        defer_queued_settlement: bool = False,
    ) -> None:
        """Commit the accepted boundary and settle a queued claim exactly once."""

        if origin in {
            ConsoleSubmissionOrigin.AGENT_WAKE,
            ConsoleSubmissionOrigin.AGENT_CHAT_START,
        }:
            return
        chain = self._chains.get(session_id)
        if chain is None:
            if origin is ConsoleSubmissionOrigin.MANUAL:
                return
            if origin is ConsoleSubmissionOrigin.AGENT_WAKE and entry_id is None:
                # A wake was never queued, so there is no claim to settle and
                # no chain to require. It reaches here after `leave_console`
                # tombstoned this visit's chains; that method's own owner
                # ruling keeps an in-flight wake running headless, and raising
                # refused it instead -- re-creating the "only completes if you
                # stay" gap task-15860 closed.
                #
                # Deliberately scoped to AGENT_WAKE rather than "any
                # non-MANUAL origin": a QUEUED acceptance that arrives with no
                # chain AND no entry id is a real accounting bug, and must
                # keep failing loudly.
                return
            raise RuntimeError("accepted queued chain is unavailable")
        if origin is ConsoleSubmissionOrigin.MANUAL:
            snapshot = self.registry.snapshot(session_id)
            result = self.registry.begin_chain(
                session_id,
                context_epoch=context_epoch,
                expected_revision=snapshot.revision,
            )
            if result.status not in {
                QueueMutationStatus.APPLIED,
                QueueMutationStatus.UNCHANGED,
            }:
                raise RuntimeError("manual turn could not establish its queue chain")
        else:
            if entry_id is None or chain.current_entry_id != entry_id:
                raise RuntimeError("queued acceptance did not match the claimed entry")
            if not defer_queued_settlement:
                self._settle_queued_claim(session_id, entry_id)
        chain.accepted_live_turn = True
        if preparation_id:
            chain.logical_outcome_id = f"queue-chain:{preparation_id}"
        self._changed(session_id)

    def acknowledge_durable_acceptance(
        self,
        session_id: str,
        *,
        entry_id: str,
        preparation_id: str,
        context_epoch: int,
    ) -> bool:
        """Acknowledge one exact committed queue claim independently of a chain."""

        del context_epoch  # The committed checkpoint, not a live epoch, owns re-entry.
        chain = self._chains.get(session_id)
        if chain is not None and chain.current_entry_id not in {None, entry_id}:
            return False
        # TASK-33621.19: the ordinary durable post-commit path acknowledges
        # EVERY queued turn while its live chain is still draining. That
        # chain advances the queue after the turn ends, so only a detached
        # acknowledgement (no exact live owner) may pause later work.
        live_owner = chain is not None and chain.current_entry_id == entry_id
        result = self.registry.settle_durable_acceptance(
            session_id,
            entry_id=entry_id,
            preparation_id=preparation_id,
            live_chain_owns_claim=live_owner,
        )
        if result.status not in {
            QueueMutationStatus.APPLIED,
            QueueMutationStatus.UNCHANGED,
        }:
            return False
        if session_id in self._dispatch_recoveries:
            snapshot = self.registry.snapshot(session_id)
            if snapshot.total_count and snapshot.mode is not PromptQueueMode.PAUSED:
                paused = self.registry.pause(
                    session_id,
                    reason=PromptQueuePauseReason.FAILED,
                    expected_revision=snapshot.revision,
                )
                if paused.status not in {
                    QueueMutationStatus.APPLIED,
                    QueueMutationStatus.UNCHANGED,
                }:
                    return False
        if chain is not None:
            chain.accepted_live_turn = True
            chain.logical_outcome_id = f"queue-chain:{preparation_id}"
            # The live owner keeps ``current_entry_id`` exactly as the
            # ephemeral ``turn_accepted`` path does: its own post-turn step
            # (the drain loop, or ``finish_recovered_entry`` for a reclaimed
            # preparation) must still recognise this turn to advance or
            # finish the chain. Clearing it here left a reclaimed durable
            # entry's chain DRAINING/HELD with nothing to drive it.
            self._changed(session_id)
        if result.status is QueueMutationStatus.APPLIED:
            callback = self.on_queued_accepted
            if callback is not None:
                callback(ConsoleQueuedAcceptanceEvent(session_id, entry_id))
        return True

    def bind_claimed_preparation(
        self,
        session_id: str,
        *,
        entry_id: str,
        preparation_id: str,
    ) -> bool:
        """Bind the exact current claim to its precommit recovery owner."""

        chain = self._chains.get(session_id)
        if chain is None or chain.current_entry_id != entry_id:
            return False
        result = self.registry.bind_claimed_preparation(
            session_id,
            entry_id=entry_id,
            preparation_id=preparation_id,
        )
        return result.status in {
            QueueMutationStatus.APPLIED,
            QueueMutationStatus.UNCHANGED,
        }

    def retain_durable_acceptance(self, session_id: str) -> None:
        """Fence a committed queued claim from returning to pending.

        Task 14 uses this only after SQLite has committed the durable owner and
        a later publication effect failed.  Recovery presentation belongs to
        Task 15; this narrow fence merely keeps the queue boundary truthful.
        """

        chain = self._chains.get(session_id)
        if chain is None:
            return
        chain.accepted_live_turn = True
        self._changed(session_id)

    async def _after_turn(self, session_id: str, result: "ConsoleSubmitResult") -> None:
        chain = self._chains.get(session_id)
        if chain is None:
            return
        current_entry_id = chain.current_entry_id
        accepted = chain.accepted_live_turn
        chain.accepted_live_turn = False
        status = self._terminal_status(session_id, result)
        chain.last_terminal_status = status
        self._changed(session_id)

        if not accepted:
            if current_entry_id is not None:
                self._return_claim(
                    session_id,
                    current_entry_id,
                    PromptQueuePauseReason.DISPATCH_REFUSED,
                )
            return
        if current_entry_id is not None:
            snapshot = self.registry.snapshot(session_id)
            if snapshot.claimed_count:
                self._settle_queued_claim(session_id, current_entry_id)
        chain.current_entry_id = None
        if status not in self._SUCCESS:
            chain.hook_cancel_event = None
            reason = (
                PromptQueuePauseReason.STOPPED
                if status is ConsoleRunStatus.STOPPED
                else PromptQueuePauseReason.FAILED
            )
            self._pause_or_finish(session_id, reason, status)
            return

        await self._stop_then_enqueue(session_id)
        await self._drain_waiting(session_id, status)

    def _settle_queued_claim(self, session_id: str, entry_id: str) -> None:
        """Acknowledge one exact claimed entry and emit its acceptance event."""

        snapshot = self.registry.snapshot(session_id)
        settled = self.registry.settle_claim(
            session_id,
            entry_id=entry_id,
            expected_revision=snapshot.revision,
        )
        if not settled.applied:
            raise RuntimeError("queued acceptance could not settle its claim")
        callback = self.on_queued_accepted
        if callback is not None:
            callback(ConsoleQueuedAcceptanceEvent(session_id, entry_id))

    async def _drain_waiting(self, session_id: str, status: ConsoleRunStatus) -> None:
        """Claim and submit FIFO entries until the chain empties or pauses."""

        chain = self._chains[session_id]
        while not self._shutting_down:
            snapshot = self.registry.snapshot(session_id)
            if snapshot.mode is PromptQueueMode.PAUSE_AFTER_TURN:
                self.registry.pause(
                    session_id,
                    reason=PromptQueuePauseReason.MANUAL,
                    expected_revision=snapshot.revision,
                )
                self._finish_visible_terminal(session_id, status)
                return
            if self._maintenance_paused:
                if snapshot.waiting_count and snapshot.mode is PromptQueueMode.DRAINING:
                    paused = self.registry.pause(
                        session_id, reason=PromptQueuePauseReason.MANUAL,
                        expected_revision=snapshot.revision,
                    )
                    if paused.applied:
                        self._maintenance_suspended[session_id] = paused.snapshot.revision
                elif not snapshot.waiting_count:
                    self.registry.finalize_empty_chain(
                        session_id, expected_revision=snapshot.revision,
                    )
                self._finish_visible_terminal(session_id, status)
                return
            if self._context_epoch(session_id) != snapshot.expected_context_epoch:
                if snapshot.total_count:
                    self.registry.pause(
                        session_id,
                        reason=PromptQueuePauseReason.CONTEXT_CHANGED,
                        expected_revision=snapshot.revision,
                    )
                    self._finish_visible_terminal(session_id, status)
                    return
            if snapshot.waiting_count == 0:
                self.registry.finalize_empty_chain(
                    session_id,
                    expected_revision=snapshot.revision,
                )
                self._finish_visible_terminal(session_id, status)
                return

            claim_result = self.registry.claim_next(
                session_id,
                expected_revision=snapshot.revision,
            )
            if not claim_result.applied or claim_result.claim is None:
                self._pause_after_exception(session_id)
                return
            claim = claim_result.claim
            chain.current_entry_id = claim.prompt.entry_id
            chain.machine_entry_id = (
                claim.prompt.entry_id
                if claim.prompt.entry_id in self._machine_entries
                else None
            )
            if claim.prompt.entry_id not in self._machine_entries:
                chain.continuation = None
                chain.continuation_started = None
            self._changed(session_id)
            if self._has_staged_rider(session_id):
                self._return_claim(
                    session_id,
                    claim.prompt.entry_id,
                    PromptQueuePauseReason.DISPATCH_REFUSED,
                )
                return
            authorization = QueueGenerationAuthorization(
                self, session_id, _key=_AUTHORIZATION_KEY
            )
            try:
                queued_result = await self._submit_queued(
                    claim.prompt,
                    session_id=session_id,
                    entry_id=claim.prompt.entry_id,
                    authorization=authorization,
                )
            except BaseException:
                self._pause_after_exception(session_id)
                raise
            finally:
                chain.current_entry_id = None
            accepted = chain.accepted_live_turn
            chain.accepted_live_turn = False
            status = self._terminal_status(session_id, queued_result)
            chain.last_terminal_status = status
            self._changed(session_id)
            if not accepted:
                if claim.prompt.entry_id in self._machine_entries:
                    snapshot = self.registry.snapshot(session_id)
                    if snapshot.claimed_count:
                        self.registry.settle_claim(
                            session_id,
                            entry_id=claim.prompt.entry_id,
                            expected_revision=snapshot.revision,
                        )
                    rollback = chain.rollback_epoch
                    chain.rollback_epoch = None
                    snapshot = self.registry.snapshot(session_id)
                    if (
                        rollback is not None
                        and snapshot.expected_context_epoch == rollback[0]
                        and self._context_epoch(session_id) == rollback[1]
                    ):
                        self.registry.adopt_recovery_context_baseline(
                            session_id,
                            context_epoch=rollback[1],
                            expected_revision=snapshot.revision,
                        )
                    self._retire_machine_entry(claim.prompt.entry_id)
                    chain.continuation = None
                    chain.continuation_started = None
                    if session_id not in self._dispatch_recoveries:
                        continue
                self._return_claim(
                    session_id,
                    claim.prompt.entry_id,
                    PromptQueuePauseReason.DISPATCH_REFUSED,
                )
                return
            self._retire_machine_entry(claim.prompt.entry_id)
            if status not in self._SUCCESS:
                reason = (
                    PromptQueuePauseReason.STOPPED
                    if status is ConsoleRunStatus.STOPPED
                    else PromptQueuePauseReason.FAILED
                )
                self._pause_or_finish(session_id, reason, status)
                return
            await self._stop_then_enqueue(session_id)

    def _return_claim(
        self, session_id: str, entry_id: str, reason: PromptQueuePauseReason
    ) -> None:
        snapshot = self.registry.snapshot(session_id)
        if entry_id in self._machine_entries:
            if snapshot.claimed_count:
                self.registry.settle_claim(
                    session_id, entry_id=entry_id, expected_revision=snapshot.revision
                )
            self._retire_machine_entry(entry_id)
            self._pause_or_finish(session_id, reason, self._run_status(session_id))
            return
        result = self.registry.return_claim_to_head(
            session_id,
            entry_id=entry_id,
            reason=reason,
            expected_revision=snapshot.revision,
        )
        if result.status not in {
            QueueMutationStatus.APPLIED,
            QueueMutationStatus.CLOSING,
            QueueMutationStatus.SHUTTING_DOWN,
        }:
            raise RuntimeError("claimed queue entry could not be restored")
        self._finish_visible_terminal(session_id, self._run_status(session_id))

    def _pause_or_finish(
        self,
        session_id: str,
        reason: PromptQueuePauseReason,
        status: ConsoleRunStatus,
    ) -> None:
        snapshot = self.registry.snapshot(session_id)
        if snapshot.total_count:
            self.registry.pause(
                session_id,
                reason=reason,
                expected_revision=snapshot.revision,
            )
        else:
            self.registry.finalize_empty_chain(
                session_id,
                expected_revision=snapshot.revision,
            )
        self._finish_visible_terminal(session_id, status)

    def _pause_after_exception(self, session_id: str) -> None:
        if self._shutting_down:
            return
        chain = self._chains.get(session_id)
        snapshot = self.registry.snapshot(session_id)
        if snapshot.claimed_count and chain and chain.current_entry_id:
            self._return_claim(
                session_id,
                chain.current_entry_id,
                PromptQueuePauseReason.DISPATCH_REFUSED,
            )
            return
        if snapshot.total_count:
            self.registry.pause(
                session_id,
                reason=PromptQueuePauseReason.FAILED,
                expected_revision=snapshot.revision,
            )
        elif snapshot.expected_context_epoch is not None:
            self.registry.finalize_empty_chain(
                session_id,
                expected_revision=snapshot.revision,
            )
        self._finish_visible_terminal(session_id, ConsoleRunStatus.FAILED)

    def _retire_machine_entry(self, entry_id: str | None) -> None:
        issued = self._machine_entries.pop(entry_id, None)
        if issued is not None:
            issued[1].inherited_budgets.pop(issued[6], None)

    def _finish_visible_terminal(
        self, session_id: str, status: ConsoleRunStatus
    ) -> None:
        chain = self._chains.get(session_id)
        logical_outcome_id = (
            chain.logical_outcome_id
            if chain is not None
            else self._recovered_logical_outcomes.get(session_id)
        )
        if chain is not None:
            chain.accepted_live_turn = False
            self._retire_machine_entry(chain.machine_entry_id)
        self._chains.pop(session_id, None)
        self._recovered_logical_outcomes.pop(session_id, None)
        self._changed(session_id)
        callback = self.on_chain_terminal
        if callback is not None and status in self._TERMINAL:
            callback(session_id, status, logical_outcome_id)

    def resume(self, session_id: str) -> PromptQueueMutationResult:
        """Reacquire a slot and resume a manually/dispatch-paused queue.

        Args:
            session_id: Session whose paused queue should resume.

        Returns:
            The resume result, or a refusal. If the conversation context
            changed since the queue's baseline, the queue is re-paused as
            CONTEXT_CHANGED instead. That result is APPLIED (or UNCHANGED if
            it already was), its mode is still PAUSED, no chain is created,
            and ``detail`` carries ``CONTEXT_CHANGED_REVIEW_NOTICE`` for the
            Resume or Retry press that asked to run. A caller that drains
            must stop on a PAUSED result (TASK-33621.19).
        """
        if self._maintenance_paused:
            return self._maintenance_refusal(session_id)

        snapshot = self.registry.snapshot(session_id)
        if session_id in self._dispatch_recoveries:
            return PromptQueueMutationResult(
                QueueMutationStatus.INVALID,
                snapshot,
                detail="Finish or discard the pending response first.",
            )
        if snapshot.mode is not PromptQueueMode.PAUSED:
            return PromptQueueMutationResult(QueueMutationStatus.INVALID, snapshot)
        if self._context_epoch(session_id) != snapshot.expected_context_epoch:
            result = self.registry.pause(
                session_id,
                reason=PromptQueuePauseReason.CONTEXT_CHANGED,
                expected_revision=snapshot.revision,
            )
            self._changed(session_id)
            if result.status in {
                QueueMutationStatus.APPLIED,
                QueueMutationStatus.UNCHANGED,
            }:
                return replace(result, detail=CONTEXT_CHANGED_REVIEW_NOTICE)
            return result
        if not self._can_reacquire_slot(session_id):
            return PromptQueueMutationResult(
                QueueMutationStatus.INVALID,
                snapshot,
                detail="All agent slots are currently in use.",
            )
        reserved = self.registry.reserve(
            session_id, expected_revision=snapshot.revision
        )
        if not reserved.applied:
            return reserved
        resumed = self.registry.resume(
            session_id, expected_revision=reserved.snapshot.revision
        )
        if resumed.applied:
            self._chains[session_id] = _PromptChain(
                logical_outcome_id=self._recovered_logical_outcomes.get(session_id)
            )
            self._changed(session_id)
        return resumed

    async def resume_and_drain(self, session_id: str) -> PromptQueueMutationResult:
        """Reacquire one slot and dispatch the next waiting entry."""

        resumed = self.resume(session_id)
        # A changed context epoch makes resume() re-pause as CONTEXT_CHANGED,
        # which the registry reports as APPLIED with no chain created. Drain
        # only a queue that actually resumed (TASK-33621.19).
        if not resumed.applied or resumed.snapshot.mode is PromptQueueMode.PAUSED:
            return resumed
        await self._drain_waiting(session_id, self._run_status(session_id))
        return resumed

    def reclaim_prepared_entry(
        self, session_id: str, entry_id: str, preparation_id: str
    ) -> QueueGenerationAuthorization | None:
        """Resume and reclaim the exact head entry owned by a paused preparation."""

        resumed = self.resume(session_id)
        if not resumed.applied:
            return None
        claimed = self.registry.claim_next(
            session_id, expected_revision=resumed.snapshot.revision
        )
        if (
            not claimed.applied
            or claimed.claim is None
            or claimed.claim.prompt.entry_id != entry_id
        ):
            if claimed.claim is not None:
                self._return_claim(
                    session_id,
                    claimed.claim.prompt.entry_id,
                    PromptQueuePauseReason.DISPATCH_REFUSED,
                )
            return None
        chain = self._chains[session_id]
        chain.current_entry_id = entry_id
        if not self.bind_claimed_preparation(
            session_id,
            entry_id=entry_id,
            preparation_id=preparation_id,
        ):
            self._return_claim(
                session_id,
                entry_id,
                PromptQueuePauseReason.DISPATCH_REFUSED,
            )
            return None
        self._changed(session_id)
        return QueueGenerationAuthorization(self, session_id, _key=_AUTHORIZATION_KEY)

    async def finish_recovered_entry(
        self,
        session_id: str,
        entry_id: str,
        result: "ConsoleSubmitResult" | None,
    ) -> None:
        """Finish one exact reclaimed send without losing claim ownership."""

        chain = self._chains.get(session_id)
        if chain is None:
            return
        if chain.current_entry_id != entry_id:
            if result is None and chain.current_entry_id is None:
                self._pause_after_exception(session_id)
            return
        if result is None:
            self._pause_after_exception(session_id)
            return
        await self._after_turn(session_id, result)

    def recovered_entry_is_accepted(self, session_id: str, entry_id: str) -> bool:
        """Return whether the exact reclaimed entry crossed acceptance."""

        chain = self._chains.get(session_id)
        return bool(
            chain is not None
            and chain.current_entry_id == entry_id
            and chain.accepted_live_turn
        )

    async def recover_and_drain(
        self,
        session_id: str,
        recovery_turn: Callable[
            [QueueGenerationAuthorization], Awaitable["ConsoleSubmitResult"]
        ],
    ) -> PromptQueueMutationResult:
        """Run one typed failed/stopped recovery, adopt its epoch, then drain."""

        resumed = self.resume(session_id)
        # Same guard as resume_and_drain: a CONTEXT_CHANGED re-pause is
        # APPLIED with no chain, so a recovery turn run now is refused, its
        # refusal is lost and the FAILED/STOPPED pause is written back -- a
        # Retry that silently did nothing. Stop at the review (TASK-33621.19).
        if not resumed.applied or resumed.snapshot.mode is PromptQueueMode.PAUSED:
            return resumed
        authorization = QueueGenerationAuthorization(
            self, session_id, _key=_AUTHORIZATION_KEY
        )
        try:
            result = await recovery_turn(authorization)
        except BaseException:
            self._pause_after_exception(session_id)
            raise
        status = self._terminal_status(session_id, result)
        if not result.accepted or status not in self._SUCCESS:
            reason = (
                PromptQueuePauseReason.STOPPED
                if status is ConsoleRunStatus.STOPPED
                else PromptQueuePauseReason.FAILED
            )
            self._pause_or_finish(session_id, reason, status)
            return resumed
        snapshot = self.registry.snapshot(session_id)
        adopted = self.registry.adopt_recovery_context_baseline(
            session_id,
            context_epoch=self._context_epoch(session_id),
            expected_revision=snapshot.revision,
        )
        if adopted.status not in {
            QueueMutationStatus.APPLIED,
            QueueMutationStatus.UNCHANGED,
        }:
            self._pause_after_exception(session_id)
            return adopted
        await self._drain_waiting(session_id, status)
        return resumed

    async def use_current_context_and_resume(
        self,
        session_id: str,
        *,
        expected_revision: int,
        reviewed_context_epoch: int,
    ) -> PromptQueueMutationResult:
        """Adopt an explicitly reviewed epoch, then visibly reacquire a slot."""
        if self._maintenance_paused:
            return self._maintenance_refusal(session_id)

        snapshot = self.registry.snapshot(session_id)
        current_epoch = self._context_epoch(session_id)
        if (
            snapshot.revision != expected_revision
            or current_epoch != reviewed_context_epoch
        ):
            return PromptQueueMutationResult(
                QueueMutationStatus.STALE_REVISION, snapshot
            )
        adopted = self.registry.adopt_context_baseline(
            session_id,
            context_epoch=current_epoch,
            expected_revision=snapshot.revision,
        )
        if adopted.status not in {
            QueueMutationStatus.APPLIED,
            QueueMutationStatus.UNCHANGED,
        }:
            return adopted
        resumed = self.resume(session_id)
        if resumed.applied:
            await self._drain_waiting(session_id, self._run_status(session_id))
        return resumed

    def mark_closing(self, session_id: str) -> PromptQueueMutationResult:
        """Tombstone and release one chain before cancellation can resume it."""

        snapshot = self.registry.snapshot(session_id)
        result = self.registry.mark_closing(
            session_id, expected_revision=snapshot.revision
        )
        self._chains.pop(session_id, None)
        self._changed(session_id)
        return result

    def remove_session(self, session_id: str) -> PromptQueueMutationResult:
        """Remove all process-memory queue state for a tombstoned session."""

        snapshot = self.registry.snapshot(session_id)
        result = self.registry.remove_session(
            session_id, expected_revision=snapshot.revision
        )
        self._chains.pop(session_id, None)
        self._queue_snapshots.pop(session_id, None)
        self._changed(session_id)
        return result

    def shutdown(self) -> None:
        """Tombstone every chain before controller task cancellation begins."""

        if self._shutting_down:
            return
        session_ids = tuple(self._chains)
        self._shutting_down = True
        self.registry.shutdown(
            expected_registry_revision=self.registry.registry_revision
        )
        self._chains.clear()
        self._queue_snapshots.clear()
        for session_id in session_ids:
            self._changed(session_id)

    def reopen(self) -> None:
        """Re-open admission after a per-visit tombstone (task-15860).

        ``shutdown()`` is a permanent latch: every admission, mutation and
        drain path returns early on ``_shutting_down`` and nothing ever
        clears it. That was correct while a Console screen owned the
        controller and every navigation away built a new one -- the latched
        coordinator died with the screen. With the runtime app-owned, the
        SAME coordinator serves every visit, so leaving Console once would
        have left the prompt queue permanently dead for the rest of the
        app's life.

        This resets the latch only. The chains, the queue snapshots and the
        registry's queued prompts stay cleared by the tombstone, which is
        the pre-existing (and AC#2-required) leaving-Console semantics --
        this call re-opens the door, it does not restore what was behind
        it. Called by ``ConsoleChatController.begin_visit()``, never by a
        disposed controller.
        """

        if not self._shutting_down:
            return
        self._shutting_down = False
        reopen = getattr(self.registry, "reopen", None)
        if callable(reopen):
            reopen()

    def publish_registry_change(self, session_id: str) -> None:
        """Publish a UI-owned registry mutation through the activity cache."""

        self._changed(session_id)

    def _changed(self, session_id: str) -> None:
        if session_id and not self._shutting_down:
            try:
                self._queue_snapshots[session_id] = self.registry.snapshot(session_id)
            except QueueThreadViolation:
                pass
        callback = self.on_activity_changed
        if callback is not None:
            callback(session_id)
