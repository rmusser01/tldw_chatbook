"""Controller-side authority for sequential Console prompt queue drains."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

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
    QueueMutationStatus,
    QueueThreadViolation,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult


_AUTHORIZATION_KEY = object()


class QueueGenerationAuthorization:
    """Opaque, coordinator-issued authority to cross a queue-owned send gate."""

    __slots__ = ("_coordinator", "session_id")

    def __init__(self, coordinator: object, session_id: str, *, _key: object) -> None:
        if _key is not _AUTHORIZATION_KEY:
            raise PermissionError("queue generation authority is coordinator-internal")
        self._coordinator = coordinator
        self.session_id = session_id

    def __repr__(self) -> str:
        return (
            "QueueGenerationAuthorization("
            f"session_id={self.session_id!r}, authority=<redacted>)"
        )


class QueuedTurnSubmitter(Protocol):
    """Production-shaped callback used to submit one claimed queue entry."""

    def __call__(
        self,
        text: str,
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
        on_chain_terminal: Callable[[str, ConsoleRunStatus], None] | None = None,
    ) -> None:
        self.registry = registry
        self._context_epoch = context_epoch
        self._run_status = run_status
        self._submit_queued = submit_queued
        self._has_staged_rider = has_staged_rider or (lambda _session_id: False)
        self._needs_approval = needs_approval or (lambda _session_id: False)
        self._can_reacquire_slot = can_reacquire_slot or (lambda _session_id: True)
        self.on_queued_accepted = on_queued_accepted
        self.on_activity_changed = on_activity_changed
        self.on_chain_terminal = on_chain_terminal
        self._chains: dict[str, _PromptChain] = {}
        self._queue_snapshots: dict[str, PromptQueueSnapshot] = {}
        self._dispatch_recoveries: dict[str, _RecoveredQueueOwner] = {}
        self._settled_dispatch_recoveries: OrderedDict[tuple[str, str, str], None] = (
            OrderedDict()
        )
        self._shutting_down = False

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
        )

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
        preparing = (
            status
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
        if not resumed.applied:
            return resumed
        await self._drain_waiting(session_id, terminal_status)
        return resumed

    def pause_for_stop(self, session_id: str) -> PromptQueueMutationResult:
        """Release a chain reservation as soon as Stop targets its live turn."""

        snapshot = self.registry.snapshot(session_id)
        if snapshot.total_count == 0:
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
    ) -> PromptQueueMutationResult:
        """Admit text only behind an accepted turn or an existing queue."""

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
        )
        if result.applied:
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

        if session_id in self._chains:
            return await initial_turn()
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
        defer_queued_settlement: bool = False,
    ) -> None:
        """Commit the accepted boundary and settle a queued claim exactly once."""

        chain = self._chains.get(session_id)
        if chain is None:
            if origin is ConsoleSubmissionOrigin.MANUAL:
                return
            if (
                origin is ConsoleSubmissionOrigin.AGENT_WAKE
                and entry_id is None
            ):
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
        result = self.registry.settle_durable_acceptance(
            session_id,
            entry_id=entry_id,
            preparation_id=preparation_id,
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
            if chain.current_entry_id == entry_id:
                chain.current_entry_id = None
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
            reason = (
                PromptQueuePauseReason.STOPPED
                if status is ConsoleRunStatus.STOPPED
                else PromptQueuePauseReason.FAILED
            )
            self._pause_or_finish(session_id, reason, status)
            return

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
                    claim.prompt.text,
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
                self._return_claim(
                    session_id,
                    claim.prompt.entry_id,
                    PromptQueuePauseReason.DISPATCH_REFUSED,
                )
                return
            if status not in self._SUCCESS:
                reason = (
                    PromptQueuePauseReason.STOPPED
                    if status is ConsoleRunStatus.STOPPED
                    else PromptQueuePauseReason.FAILED
                )
                self._pause_or_finish(session_id, reason, status)
                return

    def _return_claim(
        self, session_id: str, entry_id: str, reason: PromptQueuePauseReason
    ) -> None:
        snapshot = self.registry.snapshot(session_id)
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

    def _finish_visible_terminal(
        self, session_id: str, status: ConsoleRunStatus
    ) -> None:
        chain = self._chains.get(session_id)
        if chain is not None:
            chain.accepted_live_turn = False
        self._chains.pop(session_id, None)
        self._changed(session_id)
        callback = self.on_chain_terminal
        if callback is not None and status in self._TERMINAL:
            callback(session_id, status)

    def resume(self, session_id: str) -> PromptQueueMutationResult:
        """Reacquire a slot and resume a manually/dispatch-paused queue."""

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
            self._chains[session_id] = _PromptChain()
            self._changed(session_id)
        return resumed

    async def resume_and_drain(self, session_id: str) -> PromptQueueMutationResult:
        """Reacquire one slot and dispatch the next waiting entry."""

        resumed = self.resume(session_id)
        if not resumed.applied:
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
        if not resumed.applied:
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
