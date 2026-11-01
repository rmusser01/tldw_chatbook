"""Settle a Stop pressed while the agent reply sets up (TASK-33621.22).

Between "console agent reply start" and the provider call, the agent reply
composes its tools (the row reads "Connecting tools…"), waits for trace
maintenance, moves the durable dispatch checkpoint from ACCEPTED to
DISPATCH_STARTED and builds the profile tools. Stop cancels the reply's task.
Only two places used to settle that cancellation: the bridge call after this
window and the checkpoint move itself. A Stop anywhere else raised a bare
``CancelledError`` out of the reply. The stream wrapper then retired the run's
cancel event, so the accepted turn's handler took the Stop for a failure, and
the turn's preparation kept the tab's send slot. Every later send was refused
("Draft or chat changed; Send again.") until restart.

:class:`PreProviderSetupStop` wraps the window. While it is open,
``stop_active_run`` leaves settlement to it, as it already did while the
checkpoint move drains: settling an ACCEPTED durable checkpoint synchronously
fails closed and showed "Response recovery failed". An explicit lifecycle cancellation (Stop, close or shutdown sets the run's
own cancel event) is suppressed at the window's edge, and :meth:`settle`
records it as Stopped. Only user Stop earns the transcript marker:

* An ACCEPTED durable checkpoint settles directly as stopped, through the
  store's accepted-turn settlement, so no provider dispatch is invented.
* Any other row settles through ``_mark_stream_stopped``, as a Stop during the
  checkpoint move does. A Retry (or a variant) enters the window with its row
  still showing the previous outcome, a failed reply, until the reply prepares
  it for the new attempt; a Stop before that leaves the row as it was, so Retry
  is offered again, and still stops the run.

A settlement that cannot complete raises ``ConsoleDispatchSettlementError``, and
the accepted-turn handler retains the turn for recovery. Any other cancellation
propagates unchanged.
"""

from __future__ import annotations

import asyncio
import threading
from types import TracebackType
from typing import TYPE_CHECKING, Any

from loguru import logger

from .console_chat_models import (
    ConsoleDispatchRecoveryKind,
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleSubmissionOrigin,
)

if TYPE_CHECKING:
    from .console_chat_controller import ConsoleSubmitResult

#: The run-state copy every other agent-reply Stop shows.
STOPPED_COPY = "Response stopped."
#: The transcript record an explicit user Stop leaves (TASK-337 AC3).
USER_STOP_RECORD = "Response stopped by user."


class AcceptedStopAuthority:
    """Snapshot publication authority without extending a native lease."""

    def __init__(self, controller: Any, session_id: str) -> None:
        self.store = controller.store
        self.persistence = self.store.persistence
        self.database = getattr(self.persistence, "db", None)
        self.session = next(
            (row for row in self.store.sessions() if row.id == session_id), None
        )

    def is_current(self, controller: Any) -> bool:
        return (
            controller.store is self.store
            and self.store.persistence is self.persistence
            and getattr(self.persistence, "db", None) is self.database
            and self.session is not None
            and any(row is self.session for row in self.store.sessions())
        )

    def live_ephemeral_acceptance(
        self, preparation: Any, recovery: Any, assistant_id: str
    ) -> bool:
        """Prove this exact process-local acceptance still owns terminalization."""
        from .console_dispatch_checkpoint import ConsoleDispatchCheckpointState
        from .console_turn_preparation import ConsoleTurnPreparationState

        if self.session is None or preparation is None or recovery is None:
            return False
        store = self.store
        with store.durable_preparation_lock:
            return (
                store.persistence is self.persistence
                and getattr(self.persistence, "db", None) is self.database
                and any(row is self.session for row in store.sessions())
                and self.session.ephemeral
                and store.preparation_for_session(self.session.id) is preparation
                and preparation.state is ConsoleTurnPreparationState.ACCEPTED
                and store.dispatch_recovery_for_session(self.session.id) is recovery
                and recovery.kind is ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED
                and recovery.preparation_id == preparation.preparation_id
                and recovery.assistant_message_id == assistant_id
                and recovery.runtime_active
                and not recovery.recovery_needed
                and not recovery.in_flight
                and recovery.checkpoint is not None
                and recovery.checkpoint.state is ConsoleDispatchCheckpointState.ACCEPTED
            )

    def accepted_before_dispatch(self, continuation: Any, assistant_id: str) -> bool:
        """Prove no dispatch from this captured owner's exact ACCEPTED state."""
        from .console_dispatch_checkpoint import ConsoleDispatchCheckpointState
        from .console_turn_preparation import ConsoleTurnPreparationState

        if (
            continuation is None
            or self.session is None
            or continuation.session_id != self.session.id
            or continuation.commit.assistant_message_id != assistant_id
        ):
            return False
        store = self.store
        preparation_id = continuation.preparation_id
        commit = continuation.commit
        try:
            with store.durable_preparation_lock:
                preparation = store.preparation_for_session(continuation.session_id)
                recovery = store.dispatch_recovery_for_session(continuation.session_id)
                return (
                    store.persistence is self.persistence
                    and getattr(self.persistence, "db", None) is self.database
                    and any(row is self.session for row in store.sessions())
                    and preparation is not None
                    and preparation.preparation_id == preparation_id
                    and preparation.state is ConsoleTurnPreparationState.ACCEPTED
                    and store.durable_acceptance_fingerprint_for(preparation_id)
                    == continuation.fingerprint
                    and store.durable_turn_commit_for(
                        preparation_id, fingerprint=continuation.fingerprint
                    )
                    is commit
                    and recovery is not None
                    and recovery.assistant_message_id == assistant_id
                    and recovery.preparation_id == preparation_id
                    and recovery.kind is ConsoleDispatchRecoveryKind.ACCEPTED
                    and not recovery.in_flight
                    and recovery.checkpoint == commit.checkpoint
                    and commit.checkpoint.state
                    is ConsoleDispatchCheckpointState.ACCEPTED
                )
        except Exception:  # noqa: BLE001 - unavailable original proof stays conservative.
            return False


class PreProviderSetupStop:
    """Own a Stop that lands in one agent reply's pre-provider setup window.

    Use it as an async context manager around the window, then call
    :meth:`settle` when :attr:`stopped` is set. Entering registers the reply's
    assistant row in the controller's ``_replies_in_pre_provider_setup``, which
    tells ``stop_active_run`` to cancel the reply instead of settling the row
    itself. The registration lasts until the window closes without a Stop, or
    until :meth:`settle` finishes, so a repeated Stop never races the
    settlement.

    Args:
        controller: The ``ConsoleChatController`` running the reply.
        assistant_message_id: The reply's assistant row.
        cancel_event: The run's own cancel event; ``_signal_stop`` sets it for
            Stop, close and shutdown.
        prepares_row: True for a Retry or a variant, whose row the reply
            prepares for the new attempt inside the window; the reply sets
            :attr:`row_prepared` once it has.
        accepted_content: User-authored direct prefill to retain on a stopped
            accepted row; never provider output.
    """

    def __init__(
        self,
        controller: Any,
        assistant_message_id: str,
        cancel_event: threading.Event,
        *,
        prepares_row: bool = False,
        accepted_content: str = "",
    ) -> None:
        self._controller = controller
        self._assistant_message_id = assistant_message_id
        self._cancel_event = cancel_event
        self._prepares_row = prepares_row
        self._accepted_content = accepted_content
        #: True once lifecycle cancellation ended the window; settlement is due.
        self.stopped = False
        #: True once a Retry or variant row was prepared for the new attempt.
        self.row_prepared = False

    def _registrations(self) -> set[str]:
        return self._controller._replies_in_pre_provider_setup

    async def __aenter__(self) -> PreProviderSetupStop:
        self._registrations().add(self._assistant_message_id)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if (
            exc_type is not None
            and issubclass(exc_type, asyncio.CancelledError)
            and self._cancel_event.is_set()
        ):
            # Keep the registration: settle() still owns this Stop.
            self.stopped = True
            return True
        self._registrations().discard(self._assistant_message_id)
        return False

    async def settle(self, session_id: str) -> ConsoleSubmitResult:
        """Record the Stop on the reply's row and return the reply's result.

        Args:
            session_id: The session that owns the reply.

        Returns:
            The stopped result the reply returns. The accepted turn then
            settles its preparation, which frees the tab's send slot.

        Raises:
            ConsoleDispatchSettlementError: The row could not settle; the
                accepted-turn handler retains the turn for recovery.
        """
        controller = self._controller
        assistant_id = self._assistant_message_id
        try:
            recovery = controller.store.dispatch_recovery_for_session(session_id)
            accepted = (
                recovery is not None
                and recovery.assistant_message_id == assistant_id
                and recovery.kind is ConsoleDispatchRecoveryKind.ACCEPTED
                and not recovery.in_flight
            )
            # Content-free: which settlement a Stop in this window took.
            logger.info(
                "Console Stop settled in pre-provider setup | checkpoint={}",
                "accepted" if accepted else "dispatch_started_or_none",
            )
            if accepted:
                return await self._settle_accepted(session_id)
            from . import console_chat_controller as _source_controller

            try:
                # A Retry stopped before its row was prepared leaves the row
                # as it was ("failed"); marking it stopped would be refused,
                # and the run would never be stopped.
                stopped = controller._mark_stream_stopped(
                    assistant_id,
                    visible_copy=STOPPED_COPY,
                    prepare_retry=self._prepares_row,
                    retry_prepared=self.row_prepared,
                )
            except KeyError:
                return controller._session_closed_result(session_id=session_id)
            return _source_controller.ConsoleSubmitResult(
                True,
                True,
                stopped.content,
                terminal_status=controller.run_state_for(session_id).status,
                provider_started=False,
            )
        finally:
            self._registrations().discard(assistant_id)

    async def _settle_accepted(self, session_id: str) -> ConsoleSubmitResult:
        """Settle an ACCEPTED durable checkpoint as stopped, never as sent."""
        from . import console_chat_controller as _source_controller
        from .console_chat_store import ConsoleDispatchSettlementError
        from .console_native_commit import (
            accepted_terminal_state_matches,
            settle_accepted_durable_turn_owned,
        )

        controller = self._controller
        store = controller.store
        assistant_id = self._assistant_message_id
        preparation = store.preparation_for_session(session_id)
        fingerprint = (
            store.durable_acceptance_fingerprint_for(preparation.preparation_id)
            if preparation is not None
            else None
        )
        if fingerprint is None or fingerprint.assistant_message_id != assistant_id:
            raise ConsoleDispatchSettlementError(
                "Stopped turn has no matching durable acceptance."
            )
        commit = store.durable_turn_commit_for(
            preparation.preparation_id, fingerprint=fingerprint
        )
        if self._accepted_content:
            continuation = controller._durable_postcommit_continuations.get(
                preparation.preparation_id
            )
            if (
                continuation is None
                or continuation.fingerprint != fingerprint
                or continuation.commit is not commit
                or continuation.prefill != self._accepted_content
            ):
                raise ConsoleDispatchSettlementError(
                    "Stopped prefill no longer matches its accepted continuation."
                )
        authority = AcceptedStopAuthority(controller, session_id)
        completion = await settle_accepted_durable_turn_owned(
            store,
            preparation.preparation_id,
            fingerprint=fingerprint,
            terminal_state="stopped",
            content=self._accepted_content,
            persistence=authority.persistence,
            database=authority.database,
            expected_accepted_prefill=self._accepted_content,
        )
        settled = (
            completion.settled and completion.error is None
        ) or accepted_terminal_state_matches(
            store,
            preparation.preparation_id,
            fingerprint=fingerprint,
            commit=commit,
            terminal_state="stopped",
            content=self._accepted_content,
        )
        current = authority.is_current(controller)
        if not settled and current:
            raise ConsoleDispatchSettlementError(
                "Stopped turn could not settle before provider dispatch."
            )
        if current:
            publish_stopped_run(controller, assistant_id, session_id, STOPPED_COPY)
        else:
            controller._deferred_user_stop_markers.discard(assistant_id)
        return _source_controller.ConsoleSubmitResult(
            True,
            current,
            self._accepted_content,
            session_id=session_id,
            user_message_id=commit.user_message_id,
            assistant_message_id=assistant_id,
            terminal_status=(
                ConsoleRunStatus.STOPPED if settled else ConsoleRunStatus.BLOCKED
            ),
            preparation_id=preparation.preparation_id,
            provider_started=False,
        )


def request_accepted_pre_stream_stop(
    controller: Any, session_id: str, *, record_user_stop: bool
) -> bool:
    """Cancel only the live accepted submit before a stream owner takes over."""
    from .console_turn_preparation import ConsoleTurnPreparationState

    if (
        session_id in controller._active_stream_tasks
        and session_id in controller._active_cancel_events
        and (
            controller._active_assistant_message_ids.get(session_id)
            in controller._replies_in_pre_provider_setup
            or controller._active_assistant_message_ids.get(session_id)
            in controller._pending_dispatch_transitions
        )
    ) or controller._ordinary_native_commit_tasks(session_id):
        return False
    store = controller.store
    with controller._active_submit_tasks_lock, store.durable_preparation_lock:
        preparation = store.preparation_for_session(session_id)
        if (
            preparation is None
            or preparation.state is not ConsoleTurnPreparationState.ACCEPTED
        ):
            return False
        continuation = controller._durable_postcommit_continuations.get(
            preparation.preparation_id
        )
        recovery = store.dispatch_recovery_for_session(session_id)
        ephemeral = AcceptedStopAuthority(
            controller, session_id
        ).live_ephemeral_acceptance(
            preparation, recovery, recovery.assistant_message_id if recovery else ""
        )
        if not ephemeral:
            if (
                continuation is None
                or continuation.session_id != session_id
                or store.durable_acceptance_fingerprint_for(preparation.preparation_id)
                != continuation.fingerprint
                or store.durable_turn_commit_for(
                    preparation.preparation_id, fingerprint=continuation.fingerprint
                )
                is not continuation.commit
                or recovery is None
                or recovery.kind is not ConsoleDispatchRecoveryKind.ACCEPTED
                or recovery.assistant_message_id
                != continuation.commit.assistant_message_id
                or recovery.preparation_id != preparation.preparation_id
                or recovery.checkpoint != continuation.commit.checkpoint
                or recovery.in_flight
                or not recovery.runtime_active
                or recovery.recovery_needed
            ):
                return False
        tasks = tuple(
            task
            for task, owner_session in controller._active_submit_tasks.items()
            if owner_session == session_id
            and controller._active_submit_preparations.get(task)
            == preparation.preparation_id
            and not task.done()
        )
        if len(tasks) != 1 or tasks[0] is asyncio.current_task():
            return False
        assistant_id = recovery.assistant_message_id
        if ephemeral and (
            controller._active_stream_tasks.get(session_id) is not tasks[0]
            or controller._active_assistant_message_ids.get(session_id) != assistant_id
            or session_id in controller._active_cancel_events
        ):
            return False
    # Stream registration precedes ledger/project startup. Keep this exact
    # accepted task's settlement ownership until the actual reply takes over.
    controller._replies_in_pre_provider_setup.add(assistant_id)
    controller._active_cancel_events.setdefault(session_id, threading.Event())
    controller._signal_stop(
        session_id=session_id, reason="user" if record_user_stop else "shutdown"
    )
    controller.prompt_queue_coordinator.pause_for_stop(session_id)
    if record_user_stop:
        controller._deferred_user_stop_markers.add(assistant_id)
    controller._cancel_task_on_owner_loop(tasks[0])
    return True


async def settle_accepted_pre_stream_stop(
    controller: Any, continuation: Any
) -> ConsoleSubmitResult:
    """Settle the cancelled accepted continuation and retire its early signal."""
    from .console_chat_controller import ConsoleSubmitResult
    from .console_chat_store import ConsoleDispatchSettlementError

    session_id = continuation.session_id
    assistant_id = continuation.commit.assistant_message_id
    cancel_event = controller._active_cancel_events.get(session_id)
    authority = AcceptedStopAuthority(controller, session_id)
    stop = PreProviderSetupStop(
        controller, assistant_id, cancel_event or threading.Event()
    )
    try:
        return await stop.settle(session_id)
    except ConsoleDispatchSettlementError:
        if authority.is_current(controller):
            controller._restore_dispatch_recovery_after_settlement_failure(
                session_id, assistant_id
            )
        if (
            authority.is_current(controller)
            and continuation.origin is ConsoleSubmissionOrigin.QUEUED
        ):
            controller.prompt_queue_coordinator.retain_durable_acceptance(session_id)
        return ConsoleSubmitResult(
            True,
            authority.is_current(controller),
            "Accepted turn is retained for recovery.",
            session_id=session_id,
            user_message_id=continuation.commit.user_message_id,
            assistant_message_id=assistant_id,
            terminal_status=ConsoleRunStatus.BLOCKED,
            origin=continuation.origin,
            queue_entry_id=continuation.queue_entry_id,
            committed_context_epoch=continuation.committed_context_epoch,
            preparation_id=continuation.preparation_id,
            provider_started=False,
        )
    finally:
        controller._deferred_user_stop_markers.discard(assistant_id)
        if (
            cancel_event is not None
            and controller._active_cancel_events.get(session_id) is cancel_event
        ):
            controller._active_cancel_events.pop(session_id, None)
            controller._active_stop_reasons.pop(session_id, None)


def publish_stopped_run(
    controller: Any,
    assistant_message_id: str,
    session_id: str | None,
    visible_copy: str,
) -> None:
    """Publish a settled Stop: the run state, then the user's Stop record.

    The one place both Stop settlements publish from (the controller's
    ``_mark_stream_stopped`` and an ACCEPTED checkpoint settled here), so the
    two cannot drift apart.

    Args:
        controller: The ``ConsoleChatController`` that ran the reply.
        assistant_message_id: The stopped reply's assistant row.
        session_id: The session that owns the row; ``None`` when it was closed
            under the run, which leaves nothing to record the Stop in.
        visible_copy: The STOPPED run-state copy.
    """
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STOPPED, visible_copy),
        session_id=session_id,
    )
    markers = controller._deferred_user_stop_markers
    if assistant_message_id in markers:
        markers.discard(assistant_message_id)
        if session_id is not None:
            controller.store.append_message(
                session_id, role=ConsoleMessageRole.SYSTEM, content=USER_STOP_RECORD
            )
