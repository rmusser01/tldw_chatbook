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
fails closed and showed "Response recovery failed". An explicit Stop (the run's
own cancel event is set) is suppressed at the window's edge, and :meth:`settle`
records it as Stopped:

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
)

if TYPE_CHECKING:
    from .console_chat_controller import ConsoleSubmitResult

#: The run-state copy every other agent-reply Stop shows.
STOPPED_COPY = "Response stopped."
#: The transcript record an explicit user Stop leaves (TASK-337 AC3).
USER_STOP_RECORD = "Response stopped by user."


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
    """

    def __init__(
        self,
        controller: Any,
        assistant_message_id: str,
        cancel_event: threading.Event,
        *,
        prepares_row: bool = False,
    ) -> None:
        self._controller = controller
        self._assistant_message_id = assistant_message_id
        self._cancel_event = cancel_event
        self._prepares_row = prepares_row
        #: True once an explicit Stop ended the window; :meth:`settle` is due.
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
            return _source_controller.ConsoleSubmitResult(True, True, stopped.content)
        finally:
            self._registrations().discard(assistant_id)

    async def _settle_accepted(self, session_id: str) -> ConsoleSubmitResult:
        """Settle an ACCEPTED durable checkpoint as stopped, never as sent."""
        from . import console_chat_controller as _source_controller
        from .console_chat_store import ConsoleDispatchSettlementError
        from .console_native_commit import settle_accepted_durable_turn_owned

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
        persistence = store.persistence
        completion = await settle_accepted_durable_turn_owned(
            store,
            preparation.preparation_id,
            fingerprint=fingerprint,
            terminal_state="stopped",
            content="",
            persistence=persistence,
            database=getattr(persistence, "db", None),
        )
        if not completion.settled or completion.error is not None:
            raise ConsoleDispatchSettlementError(
                "Stopped turn could not settle before provider dispatch."
            )
        publish_stopped_run(controller, assistant_id, session_id, STOPPED_COPY)
        return _source_controller.ConsoleSubmitResult(True, True, "")


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
