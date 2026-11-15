"""A Console send's start: from its acceptance to its own provider request.

TASK-33620.15.4 (owner-approved exception, 2026-10-24) keeps three Console
display reads on the UI loop during a send's start, from its acceptance to
its provider request and 3 s at most: the Library activity rows and the
Inspector's registry and appearance reads. Off the loop each opened its own
SQLite connection, whose CPU delayed first sends.

TASK-36610: the start belongs to its send. The first version kept one
acceptance time per screen and ended it at the process's latest provider
request, so another session's request ended this send's start early, and
sends the composer did not make (a queued prompt's dispatch, an agent wake,
a goal iteration, an agent's chat start) never opened one. Now each accepted
send's work carries its own ``SendStart`` in a context variable: tasks and
``asyncio.to_thread`` workers created inside copy it, so the provider request
that work makes (``console_send_diagnostics.record_send_stage``) ends that
send's start and no other. The Console asks whether any start on its store
is still open (``send_start_open``); the 3 s limit stays the caller's.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import weakref
from collections.abc import Coroutine, Iterator
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import console_runtime as host


class SendStart:
    """One accepted send's start; it ends once, at its provider request or end."""

    __slots__ = ("accepted_at", "ended")

    def __init__(self) -> None:
        self.accepted_at = time.monotonic()
        self.ended = False

    def end(self, *_args: object) -> None:
        """End this start (idempotent; also a task done callback)."""
        self.ended = True


_BOUND: ContextVar[SendStart | None] = ContextVar("console_send_start", default=None)

#: The starts opened on each store (the Console's whole session set), weakly
#: keyed so a disposed store takes its starts with it. Touched on the UI loop
#: only; a provider request on a worker thread only flips ``ended``.
_OPEN: weakref.WeakKeyDictionary[Any, list[SendStart]] = weakref.WeakKeyDictionary()


@contextlib.contextmanager
def _bound_start(owner: Any) -> Iterator[SendStart]:
    """Open a start on ``owner``; work created inside carries it.

    A failure inside ends the start. An owner that cannot be weakly keyed (a
    test double) still gets a start, which no Console can see.
    """
    start = SendStart()
    with contextlib.suppress(TypeError):
        starts = _OPEN.setdefault(owner, [])
        starts[:] = [item for item in starts if not item.ended]
        starts.append(start)
    token = _BOUND.set(start)
    try:
        yield start
    except BaseException:
        start.end()
        raise
    finally:
        _BOUND.reset(token)


def start_custody_task(owner: Any, coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task:
    """Create an accepted turn's custody task, carrying a start opened now.

    The start ends at the provider request the task's work makes, or when
    the task ends without one.

    Args:
        owner: The Console store the turn was accepted on.
        coroutine: The turn's custody coroutine.

    Returns:
        The lazy task (``asyncio.Task`` directly, before any configurable
        task factory could step it).
    """
    with _bound_start(owner) as start:
        task = asyncio.Task(coroutine, loop=asyncio.get_running_loop())
    task.add_done_callback(start.end)
    return task


@contextlib.contextmanager
def submission_send_start(owner: Any) -> Iterator[SendStart]:
    """A start for a submission made right here, which ends with it.

    For sends that reach the controller without runtime custody (an agent's
    chat start).

    Args:
        owner: The Console store the submission is made on.

    Yields:
        The start, open until the submission's provider request or its end.
    """
    with _bound_start(owner) as start:
        try:
            yield start
        finally:
            start.end()


def note_provider_request() -> None:
    """End the start of the send whose work makes this provider request.

    Called at every gateway adapter entry, so the send's FIRST request ends
    its start. That is normally its main request, but an auxiliary one its
    work makes first (an automatic compaction's summary, a response rule's
    evaluation) ends it there instead, and the rest of the start's display
    reads go off the loop. That costs time only, never wrong data, and the
    base build's process-wide mark ended it at the same request.
    """
    start = _BOUND.get()
    if start is not None:
        start.end()


def send_start_open(owner: Any, limit: float) -> bool:
    """Whether a send on ``owner`` is accepted and has not reached its provider.

    Args:
        owner: The Console store.
        limit: Seconds after acceptance at which a start counts as ended.

    Returns:
        True while any start on ``owner`` is open and younger than ``limit``.
    """
    try:
        starts = _OPEN.get(owner)
    except TypeError:
        return False
    if not starts:
        return False
    now = time.monotonic()
    starts[:] = [
        item for item in starts if not item.ended and now - item.accepted_at < limit
    ]
    return bool(starts)


def _register_custody(
    self,
    request: host.ConsoleTurnCustodyRequest | None,
    attachments: tuple[host.Any, ...] = (),
    staged_evidence_revision: int | None = None,
    *,
    store: host.Any = None,
    received_claim: host.Any = None,
    received_intent: host.Any = None,
) -> host._ConsoleTurnCustodyRecord:
    """Retain one request before its task may begin running."""
    from . import console_runtime as host

    identity = request if request is not None else received_intent
    if identity.turn_id in self._turn_custody:
        raise RuntimeError("Console turn is already in runtime custody.")
    record = host._ConsoleTurnCustodyRecord(
        turn_id=identity.turn_id,
        session_id=identity.session_id,
        request=request,
        store=store if store is not None else self._chat_store,
        received_claim=received_claim,
        received_intent=received_intent,
        inputs=host._ConsoleTurnCustodyInputs(
            attachments=attachments,
            staged_evidence_revision=staged_evidence_revision,
        ),
    )
    self._turn_custody[record.turn_id] = record
    self._clear_staged_evidence_receipt(record.session_id)
    if self._app is not None and self._chat_store is not None:
        session = next(
            (
                item
                for item in self._chat_store.sessions()
                if item.id == identity.session_id
            ),
            None,
        )
        record.archive_conversation_id = (
            session.persisted_conversation_id if session is not None else None
        )
        if record.archive_conversation_id:
            reservations = getattr(self._app, "_conversation_send_inflight", None)
            if reservations is None:
                reservations = self._app._conversation_send_inflight = {}
            reservations[record.archive_conversation_id] = (
                reservations.get(record.archive_conversation_id, 0) + 1
            )
    return record


def _release_custody(
    self, turn_id: str, *, user_proof_transferred: bool = False
) -> None:
    """Release turn inputs, preserving provenance already owned by the queue."""
    record = self._turn_custody.pop(turn_id, None)
    if record is not None:
        if self._archive_recovery is not None and not user_proof_transferred:
            if record.received_intent is not None:
                self._archive_recovery.release_user_proof(
                    record.received_intent.recovery_user_origin
                )
            if record.request is not None:
                self._archive_recovery.release_user_proof(
                    record.request.recovery_user_origin
                )
        if (record.store is not None and record.received_claim is not None) and (
            record.store.release_received_turn(record.received_claim)
        ):
            self._note_received_admission_changed(record.store, record.session_id)
        record.store = None
        record.received_claim = None
        record.received_intent = None
        if record.archive_conversation_id:
            reservations = self._app._conversation_send_inflight
            remaining = reservations.get(record.archive_conversation_id, 1) - 1
            if remaining:
                reservations[record.archive_conversation_id] = remaining
            else:
                reservations.pop(record.archive_conversation_id, None)
            record.archive_conversation_id = None
        record.request = None
        record.inputs.attachments = ()
        record.inputs.staged_evidence_revision = None
        record.task = None
        self._wake_first_send()


def _finish_custodied_turn(
    self,
    task: host.asyncio.Task[host.Any],
    *,
    turn_id: str,
    recover_before_acceptance: bool,
    terminal_callback: host.Callable[[bool], None] | None,
) -> None:
    """Consume a task result and release its retained sensitive inputs."""
    from . import console_runtime as host

    record = self._turn_custody.get(turn_id)
    if record is not None and record.task is not task:
        record = None
    accepted = False
    user_proof_transferred = False
    receipt_state = "outcome_unknown"
    receipt_native_id = turn_id
    try:
        result = task.result()
        from .console_prompt_queue import PromptQueueMutationResult

        if isinstance(result, PromptQueueMutationResult) and result.applied:
            user_proof_transferred = True
            receipt_state = "queued"
            receipt_native_id = result.entry_id or turn_id
        elif bool(getattr(result, "accepted", False)):
            receipt_state = "accepted"
        else:
            receipt_state = "refused"
        accepted = bool(getattr(result, "accepted", False))
    except host.asyncio.CancelledError:
        if (
            record is not None
            and recover_before_acceptance
            and not record.inputs.durable_accepted
            and self._chat_store is not None
            and any(
                item.id == record.session_id for item in self._chat_store.sessions()
            )
        ):
            self._record_turn_recovery(record)
    except BaseException as exc:
        if isinstance(exc, host._ConsoleTurnRefusedError):
            receipt_state = "refused"
        if (
            record is not None
            and recover_before_acceptance
            and not record.inputs.durable_accepted
        ):
            refused = isinstance(exc, host._ConsoleTurnRefusedError)
            self._record_turn_recovery(
                record,
                reason=exc.reason if refused else "",
                preparation_id=exc.preparation_id if refused else None,
                card_held=refused and exc.card_held,
            )
        host.logger.warning(
            "Console runtime turn ended with exception_type={}",
            type(exc).__name__,
        )
    finally:
        accepted = accepted or bool(
            record is not None and record.inputs.durable_accepted
        )
        if record is not None:
            self._observe_remote_custody(
                record.turn_id,
                state="accepted" if record.inputs.durable_accepted else receipt_state,
                native_id=receipt_native_id,
                reason="handoff_interrupted"
                if receipt_state == "outcome_unknown"
                else None,
            )
            self._release_custody(
                record.turn_id, user_proof_transferred=user_proof_transferred
            )
        if terminal_callback is not None:
            try:
                terminal_callback(accepted)
            except Exception as exc:  # noqa: BLE001 -- terminal cleanup is final
                host.logger.warning(
                    "Console runtime terminal callback failed (exception_type={})",
                    type(exc).__name__,
                )
        # Terminal rows and their exact unseen marks are committed by the
        # store before the controller returns. Recompute once per turn,
        # never from streaming/token callbacks.
        self.request_console_attention_refresh()
