"""Custodied turn lifetime, retaining the runtime's locks and native owners."""

from __future__ import annotations

from . import console_runtime as host


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
