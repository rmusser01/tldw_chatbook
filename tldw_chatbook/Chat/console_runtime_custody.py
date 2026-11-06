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
                    record.request.configuration.recovery_user_origin
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


async def _run_custodied_turn(
    self,
    record: host._ConsoleTurnCustodyRecord,
    *,
    origin: host.ConsoleSubmissionOrigin,
    queue_entry_id: str | None,
    queue_authorization: host.Any | None,
    wake_authorization: host.Any | None,
    goal_authorization: host.Any | None,
    raise_on_refusal: bool,
    hook_read: host.Any = None,
) -> host.Any:
    """Run one screen-free turn using only its frozen custody record.

    ``hook_read`` is the received intent's one full hook authority read
    (``ConsoleHookAttemptRead``); it goes, as an argument, only to this
    record's own initial submission (ADR-230 decision 3).
    """
    request = record.request
    controller = self._chat_controller
    if request is None or controller is None:
        raise RuntimeError("Console controller is unavailable for runtime custody.")

    def require_received():
        self._require_received_custody_current(record, controller)

    require_received()
    if self._archive_recovery is not None and record.received_intent is not None:
        self._archive_recovery.accept_user_proof(
            request.configuration.recovery_user_origin,
            request.session_id,
            request.turn_id,
        )
    if self._archive_recovery is not None and origin in {
        host.ConsoleSubmissionOrigin.MANUAL,
        host.ConsoleSubmissionOrigin.QUEUED,
    }:
        await self._archive_recovery.prepare_for_admission(request.session_id)
        require_received()
    if record.archive_conversation_id:
        from tldw_chatbook.Chat.conversation_archive_actions import (
            conversation_send_refusal,
        )

        refusal = await conversation_send_refusal(
            self._app, record.archive_conversation_id
        )
        if refusal:
            notify = getattr(self._app, "notify", None)
            if callable(notify):
                notify(refusal, severity="warning")
            # TASK-33621.2: the parked turn's shelf entry states this
            # refusal too, like a controller refusal's below.
            raise host._ConsoleTurnRefusedError(
                "Console conversation is unavailable for submission.",
                reason=refusal,
            )

    def mark_durable_acceptance() -> None:
        from .console_received_dispatch import publish_received_acceptance

        publish_received_acceptance(self, record, controller)
        self._observe_remote_custody(
            record.turn_id, state="accepted", native_id=record.turn_id
        )

    async def submit() -> host.Any:
        require_received()
        controller.prompt_queue_coordinator.bind_turn_request(request, origin=origin)
        from .console_received_turn import bind_received_turn_claim

        from .console_chat_controller import ConsoleChatController

        shared = (
            {"_hook_read": hook_read}
            if hook_read is not None
            and getattr(controller.submit_draft, "__func__", None)
            is ConsoleChatController.submit_draft
            else {}
        )
        issuing_store, issuing_claim = record.store, record.received_claim
        try:
            with bind_received_turn_claim(issuing_store, issuing_claim):
                return await controller.submit_draft(
                    request.draft,
                    session_id=request.session_id,
                    origin=origin,
                    preserve_composer=request.input_origin == "command",
                    queue_entry_id=queue_entry_id,
                    queue_authorization=queue_authorization,
                    wake_authorization=wake_authorization,
                    goal_authorization=goal_authorization,
                    configuration=host.replace(
                        request.configuration,
                        skill_context_maximum={
                            **request.configuration.skill_context_maximum,
                            "plugin_turn_id": request.turn_id,
                        },
                    ),
                    accepted_attachments=record.inputs.attachments,
                    captured_one_shot_prefill=request.one_shot_prefill,
                    captured_one_shot_prefill_revision=(
                        request.one_shot_prefill_revision
                    ),
                    staged_evidence_launch=request.staged_evidence_launch,
                    staged_evidence_capture=self._capture_frozen_console_staged_rag,
                    staged_evidence_release=host.functools.partial(
                        self.release_console_staged_evidence,
                        revision=record.inputs.staged_evidence_revision,
                        session_id=record.session_id,
                    ),
                    custody_acceptance_hook=mark_durable_acceptance,
                    **shared,
                )
        finally:
            # Nonpromoting routes release after physical submit completion,
            # before the enclosing prompt chain admits its next request.
            if issuing_store.release_received_turn(issuing_claim):
                self._note_received_admission_changed(issuing_store, request.session_id)

    result = (
        await submit()
        if origin
        in {
            host.ConsoleSubmissionOrigin.AGENT_WAKE,
            host.ConsoleSubmissionOrigin.GOAL_ITERATION,
        }
        else await controller.run_prompt_chain(
            session_id=request.session_id,
            initial_turn=submit,
        )
    )
    held_preparation_id = getattr(result, "preparation_id", None)
    if (
        not bool(getattr(result, "accepted", False))
        and not record.inputs.durable_accepted
        and raise_on_refusal
        # TASK-34350: a send held at the compaction threshold is waiting
        # on the user's answer in its own card, not refused; recording it
        # as a turn recovery too showed one send on two surfaces.
        and not (
            held_preparation_id is not None
            and controller.context_compaction_hold(held_preparation_id) is not None
        )
    ):
        from .console_unsent_turn import card_holds_refusal

        raise host._ConsoleTurnRefusedError(
            "Console turn was refused before durable acceptance.",
            reason=str(getattr(result, "visible_copy", "") or ""),
            preparation_id=held_preparation_id,
            card_held=card_holds_refusal(controller, held_preparation_id),
        )
    run_state_for = getattr(controller, "run_state_for", None)
    run_state = run_state_for(request.session_id) if callable(run_state_for) else None
    if (
        origin is host.ConsoleSubmissionOrigin.MANUAL
        and bool(getattr(result, "accepted", False))
        and getattr(run_state, "status", None) is host.ConsoleRunStatus.COMPLETED
    ):
        await self._record_successful_first_send()
    return result


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
