"""Resend a broken last Console turn in place (TASK-33661).

A turn is *broken* when its send was refused before acceptance, it got no
assistant reply, its reply failed, or its reply is empty and stopped,
discarded, or restored after a restart as "Response failed.". Only the last
user turn on the active path qualifies: a failed reply higher up keeps its own
Retry, and a partial (non-empty) stopped reply has Continue. A turn that holds
any tool output, text from an earlier reply (a Continue chain), or a restored
reply with text is partial, never broken: Resend would discard that work.

Resend never forks: no ``create_sibling``, no ``edit_and_resend_message``.
A failed live reply is retried in place on the same assistant row. Every other
persisted shape clears the empty reply and the stale failure or stop rows (a
subtree tombstone), then re-runs from the same user message with the gates a
normal send applies. A refused echo was never persisted, so it is removed and
its text and attachments go back through the normal send path.
"""

from __future__ import annotations

import asyncio

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from tldw_chatbook.Chat.attachment_core import PendingAttachment, vision_block_reason
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_message_actions import is_refused_echo, resend_target_id
from tldw_chatbook.Utils.input_validation import validate_console_draft

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_chat_controller import (
        ConsoleChatController,
        ConsoleSubmitResult,
    )

RESEND_NOT_BROKEN_COPY = "Only the last broken turn can be resent."
RESEND_OTHER_SESSION_COPY = "Open the original session before resending this message."
RESEND_COMPOSER_BUSY_COPY = (
    "Send or clear the composer draft before resending this message."
)
RESEND_STAGED_ATTACHMENTS_COPY = (
    "Send or remove the staged attachments before resending this message."
)


def _delete_rows_after(store: Any, session_id: str, anchor_id: str) -> str | None:
    """Tombstone everything on the active path after ``anchor_id``.

    The first tree row after the anchor roots every later row, so deleting its
    subtree clears the rest. Display-only TOOL markers are not tree nodes.

    Returns:
        The store's refusal copy when the delete was refused, else ``None``.
    """
    messages = store.messages_for_session(session_id)
    anchor = next(i for i, row in enumerate(messages) if row.id == anchor_id)
    first = next(
        (
            row
            for row in messages[anchor + 1 :]
            if row.role is not ConsoleMessageRole.TOOL
        ),
        None,
    )
    try:
        if first is not None:
            store.delete_message(first.id)
    except (ValueError, RuntimeError) as exc:
        # A pending dispatch or an unpersistable tombstone refuses the delete;
        # report it like any other refusal instead of failing the worker.
        return str(exc)
    return None


async def resend_turn(
    controller: ConsoleChatController,
    message_id: str,
    *,
    resend_echo: Callable[[ConsoleChatMessage], Awaitable[str | None]] | None = None,
) -> ConsoleSubmitResult:
    """Re-run a broken last turn in place from its user message.

    The controller's maintenance (backup) admission is taken first and held
    for the whole resend, so a backup pause refuses before anything is cleared
    and cannot start between the clear and the re-run. Then, still before any
    clear: the send-refusal copy (a live run, a queue, an unresolved dispatch
    recovery), the broken-turn check, and the vision gate for the turn's own
    attachments. A refused echo then goes to ``resend_echo`` (the normal send
    path, with its own gates). A failed reply's trailing rows are cleared and
    it is retried in place; otherwise the empty reply and the rows after the
    user message are tombstoned and the turn re-runs from the user message.
    Hook review, required initialization/input and readiness run under the
    normal submission owner before clearing. Skill refusal and the thinking
    preflight run in the Retry/Continue bodies after the clear: a refusal
    there leaves the turn still broken (Resend stays on offer) but the cleared
    rows stay cleared. In a temporary chat nothing is persisted, so a cleared
    empty reply is simply gone.

    Args:
        controller: The Console chat controller owning the session.
        message_id: The broken turn's user message id.
        resend_echo: Re-sends a refused echo; returns refusal copy or None.

    Returns:
        The re-run's submit result; a refusal carries its visible copy.
    """
    from tldw_chatbook.Chat.console_chat_controller import _maintenance_boundary

    admitted = _maintenance_boundary("turn")(_resend_turn)
    return await admitted(controller, message_id, resend_echo=resend_echo)


async def _resend_turn(
    controller: ConsoleChatController,
    message_id: str,
    *,
    resend_echo: Callable[[ConsoleChatMessage], Awaitable[str | None]] | None,
) -> ConsoleSubmitResult:
    from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult

    store = controller.store
    try:
        session_id = store.session_id_for_message(message_id)
    except KeyError:
        return ConsoleSubmitResult(False, False, RESEND_NOT_BROKEN_COPY)
    if session_id != store.active_session_id:
        return ConsoleSubmitResult(False, False, RESEND_OTHER_SESSION_COPY)
    refusal = controller.send_refusal_copy(session_id)
    if refusal:
        return ConsoleSubmitResult(False, False, refusal)
    messages = store.messages_for_session(session_id)
    if resend_target_id(messages) != message_id:
        return ConsoleSubmitResult(False, False, RESEND_NOT_BROKEN_COPY)
    user = store.get_message(message_id)
    if is_refused_echo(user):
        copy = await resend_echo(user) if resend_echo else RESEND_NOT_BROKEN_COPY
        return ConsoleSubmitResult(copy is None, False, copy or "")
    if any(attachment.data is not None for attachment in user.attachments):
        configuration = await controller.capture_turn_configuration_snapshot(session_id)
        block_reason = vision_block_reason(
            configuration.provider_selection.provider,
            configuration.effective_model,
            is_capable=lambda _provider, _model: bool(
                configuration.capabilities.get("vision", False)
            ),
        )
        if block_reason is not None:
            return controller._block(session_id, block_reason)
    position = next(i for i, row in enumerate(messages) if row.id == message_id)
    reply = next(
        (
            row
            for row in reversed(messages[position + 1 :])
            if row.role is ConsoleMessageRole.ASSISTANT
        ),
        None,
    )
    snapshot = tuple((row.id, row.status, row.content) for row in messages)

    async def replay(resolution, context, hook_rows):
        current = store.messages_for_session(session_id)
        if (
            resend_target_id(current) != message_id
            or tuple((row.id, row.status, row.content) for row in current) != snapshot
        ):
            return ConsoleSubmitResult(False, False, RESEND_NOT_BROKEN_COPY)
        in_place = reply is not None and reply.status == "failed"
        anchor_id = reply.id if in_place else message_id
        refusal = _delete_rows_after(store, session_id, anchor_id)
        if refusal:
            return ConsoleSubmitResult(False, False, refusal)
        if in_place:
            return await controller._retry_message_body(
                reply.id, resolution, context, hook_rows
            )
        return await controller._continue_from_message_body(
            message_id, resolution, context, hook_rows, resend=True
        )

    return await _replay_message(controller, message_id, replay, prompt=user.content)


async def _replay_message(
    controller, message_id, operation, *, queue_authorization=None, prompt=None
):
    """Run an in-place replay under normal Send's exact submission owner."""
    from tldw_chatbook.Agents.automatic_work_runtime import manual_work_scope
    from tldw_chatbook.Chat.console_chat_controller import (
        ConsoleRunState,
        ConsoleRunStatus,
        ConsoleSubmissionOrigin,
        ConsoleSubmitResult,
    )

    session_id = controller.store.session_id_for_message(message_id)
    origin = (
        ConsoleSubmissionOrigin.QUEUED
        if controller.prompt_queue_coordinator.authorizes(
            queue_authorization, session_id
        )
        else ConsoleSubmissionOrigin.MANUAL
    )
    if prompt is None:
        rows = controller.store.messages_for_session(session_id)
        position = next(i for i, row in enumerate(rows) if row.id == message_id)
        prompt = next(
            (
                row.content
                for row in reversed(rows[: position + 1])
                if row.role is ConsoleMessageRole.USER
            ),
            "",
        )

    def refused(copy):
        controller._set_run_state(ConsoleRunState.blocked(copy), session_id=session_id)
        return ConsoleSubmitResult(
            False, False, copy, session_id=session_id, origin=origin
        )

    async def replay():
        with manual_work_scope():
            rejection = controller._active_run_rejection(
                session_id=session_id, queue_authorization=queue_authorization
            )
            if rejection is not None:
                return rejection
            if origin is ConsoleSubmissionOrigin.MANUAL and (
                controller.store.active_session_id != session_id
            ):
                return ConsoleSubmitResult(False, False, RESEND_OTHER_SESSION_COPY)
            session = next(
                row for row in controller.store.sessions() if row.id == session_id
            )
            configuration = await controller.capture_turn_configuration_snapshot(
                session_id
            )
            rejection = await controller._prepare_submission_hooks(
                session, configuration, origin, queue_authorization, recovery=True
            )
            if rejection is not None:
                return rejection
            controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider."),
                session_id=session_id,
            )
            (
                resolution,
                context,
            ) = await controller._capture_and_resolve_turn_execution_context(
                session_id, configuration
            )
            if not getattr(resolution, "ready", False):
                return controller._block(
                    session_id,
                    controller._blocked_visible_copy(
                        getattr(resolution, "visible_copy", "")
                    ),
                )
            try:
                hook_rows = await controller._submission_hook_input(prompt, origin)
            except Exception:  # noqa: BLE001 -- same required input boundary as Send
                return refused("Required hook input failed.")
            if origin is ConsoleSubmissionOrigin.MANUAL:
                outcome = await controller._legacy_submission_hook_input(
                    prompt, session_id
                )
                if outcome is not None and outcome.blocked:
                    return refused(f"Blocked by hook: {outcome.reason}")
                if outcome is not None and outcome.context:
                    hook_rows = [
                        *hook_rows,
                        {"role": "user", "content": outcome.context},
                    ]
            reason = await controller.hook_admission_reason()
            if reason is not None:
                return refused(reason)
            task = asyncio.current_task()
            owner = getattr(controller, "_hooks_v2_submissions", {}).get(task)
            if (
                controller._submit_task_session(task) != session_id
                or controller.run_state_for(session_id).status
                is not ConsoleRunStatus.VALIDATING
                or (
                    origin is ConsoleSubmissionOrigin.MANUAL
                    and controller.store.active_session_id != session_id
                )
                or (
                    owner is not None
                    and (
                        owner[2] != session_id
                        or (owner[0] is not None and not owner[0].current())
                    )
                )
            ):
                return refused("Turn changed before replay; try again.")
            return await operation(resolution, context, hook_rows)

    return await controller._submit_draft_lifecycle(
        prompt, session_id=session_id, origin=origin, _operation=replay
    )


def _matching_recovery(runtime: Any, session_id: str, echo: ConsoleChatMessage) -> Any:
    recoveries = runtime.recoveries_for_session(session_id) if runtime else ()
    return next(
        (
            entry
            for entry in reversed(recoveries)
            if validate_console_draft(entry.draft, allow_empty=True)[0] == echo.content
        ),
        None,
    )


def _pending_from_echo(echo: ConsoleChatMessage) -> tuple[PendingAttachment, ...]:
    return tuple(
        PendingAttachment(
            file_path="",
            display_name=attachment.display_name,
            file_type="image",
            insert_mode="attachment",
            data=attachment.data,
            mime_type=attachment.mime_type,
            original_size=len(attachment.data),
            processed_size=len(attachment.data),
        )
        for attachment in echo.attachments
        if attachment.data is not None
    )


async def resend_refused_echo(
    echo: ConsoleChatMessage,
    *,
    store: Any,
    runtime: Any,
    composer: Any | None,
    dispatch: Callable[[str, Any, str], Awaitable[bool]],
) -> str | None:
    """Re-send a refused, never-persisted echo through the normal send path.

    ``resend_turn`` has already checked the gates and the broken shape. The
    runtime's unsent-turn recovery for the echo, when present, is the
    exact copy of what was refused: its draft and attachment objects are used
    and the recovery is consumed, so the shelf never offers a duplicate. A
    composer already holding the same text is committed by the send, so it is
    not left holding a duplicate either. A composer holding other text, or
    staged attachments other than the echo's own, refuses the resend instead.

    Args:
        echo: The refused USER echo row.
        store: The Console chat store.
        runtime: The Console runtime (unsent-turn recoveries), or ``None``.
        composer: The echo session's visible composer, or ``None``.
        dispatch: ``(draft, stash, session_id) -> sent`` for the normal path.

    Returns:
        Refusal copy when nothing was handed to the send path, else ``None``.
    """
    session_id = store.session_id_for_message(echo.id)
    recovery = _matching_recovery(runtime, session_id, echo)
    draft = recovery.draft if recovery is not None else echo.content
    composer_text = composer.draft_text() if composer is not None else ""
    if composer_text.strip() and composer_text.strip() != draft.strip():
        return RESEND_COMPOSER_BUSY_COPY
    # Admission sends every staged attachment, so anything staged after the
    # refusal would ride along. The only staged files allowed are the echo's
    # own, put back by a shelf Restore (a live recovery still holds them).
    staged = [item.data for item in store.pending_attachments(session_id)]
    own = [item.data for item in echo.attachments if item.data is not None]
    if staged and (recovery is not None or staged != own):
        return RESEND_STAGED_ATTACHMENTS_COPY
    try:
        store.delete_message(echo.id)
    except (ValueError, RuntimeError) as exc:
        return str(exc)
    if recovery is not None:
        store.restore_transferred_pending_attachments(session_id, recovery.attachments)
        runtime.discard_turn_recovery(recovery.turn_id)
    elif not store.pending_attachments(session_id):
        store.restore_transferred_pending_attachments(
            session_id, _pending_from_echo(echo)
        )
    stash = composer.capture_draft_for_send() if composer_text.strip() else None
    try:
        sent = await dispatch(draft, stash, session_id)
    except BaseException:
        # The echo and its recovery are already gone: a cancelled or failing
        # send must not take the text with it (review I2).
        _keep_draft(store, session_id, composer, draft)
        raise
    if not sent:
        # A refused normal send keeps its draft in the composer; so does this.
        _keep_draft(store, session_id, composer, draft)
    return None


def _keep_draft(store: Any, session_id: str, composer: Any | None, draft: str) -> None:
    """Put an unsent draft back, never over text the composer holds now.

    The composer is re-read after the send-path await: it may still hold this
    draft, or text the user typed meanwhile, which always wins (review M2).
    """
    if composer is not None and composer.draft_text().strip():
        return
    store.set_session_draft(session_id, draft)
    if composer is not None:
        composer.load_draft(draft)
