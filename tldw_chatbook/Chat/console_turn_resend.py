"""Resend a broken last Console turn in place (TASK-33661).

A turn is *broken* when its send was refused before acceptance, it got no
assistant reply, its reply failed, or its reply is empty and stopped,
discarded, or restored after a restart as "Response failed.". Only the last
user turn on the active path qualifies: a failed reply higher up keeps its own
Retry, and a partial (non-empty) stopped reply has Continue. A turn that holds
any text an earlier reply produced (a Continue chain) or any tool output is
partial, never broken: Resend would discard that work.

Resend never forks: no ``create_sibling``, no ``edit_and_resend_message``.
A failed live reply is retried in place on the same assistant row. Every other
persisted shape clears the empty reply and the stale failure or stop rows (a
subtree tombstone), then re-runs from the same user message with the gates a
normal send applies. A refused echo was never persisted, so it is removed and
its text and attachments go back through the normal send path.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any

from tldw_chatbook.Chat.attachment_core import PendingAttachment, vision_block_reason
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_DISPATCH_DISCARDED_COPY,
    ConsoleChatMessage,
    ConsoleMessageRole,
)
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


def is_refused_echo(message: ConsoleChatMessage) -> bool:
    """Whether ``message`` is a USER echo refused before the send was accepted."""
    return (
        message.role is ConsoleMessageRole.USER
        and message.status == "failed"
        and message.persisted_message_id is None
    )


def _reply_text(message: ConsoleChatMessage) -> str:
    text = message.content.strip()
    if (
        message.assistant_generation_state == "discarded"
        and text == CONSOLE_DISPATCH_DISCARDED_COPY
    ):
        return ""
    return text


def resend_target_id(messages: Sequence[ConsoleChatMessage]) -> str | None:
    """Return the last USER row's id when its turn is broken, else ``None``.

    Args:
        messages: The session's active-path rows, oldest first.

    Returns:
        The id of the user message Resend re-runs, or ``None`` when the last
        turn is healthy, partial, still running, or has no user message. An
        unpersisted user row with no reply is an in-flight send (validating,
        or paused for preparation), never a broken one. Text from an earlier
        reply of the turn, or any tool output, makes the turn partial: the
        clear would tombstone it (review C1/I1).
    """
    index = next(
        (
            position
            for position in range(len(messages) - 1, -1, -1)
            if messages[position].role is ConsoleMessageRole.USER
        ),
        None,
    )
    if index is None:
        return None
    user = messages[index]
    replies = [
        row for row in messages[index + 1 :] if row.role is ConsoleMessageRole.ASSISTANT
    ]
    if is_refused_echo(user):
        return None if replies else user.id
    if user.status != "complete":
        return None
    tool_output = any(
        row.role is ConsoleMessageRole.TOOL and row.content.strip()
        for row in messages[index + 1 :]
    )
    if not replies:
        persisted = user.persisted_message_id is not None
        return user.id if persisted and not tool_output else None
    last = replies[-1]
    if last.status in {"pending", "streaming"}:
        return None
    if any(_reply_text(reply) for reply in replies[:-1]):
        return None
    if last.status == "failed" or last.assistant_generation_state == "failed":
        return user.id
    ended_empty = (
        last.status == "stopped"
        or last.assistant_generation_state in {"stopped", "discarded"}
    ) and not (_reply_text(last) or tool_output)
    return user.id if ended_empty else None


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

    Only some gates run before anything is cleared: the send-refusal copy (a
    live run, a queue, an unresolved dispatch recovery), the broken-turn
    check, and the vision gate for the turn's own attachments. A refused echo
    then goes to ``resend_echo`` (the normal send path, with its own gates). A
    failed reply's trailing rows are cleared and it is retried in place;
    otherwise the empty reply and the rows after the user message are
    tombstoned and the turn re-runs from the user message. Readiness, skill
    refusal, the thinking preflight and the maintenance (backup) pause run
    inside ``retry_message``/``continue_from_message``, after that clear: a
    refusal there leaves the turn still broken (Resend stays on offer) but the
    cleared rows stay cleared. In a temporary chat nothing is persisted, so a
    cleared empty reply is simply gone.

    Args:
        controller: The Console chat controller owning the session.
        message_id: The broken turn's user message id.
        resend_echo: Re-sends a refused echo; returns refusal copy or None.

    Returns:
        The re-run's submit result; a refusal carries its visible copy.
    """
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
        configuration = controller.resolve_turn_configuration_snapshot(session_id)
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
    in_place = reply is not None and reply.status == "failed"
    anchor_id = reply.id if in_place else message_id
    refusal = _delete_rows_after(store, session_id, anchor_id)
    if refusal:
        return ConsoleSubmitResult(False, False, refusal)
    if in_place:
        return await controller.retry_message(reply.id)
    return await controller.continue_from_message(message_id, resend=True)


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
    not left holding a duplicate either. A composer holding other text is
    never overwritten.

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
