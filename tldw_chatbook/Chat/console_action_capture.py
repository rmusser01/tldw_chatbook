"""Capture-On dispatch for Edit & resend, Regenerate and Continue (TASK-42560).

ADR-337 amendment 2026-11-26 (owner-approved). These actions used to dispatch
with no turn preparation, so they sent with Capture Off, and the next captured
send in a chat with project instructions was refused. Each action now freezes
the Capture-On request a normal send of its route builds: the chat's effective
capture policy, the saved rows it read before any per-send transform, and its
prompt mix-ins composed into the request. Retry is TASK-42555's and is not
routed here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
    TraceProvenancePersistenceError,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_chat_controller import (
        ConsoleChatController,
        ConsoleSubmitResult,
    )

#: A temporary chat cannot hold a durable call (ADR-337 decision 15).
TEMPORARY_CAPTURE_COPY = (
    "Capture On needs a saved chat. Save the chat or turn Capture off, then try again."
)
#: The kept reply's failure copy when a call of its run could not be traced;
#: an earlier call of the run may have been sent.
TRACE_REFUSED_COPY = "Trace capture could not be saved, so this reply stopped."
#: The same for a Regenerate, whose earlier reply is back on the branch.
REGENERATE_REFUSED_COPY = (
    "Trace capture could not be saved, so the regenerated reply stopped. "
    "The earlier reply is shown."
)
#: Why an action the trace refused before dispatch was undone (owner
#: decision 2026-11-26).
ACTION_NOT_SENT_COPY = (
    "Not sent: trace capture could not start. Choose Capture off for the next "
    "send, or turn Capture off for this chat."
)
#: The same for an Edit & resend, whose edited message is removed with it.
EDIT_NOT_SENT_COPY = ACTION_NOT_SENT_COPY + " Your edit was not kept."
#: The reply path's own copy when the mix-ins cannot be composed.
MIXIN_UNAVAILABLE_COPY = "Mix-in attempt could not be saved; response was not sent."


def saved_rows(provider_messages: Sequence[Mapping[str, Any]]) -> tuple[dict, ...]:
    """Snapshot the rows an action read, before any per-send transform."""
    return tuple(dict(row) for row in provider_messages)


def _capture_mode(
    controller: ConsoleChatController, session_id: str
) -> tuple[ConsoleTraceCaptureMode, Any]:
    """Resolve the capture mode a manual send of this chat would freeze.

    The one-time next-send choice applies, as it would to a send;
    :func:`stream_action` uses it up.
    """
    try:
        snapshot = controller.capture_policy_snapshot(session_id)
    except Exception as exc:  # noqa: BLE001 -- a normal send falls back the same way
        from loguru import logger

        logger.bind(error_type=type(exc).__name__).warning(
            "capture_policy_preparation_failed"
        )
        return ConsoleTraceCaptureMode.CAPTURE_OFF, None
    if snapshot.effective_capture_enabled and bool(
        getattr(controller.provider_gateway, "supports_durable_capture", False)
    ):
        return ConsoleTraceCaptureMode.CAPTURE_ON, snapshot
    return ConsoleTraceCaptureMode.CAPTURE_OFF, snapshot


def continues_earlier_message(
    controller: ConsoleChatController, session_id: str, message_id: str
) -> bool:
    """Whether a Continue of ``message_id`` leaves later rows out of its request.

    Its request runs through the selected message, but its reply is appended
    at the end of the chat.
    """
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    rows = controller.store.messages_for_session(session_id)
    position = next((i for i, row in enumerate(rows) if row.id == message_id), None)
    return position is None or any(
        row.role is not ConsoleMessageRole.SYSTEM for row in rows[position + 1 :]
    )


def _keeps_capture_off(
    controller: ConsoleChatController,
    session_id: str,
    resolution: Any,
    prefill: str | None,
    earlier: bool,
) -> bool:
    """Whether the action sends with capture off whatever the chat's mode.

    ponytail: each needs its own trace rule first. A captured send with a
    pinned prefill already stops at the second message
    (surface_prefix_mismatch, TASK-42583); a chat whose provider replays its
    own continuations (Kimi reasoning) carries rows the re-record's proof
    cannot name (TASK-42584); a Continue of an earlier message re-records a
    turn whose later turns it leaves out (TASK-42586).
    """
    return (
        bool(prefill)
        or earlier
        or _replays_provider_continuations(controller, session_id, resolution)
    )


def action_capture_refusal(
    controller: ConsoleChatController,
    session_id: str,
    resolution: Any,
    prefill: str | None = None,
    *,
    earlier: bool = False,
) -> str | None:
    """Return refusal copy before the action changes the transcript, if any.

    Only an action that would be traced is refused.
    """
    mode, _snapshot = _capture_mode(controller, session_id)
    if (
        mode is ConsoleTraceCaptureMode.CAPTURE_ON
        and controller.store.session_is_ephemeral(session_id)
        and not _keeps_capture_off(controller, session_id, resolution, prefill, earlier)
    ):
        return TEMPORARY_CAPTURE_COPY
    return None


def turn_user_id(
    controller: ConsoleChatController, session_id: str, assistant_message_id: str
) -> str:
    """Return the saved user row that owns the reply, as the reply path does."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    transcript = controller.store.messages_for_session(session_id)
    position = next(
        i for i, row in enumerate(transcript) if row.id == assistant_message_id
    )
    return next(
        row.id
        for row in reversed(transcript[:position])
        if row.role is ConsoleMessageRole.USER
    )


def _trace_request(
    controller: ConsoleChatController,
    snapshot: Any,
    *,
    route: ConsoleRequestRoute,
    resolution: Any,
    turn_context: Any,
    provider_messages: list[dict[str, Any]],
    trace_source_messages: tuple[dict[str, Any], ...],
    user_id: str,
) -> Any:
    """Freeze the action's Capture-On request, as a normal send freezes its own."""
    from tldw_chatbook.Chat.console_turn_preparation import (
        ConsoleTurnPreparation,
        ConsoleTurnPreparationState,
    )

    user = controller.store.get_message(user_id)
    pii = bool(snapshot.pii_redaction_enabled)
    # A policy carrier only: it is never registered or published.
    preparation = ConsoleTurnPreparation(
        preparation_id=str(uuid4()),
        attempt_id=turn_context.library_authority.attempt_id,
        session_id=turn_context.session_id,
        origin="manual",
        queue_entry_id=None,
        executed_draft=user.content,
        execution_context=turn_context,
        transient_user_message_id=None,
        # A preparation needs a draft or an attachment; an image-only turn
        # names its attachment positions.
        attachment_ids=()
        if user.content
        else tuple(f"attachment-{item.position}" for item in user.attachments),
        evidence_ids=(),
        prefill_id=None,
        queue_generation=None,
        pre_send_title=(snapshot.conversation_title or "").strip() or "Console chat",
        pre_send_conversation_id=None,
        state=ConsoleTurnPreparationState.ACCEPTED,
        pause_kind=None,
        one_shot_bypass=False,
        ephemeral=False,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
        pii_redaction_enabled=pii,
        pii_ruleset_revision_id=snapshot.pii_ruleset_revision_id if pii else None,
    )
    return controller._build_durable_trace_request(
        preparation=preparation,
        resolution=resolution,
        provider_messages=provider_messages,
        trace_source_messages=trace_source_messages,
        echoed_user_id=user_id,
        committed_user_id=user.persisted_message_id,
        route=route,
    )


def _use_up_next_send_choice(
    controller: ConsoleChatController, session_id: str, snapshot: Any
) -> None:
    """Consume the one-time Capture/PII choice the action's mode honoured."""
    if snapshot is None or (
        snapshot.next_capture_enabled is None
        and snapshot.next_pii_redaction_enabled is None
    ):
        return
    try:
        controller.store.consume_session_next_trace_privacy(
            session_id, expected_next_revision=snapshot.next_privacy_revision
        )
    except Exception as exc:  # noqa: BLE001 -- a normal send logs and goes on
        from loguru import logger

        logger.bind(
            phase="privacy_one_shot_consumption", error_type=type(exc).__name__
        ).warning("capture_policy_resolution_failed")


def _replays_provider_continuations(
    controller: ConsoleChatController, session_id: str, resolution: Any
) -> bool:
    """Whether this send would replay provider continuation checkpoints."""
    try:
        sidecars, target = controller._provider_continuation_history_for_resolution(
            session_id, resolution
        )
    except Exception:  # noqa: BLE001 -- the stream refuses a conflict itself
        return True
    return bool(sidecars) and target is not None


async def stream_action(
    controller: ConsoleChatController,
    *,
    trace_source_messages: Sequence[Mapping[str, Any]],
    user_id: str,
    earlier: bool = False,
    previous_leaf_id: str | None = None,
    note_ids: tuple[str, ...] = (),
    **stream: Any,
) -> ConsoleSubmitResult:
    """Stream an action's reply, traced under Capture On like a normal send.

    Args:
        controller: The Console controller dispatching the action.
        trace_source_messages: The saved rows as read, before transforms.
        user_id: Native id of the saved user row that owns the call.
        earlier: A Continue of a message with later rows after it.
        previous_leaf_id: The active leaf before the action added any row.
        note_ids: Transcript-only skipped-skill notes the action added.
        **stream: The action's ``_stream_assistant_response`` arguments.

    Returns:
        The stream's result. When the trace refuses the action before
        anything is sent, the action is undone (:func:`action_dropped`); a
        reply that cannot be undone fails with :func:`kept_reply_copy`.
        A mix-in composition failure is blocked with
        :data:`MIXIN_UNAVAILABLE_COPY`.
    """
    from tldw_chatbook.Chat.console_prepared_request import (
        compose_current_mixin_messages,
    )
    from tldw_chatbook.Chat.console_prompt_mixin_models import (
        MixinSnapshotUnavailableError,
    )
    from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError

    turn_context = stream["turn_context"]
    session_id = turn_context.session_id
    mode, snapshot = _capture_mode(controller, session_id)
    excepted = controller.store.session_is_ephemeral(session_id) or _keeps_capture_off(
        controller, session_id, stream["resolution"], stream.get("prefill"), earlier
    )
    if mode is ConsoleTraceCaptureMode.CAPTURE_OFF or excepted:
        if mode is ConsoleTraceCaptureMode.CAPTURE_OFF:
            # The action follows the chat's capture mode, so it uses up a
            # one-time choice as a normal send does.
            _use_up_next_send_choice(controller, session_id, snapshot)
        return await controller._stream_assistant_response(**stream)
    assistant_id = stream["assistant_message_id"]
    # An Edit & resend also created its edited message.
    root_id = user_id if stream["route"] is ConsoleRequestRoute.EDIT else assistant_id
    kept_copy = kept_reply_copy(stream["route"])
    undo = _Undo(
        root_id=root_id,
        previous_leaf_id=previous_leaf_id,
        note_ids=note_ids,
        conversation_id=next(
            (
                row.persisted_conversation_id
                for row in controller.store.sessions()
                if row.id == session_id
            ),
            None,
        ),
        turn_id=controller.store.get_message(user_id).persisted_message_id,
        calls_before=None,
        copy=EDIT_NOT_SENT_COPY if root_id == user_id else ACTION_NOT_SENT_COPY,
        kept_copy=kept_copy,
    )
    undo = replace(undo, calls_before=_turn_calls(controller, undo))
    try:
        # The reply path composes the mix-ins too; composing them first
        # freezes the request with the body the provider receives.
        provider_messages = compose_current_mixin_messages(
            stream["provider_messages"],
            turn_context.prompt_mixins,
            user_message_id=stream.get("mixin_anchor_id") or user_id,
            attempt_id=turn_context.library_authority.attempt_id,
        )
        request = _trace_request(
            controller,
            snapshot,
            route=stream["route"],
            resolution=stream["resolution"],
            turn_context=turn_context,
            provider_messages=provider_messages,
            trace_source_messages=tuple(trace_source_messages),
            user_id=user_id,
        )
    except MixinSnapshotUnavailableError:
        # As the reply path refuses its own composition; nothing was sent, so
        # a one-time choice stays armed.
        return controller._block_context_preflight(
            session_id=session_id,
            assistant_message_id=assistant_id,
            visible_copy=MIXIN_UNAVAILABLE_COPY,
        )
    except Exception as exc:  # noqa: BLE001 -- a normal send refuses the same way
        from loguru import logger

        logger.bind(error_type=type(exc).__name__).warning(
            "action_trace_request_failed"
        )
        return _refused(controller, session_id, snapshot, assistant_id, undo)
    try:
        result = await controller._stream_assistant_response(
            **(stream | {"provider_messages": provider_messages}),
            capture_mode_override=mode,
            trace_request=request,
            # The agent path raises a refused call's error instead of
            # settling it, so a refusal before dispatch can be undone.
            propagate_trace_call_persistence_errors=True,
        )
    except TraceProvenancePersistenceError:
        return _refused(controller, session_id, snapshot, assistant_id, undo)
    except TraceCallPersistenceError as exc:
        if _dispatch_may_have_started(exc):
            _use_up_next_send_choice(controller, session_id, snapshot)
            return _fail_unsent(controller, session_id, assistant_id, kept_copy)
        return _refused(controller, session_id, snapshot, assistant_id, undo, exc)
    except BaseException:
        _use_up_next_send_choice(controller, session_id, snapshot)
        raise
    _use_up_next_send_choice(controller, session_id, snapshot)
    return result


@dataclass(frozen=True, slots=True)
class _Undo:
    """What a refused action created, and the branch before it."""

    root_id: str
    previous_leaf_id: str | None
    note_ids: tuple[str, ...]
    conversation_id: str | None
    turn_id: str | None
    #: The turn's trace calls before the action; None if unreadable.
    calls_before: frozenset[str] | None
    copy: str
    kept_copy: str


#: The turn's calls, through the owner (``idx_console_trace_calls_owner_order``).
TURN_CALLS_SQL = (
    "SELECT c.call_id FROM console_trace_owners o JOIN console_trace_calls c "
    "ON c.owner_id = o.owner_id WHERE o.conversation_id = ? AND c.turn_id = ?"
)


def _turn_calls(
    controller: ConsoleChatController, undo: _Undo, *, dispatched: bool = False
) -> frozenset[str] | None:
    """The turn's trace call ids (None if the ledger cannot be read, which
    keeps a refused reply)."""
    database = getattr(getattr(controller.store, "persistence", None), "db", None)
    if database is None or undo.conversation_id is None or undo.turn_id is None:
        return None
    try:
        with database.transaction() as cursor:
            rows = cursor.execute(
                TURN_CALLS_SQL
                + (" AND c.dispatch_started_at IS NOT NULL" if dispatched else ""),
                (undo.conversation_id, undo.turn_id),
            ).fetchall()
    except Exception:  # noqa: BLE001 -- unreadable: keep the reply
        return None
    return frozenset(str(row[0]) for row in rows)


def _since(controller: ConsoleChatController, undo: _Undo, **kwargs: bool) -> bool:
    """Whether a call of the action's turn was reserved since it began (a
    call of another chat or turn never counts); True if unreadable."""
    calls = _turn_calls(controller, undo, **kwargs)
    return calls is None or undo.calls_before is None or bool(calls - undo.calls_before)


def _settle_refused_call(
    controller: ConsoleChatController, undo: _Undo, exc: Exception | None
) -> bool:
    """Mark a refused call's established reservation not dispatched, as a
    canceled trace pause does; False when that cannot be proven."""
    boundary = getattr(exc, "boundary", None)
    if boundary is not None:
        status = getattr(boundary, "reservation_status", "unknown")
        if status == "established":
            try:
                boundary.mark_not_dispatched()
            except Exception:  # noqa: BLE001 -- unsaved: keep the reply
                return False
            return True
        if status != "not_established":
            return False
    # The refusal holds no reservation. A factory that failed after its own
    # commit leaves one anyway, of unknown state: keep the reply then.
    return not _since(controller, undo)


def _dispatch_may_have_started(exc: Exception) -> bool:
    """Read the refused call's dispatch evidence as a normal send does."""
    boundary = getattr(exc, "boundary", None)
    if getattr(boundary, "dispatch_outcome", None) == "unknown":
        return True
    started = getattr(boundary, "dispatch_started", None)
    if type(started) is bool:
        return started
    return getattr(exc, "reservation_status", None) is None


def _holds_nothing(controller: ConsoleChatController, message_id: str) -> bool:
    try:
        reply = controller.store.get_message(message_id)
    except KeyError:
        return False
    return not (
        reply.content
        or reply.image_data is not None
        or reply.attachments
        or reply.provider_continuation is not None
        or reply.thinking is not None
        or reply.opaque_thinking_json
        or reply.tool_output_full
        or reply.tool_diff
    )


def _refused(
    controller: ConsoleChatController,
    session_id: str,
    snapshot: Any,
    assistant_id: str,
    undo: _Undo,
    exc: Exception | None = None,
) -> ConsoleSubmitResult:
    """Undo an action the trace refused before anything was sent (owner
    decision 2026-11-26), or fail its reply when that cannot be proven."""
    from loguru import logger

    from tldw_chatbook.Chat.console_chat_controller import (
        ConsoleRunState,
        ConsoleRunStatus,
        ConsoleSubmitResult,
    )

    settled = _settle_refused_call(controller, undo, exc)
    dispatched = _since(controller, undo, dispatched=True)
    if not settled or dispatched or not _holds_nothing(controller, assistant_id):
        if dispatched:  # a call of the turn reached the provider
            _use_up_next_send_choice(controller, session_id, snapshot)
        return _fail_unsent(controller, session_id, assistant_id, undo.kept_copy)
    try:
        # A pending reply cannot be deleted; it never streamed.
        controller.store.mark_message_failed(assistant_id)
        if undo.previous_leaf_id is not None:
            controller.store.set_active_leaf(session_id, undo.previous_leaf_id)
        controller.store.delete_message(undo.root_id)
    except KeyError:
        return controller._session_closed_result(session_id=session_id)
    except Exception as error:  # noqa: BLE001 -- settle the run whatever failed
        logger.bind(error_type=type(error).__name__).warning("action_undo_failed")
        try:  # the kept reply fails on the branch it is on
            controller.store.set_active_leaf(session_id, assistant_id)
        except Exception:  # noqa: BLE001 -- fail it where it is
            pass
        return _fail_unsent(controller, session_id, assistant_id, undo.kept_copy)
    for note_id in undo.note_ids[:1]:  # each note hangs off the one before
        try:
            controller.store.delete_message(note_id)
        except Exception as error:  # noqa: BLE001 -- the note stays off the branch
            logger.bind(error_type=type(error).__name__).warning("action_note_kept")
    controller._set_run_state(ConsoleRunState.blocked(undo.copy), session_id=session_id)
    return ConsoleSubmitResult(
        False,
        False,
        undo.copy,
        session_id=session_id,
        terminal_status=ConsoleRunStatus.BLOCKED,
    )


def take_original_attempts(
    controller: ConsoleChatController, message_ids: Iterable[str]
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    """Clear the citation-repair previews of the messages an action
    replaces, as before it streams, and return them, with the cache's order,
    for an undo."""
    message_ids = tuple(message_ids)
    order = tuple(controller._original_attempts)
    taken = tuple(
        (message_id, body)
        for message_id in message_ids
        if (body := controller._original_attempts.get(message_id)) is not None
    )
    for message_id in message_ids:
        controller.clear_original_attempt(message_id)
    return taken, order


def restore_original_attempts(
    controller: ConsoleChatController,
    taken: tuple[tuple[tuple[str, str], ...], tuple[str, ...]],
) -> None:
    """Put back the previews an undone action cleared, in their order."""
    entries, order = taken
    for message_id, body in entries:
        controller._remember_original_attempt(message_id, body)
    rank = {message_id: index for index, message_id in enumerate(order)}
    for message_id in sorted(
        controller._original_attempts, key=lambda key: rank.get(key, len(rank))
    ):
        controller._original_attempts.move_to_end(message_id)


def action_dropped(result: ConsoleSubmitResult | None) -> bool:
    """Whether ``stream_action`` undid a refused action."""
    return (
        result is not None
        and not result.accepted
        and result.visible_copy
        in {
            ACTION_NOT_SENT_COPY,
            EDIT_NOT_SENT_COPY,
        }
    )


def kept_reply_copy(route: ConsoleRequestRoute | str) -> str:
    """The failure copy of an action's kept reply."""
    return (
        REGENERATE_REFUSED_COPY
        if getattr(route, "value", route) == ConsoleRequestRoute.REGENERATE.value
        else TRACE_REFUSED_COPY
    )


def _fail_unsent(
    controller: ConsoleChatController,
    session_id: str,
    assistant_id: str,
    copy: str = TRACE_REFUSED_COPY,
) -> ConsoleSubmitResult:
    """Keep and fail the reply of an action the trace refused."""
    from loguru import logger

    from tldw_chatbook.Chat.console_chat_controller import (
        ConsoleRunState,
        ConsoleRunStatus,
        ConsoleSubmitResult,
    )

    # The agent path settles its run the same way; the reply fails and keeps
    # Retry. The run state is settled whatever else fails.
    try:
        controller.store.mark_message_failed(assistant_id)
    except KeyError:
        if not any(row.id == session_id for row in controller.store.sessions()):
            return controller._session_closed_result(session_id=session_id)
    except Exception as error:  # noqa: BLE001
        logger.bind(error_type=type(error).__name__).warning("action_reply_not_failed")
    try:
        controller._append_failure_system_row(session_id, copy)
    except Exception as error:  # noqa: BLE001
        logger.bind(error_type=type(error).__name__).warning("action_reply_not_failed")
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.FAILED, copy),
        session_id=session_id,
    )
    return ConsoleSubmitResult(True, False, copy)


__all__ = [
    "ACTION_NOT_SENT_COPY",
    "EDIT_NOT_SENT_COPY",
    "MIXIN_UNAVAILABLE_COPY",
    "REGENERATE_REFUSED_COPY",
    "TEMPORARY_CAPTURE_COPY",
    "TRACE_REFUSED_COPY",
    "action_capture_refusal",
    "action_dropped",
    "kept_reply_copy",
    "restore_original_attempts",
    "take_original_attempts",
    "continues_earlier_message",
    "saved_rows",
    "stream_action",
    "turn_user_id",
]
