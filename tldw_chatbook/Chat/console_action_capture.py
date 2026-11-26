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

from collections.abc import Mapping, Sequence
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
#: The reply's failure copy when its call could not be traced.
TRACE_REFUSED_COPY = "Trace capture could not be saved, so this request was not sent."
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
    **stream: Any,
) -> ConsoleSubmitResult:
    """Stream an action's reply, traced under Capture On like a normal send.

    Args:
        controller: The Console controller dispatching the action.
        trace_source_messages: The saved rows as read, before transforms.
        user_id: Native id of the saved user row that owns the call.
        earlier: A Continue of a message with later rows after it.
        **stream: The action's ``_stream_assistant_response`` arguments.

    Returns:
        The stream's result. A request the trace refuses before dispatch
        fails the reply with :data:`TRACE_REFUSED_COPY`, or blocks it with
        :data:`MIXIN_UNAVAILABLE_COPY` when its mix-ins cannot be composed.
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
    if mode is ConsoleTraceCaptureMode.CAPTURE_OFF or not excepted:
        # The action follows the chat's capture mode, so it uses up a
        # one-time choice as a normal send does.
        _use_up_next_send_choice(controller, session_id, snapshot)
    if mode is ConsoleTraceCaptureMode.CAPTURE_OFF or excepted:
        return await controller._stream_assistant_response(**stream)
    assistant_id = stream["assistant_message_id"]
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
        # As the reply path refuses its own composition.
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
        return _fail_unsent(controller, session_id, assistant_id)
    try:
        return await controller._stream_assistant_response(
            **(stream | {"provider_messages": provider_messages}),
            capture_mode_override=mode,
            trace_request=request,
        )
    except (TraceCallPersistenceError, TraceProvenancePersistenceError):
        return _fail_unsent(controller, session_id, assistant_id)


def _fail_unsent(
    controller: ConsoleChatController, session_id: str, assistant_id: str
) -> ConsoleSubmitResult:
    """Fail the reply of a request the trace refused before dispatch."""
    from tldw_chatbook.Chat.console_chat_controller import (
        ConsoleRunState,
        ConsoleRunStatus,
        ConsoleSubmitResult,
    )

    # Nothing reached the provider. The agent path settles its run the
    # same way; the reply fails and keeps Retry.
    try:
        controller.store.mark_message_failed(assistant_id)
    except KeyError:
        return controller._session_closed_result(session_id=session_id)
    controller._append_failure_system_row(session_id, TRACE_REFUSED_COPY)
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.FAILED, TRACE_REFUSED_COPY),
        session_id=session_id,
    )
    return ConsoleSubmitResult(True, False, TRACE_REFUSED_COPY)


__all__ = [
    "MIXIN_UNAVAILABLE_COPY",
    "TEMPORARY_CAPTURE_COPY",
    "TRACE_REFUSED_COPY",
    "action_capture_refusal",
    "continues_earlier_message",
    "saved_rows",
    "stream_action",
    "turn_user_id",
]
