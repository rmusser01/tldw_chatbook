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


def saved_rows(provider_messages: Sequence[Mapping[str, Any]]) -> tuple[dict, ...]:
    """Snapshot the rows an action read, before any per-send transform."""
    return tuple(dict(row) for row in provider_messages)


def _capture_mode(
    controller: ConsoleChatController, session_id: str
) -> tuple[ConsoleTraceCaptureMode, Any]:
    """Resolve the capture mode a manual send of this chat would freeze.

    The one-time next-send choice applies, as it would to a send, but stays
    armed for the next send.
    """
    try:
        snapshot = controller.capture_policy_snapshot(session_id)
    except Exception:  # noqa: BLE001 -- a normal send falls back the same way
        return ConsoleTraceCaptureMode.CAPTURE_OFF, None
    if snapshot.effective_capture_enabled and bool(
        getattr(controller.provider_gateway, "supports_durable_capture", False)
    ):
        return ConsoleTraceCaptureMode.CAPTURE_ON, snapshot
    return ConsoleTraceCaptureMode.CAPTURE_OFF, snapshot


def action_capture_refusal(
    controller: ConsoleChatController, session_id: str
) -> str | None:
    """Return refusal copy before the action changes the transcript, if any."""
    mode, _snapshot = _capture_mode(controller, session_id)
    if (
        mode is ConsoleTraceCaptureMode.CAPTURE_ON
        and controller.store.session_is_ephemeral(session_id)
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


def _replays_provider_continuations(
    controller: ConsoleChatController, session_id: str, resolution: Any
) -> bool:
    """Whether this send would replay provider continuation checkpoints."""
    try:
        sidecars, target = controller._provider_continuation_history_for_resolution(
            session_id, resolution
        )
    except Exception:  # noqa: BLE001 -- the stream refuses a conflict itself
        return False
    return bool(sidecars) and target is not None


async def stream_action(
    controller: ConsoleChatController,
    *,
    trace_source_messages: Sequence[Mapping[str, Any]],
    user_id: str,
    **stream: Any,
) -> ConsoleSubmitResult:
    """Stream an action's reply, traced under Capture On like a normal send.

    Args:
        controller: The Console controller dispatching the action.
        trace_source_messages: The saved rows as read, before transforms.
        user_id: Native id of the saved user row that owns the call.
        **stream: The action's ``_stream_assistant_response`` arguments.

    Returns:
        The stream's result. A request the trace refuses before dispatch
        fails the reply with :data:`TRACE_REFUSED_COPY`.
    """
    from tldw_chatbook.Chat.console_chat_controller import (
        ConsoleRunState,
        ConsoleRunStatus,
        ConsoleSubmitResult,
    )
    from tldw_chatbook.Chat.console_prepared_request import (
        compose_current_mixin_messages,
    )
    from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError

    turn_context = stream["turn_context"]
    session_id = turn_context.session_id
    mode, snapshot = _capture_mode(controller, session_id)
    if (
        mode is ConsoleTraceCaptureMode.CAPTURE_OFF
        or controller.store.session_is_ephemeral(session_id)
        # ponytail: these keep the action Capture Off, as before. A captured
        # send with a pinned prefill already stops at the second message
        # (surface_prefix_mismatch), and a chat whose provider replays its
        # own continuations (Kimi reasoning) carries rows the re-record's
        # proof cannot name; each needs its own rule first.
        or stream.get("prefill")
        or _replays_provider_continuations(controller, session_id, stream["resolution"])
    ):
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
        return await controller._stream_assistant_response(
            **(stream | {"provider_messages": provider_messages}),
            capture_mode_override=mode,
            trace_request=request,
        )
    except (TraceCallPersistenceError, TraceProvenancePersistenceError):
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
    "TEMPORARY_CAPTURE_COPY",
    "TRACE_REFUSED_COPY",
    "action_capture_refusal",
    "saved_rows",
    "stream_action",
    "turn_user_id",
]
