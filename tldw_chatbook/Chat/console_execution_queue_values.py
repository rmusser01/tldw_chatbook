"""Finite original queue facts; acknowledgement and collapse remain view-owned."""

from dataclasses import dataclass

from .console_chat_models import ConsoleControllerActivity
from .console_execution_projection_values import (
    _boolean,
    _count,
    _enum_record,
    _optional,
    _record,
    _text,
    _tuple,
)
from .console_prompt_queue import (
    PromptQueueEntryPhase,
    PromptQueueEntrySnapshot,
    PromptQueueMode,
    PromptQueuePauseReason,
    PromptQueueReservation,
    PromptQueueSnapshot,
)


@dataclass(frozen=True, slots=True)
class ConsoleQueueDisplay:
    snapshot: PromptQueueSnapshot
    activity: ConsoleControllerActivity
    dispatch_recovery_blocks_queue: bool
    dispatch_recovery_blocks_submission: bool
    turn_recovery_id: str | None
    turn_recovery_reason: str
    failed_turn_preview: str | None
    received_preparing: bool
    chat_start_prepared: bool
    chat_start_accepted: bool
    run_stop_allowed: bool
    wake_turn_active: bool


def _entry(value, decode):
    return _record(
        value,
        PromptQueueEntrySnapshot,
        {
            "entry_id": _text,
            "preview": _text,
            "insertion_order": _count,
            "position": _count,
            "phase": lambda v, d: _enum_record(v, PromptQueueEntryPhase, d),
        },
        decode,
    )


def _snapshot(value, decode):
    return _record(
        value,
        PromptQueueSnapshot,
        {
            "session_id": _text,
            "revision": _count,
            "entries": _tuple(_entry),
            "waiting_count": _count,
            "claimed_count": _count,
            "total_count": _count,
            "mode": lambda v, d: _enum_record(v, PromptQueueMode, d),
            "pause_reason": _optional(
                lambda v, d: _enum_record(v, PromptQueuePauseReason, d)
            ),
            "reservation": lambda v, d: _enum_record(v, PromptQueueReservation, d),
            "expected_context_epoch": _optional(_count),
            "closing": _boolean,
        },
        decode,
    )


def _activity(value, decode):
    return _record(
        value,
        ConsoleControllerActivity,
        {
            "session_id": _text,
            "occupies_slot": _boolean,
            "preparing_before_acceptance": _boolean,
            "accepted_live_turn": _boolean,
            "needs_approval": _boolean,
            "queued_count": _count,
            "queue_paused": _boolean,
            "terminal_notification_eligible": _boolean,
            "agent_handoff_status": _text,
        },
        decode,
    )


def queue_display(value, decode):
    result = _record(
        value,
        ConsoleQueueDisplay,
        {
            "snapshot": _snapshot,
            "activity": _activity,
            "dispatch_recovery_blocks_queue": _boolean,
            "dispatch_recovery_blocks_submission": _boolean,
            "turn_recovery_id": _optional(_text),
            "turn_recovery_reason": _text,
            "failed_turn_preview": _optional(_text),
            "received_preparing": _boolean,
            "chat_start_prepared": _boolean,
            "chat_start_accepted": _boolean,
            "run_stop_allowed": _boolean,
            "wake_turn_active": _boolean,
        },
        decode,
    )
    captured = result if decode else value
    if captured.snapshot.session_id != captured.activity.session_id:
        from .console_execution_protocol import ConsoleProtocolError

        raise ConsoleProtocolError("queue_display_session_changed")
    return result


def queue_presentation(value: ConsoleQueueDisplay, *, sending: bool, collapsed: bool):
    """Use the same original vocabulary and precedence as the local queue UI."""
    from ..UI.Console_Modules.prompt_queue import (
        derive_prompt_queue_presentation,
        with_prompt_queue_start_state,
    )

    presentation = derive_prompt_queue_presentation(
        value.snapshot,
        value.activity,
        composer_collapsed=collapsed,
        dispatch_recovery_blocked=value.dispatch_recovery_blocks_queue,
        turn_recovery_id=value.turn_recovery_id,
        turn_recovery_reason=value.turn_recovery_reason,
        failed_turn_preview=value.failed_turn_preview,
        sending=sending,
    )
    return with_prompt_queue_start_state(
        presentation,
        received_preparing=value.received_preparing,
        prepared=value.chat_start_prepared,
        accepted=value.chat_start_accepted,
    )
