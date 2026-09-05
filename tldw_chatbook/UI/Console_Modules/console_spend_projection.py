"""Cheap, UI-neutral projections for Console current and next-send spend."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ...Chat.assistant_generation_state import assistant_state_allows_provider_history
from ...Chat.console_chat_controller import _is_empty_transcript_row
from ...Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleDispatchRecoveryKind,
    ConsoleDispatchRecoveryState,
    ConsoleMessageRole,
    ConsoleRunStatus,
    fold_greeting_into_system_prompt,
)
from ...Chat.console_cost_tracker import (
    ConsoleCacheState,
    ConsoleCostSnapshot,
    ConsoleCostState,
    build_cost_state,
)
from ...Chat.console_turn_preparation import (
    ConsoleTurnPreparation,
    ConsoleTurnPreparationState,
)
from ...Chat.provider_continuation import ProviderContinuationCheckpoint
from ...Utils.input_validation import validate_console_draft
from ...Widgets.Console.console_context_controls import (
    ConsoleContextControlState,
    ConsoleNextSendSpendState,
    build_console_context_cost_state,
    build_console_next_send_spend_state,
)

_ACCEPTED_RECOVERY_KINDS = frozenset(
    {
        ConsoleDispatchRecoveryKind.ACCEPTED,
        ConsoleDispatchRecoveryKind.DISPATCH_STARTED,
        ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED,
        ConsoleDispatchRecoveryKind.EPHEMERAL_DISPATCH_STARTED,
        ConsoleDispatchRecoveryKind.REMOTE_ACCEPTED,
        ConsoleDispatchRecoveryKind.REMOTE_DISPATCH_STARTED,
    }
)
_ACCEPTED_PREPARATION_STATES = frozenset(
    {
        ConsoleTurnPreparationState.ACCEPTED,
        ConsoleTurnPreparationState.DISPATCH_STARTED,
        ConsoleTurnPreparationState.DISPATCHED,
    }
)
_REMOTE_RECOVERY_KINDS = frozenset(
    {
        ConsoleDispatchRecoveryKind.REMOTE_ACCEPTED,
        ConsoleDispatchRecoveryKind.REMOTE_DISPATCH_STARTED,
    }
)


def fold_system_prompt(system_prompt: str | None, greeting: str) -> str:
    """Fold the seeded greeting exactly as the provider send path does."""
    return fold_greeting_into_system_prompt(system_prompt or "", greeting)


@dataclass(frozen=True, slots=True)
class ConsoleSpendHistoryProjection:
    """Message ownership split for the next request and realized Current."""

    request_ids: frozenset[str]
    current_ids: frozenset[str]


def _remote_active_user_id(
    messages: Sequence[ConsoleChatMessage],
    recovery: ConsoleDispatchRecoveryState | None,
) -> str | None:
    if recovery is None or recovery.kind not in _REMOTE_RECOVERY_KINDS:
        return None
    assistant = next(
        (
            message
            for message in messages
            if message.role is ConsoleMessageRole.ASSISTANT
            and message.persisted_message_id == recovery.assistant_message_id
        ),
        None,
    )
    if assistant is None or assistant.parent_message_id is None:
        return None
    return next(
        (
            message.id
            for message in messages
            if message.role is ConsoleMessageRole.USER
            and message.persisted_message_id == assistant.parent_message_id
        ),
        None,
    )


def build_console_spend_history_projection(
    messages: Sequence[ConsoleChatMessage],
    recovery: ConsoleDispatchRecoveryState | None,
    preparation: ConsoleTurnPreparation | None,
    run_status: ConsoleRunStatus,
    has_submit_task: bool,
) -> ConsoleSpendHistoryProjection:
    """Project provider-request and billed-history rows without reading media."""
    request_excluded_user_id: str | None = None
    current_excluded_user_id: str | None = None
    checkpoint = recovery.checkpoint if recovery is not None else None
    remote_user = _remote_active_user_id(messages, recovery)
    if checkpoint is not None:
        current_excluded_user_id = checkpoint.user_message_id
        if recovery.kind not in _ACCEPTED_RECOVERY_KINDS:
            request_excluded_user_id = checkpoint.user_message_id
    elif remote_user is not None:
        current_excluded_user_id = remote_user
    elif preparation is not None and preparation.transient_user_message_id is not None:
        current_excluded_user_id = preparation.transient_user_message_id
        if preparation.state not in _ACCEPTED_PREPARATION_STATES:
            request_excluded_user_id = preparation.transient_user_message_id
    elif (
        run_status is ConsoleRunStatus.VALIDATING
        and has_submit_task
        and messages
        and messages[-1].role is ConsoleMessageRole.USER
    ):
        request_excluded_user_id = current_excluded_user_id = messages[-1].id

    request_ids: set[str] = set()
    current_ids: set[str] = set()
    seen_user = False
    for message in messages:
        if message.role not in {ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT}:
            continue
        excluded_request = message.id == request_excluded_user_id
        excluded_current = message.id == current_excluded_user_id
        if message.status == "failed":
            if (
                not excluded_current
                and message.role is ConsoleMessageRole.ASSISTANT
                and seen_user
            ):
                current_ids.add(message.id)
            continue
        if _is_empty_transcript_row(message):
            continue
        if not seen_user and message.role is ConsoleMessageRole.ASSISTANT:
            continue
        if message.role is ConsoleMessageRole.USER:
            seen_user = True
        provider_eligible = not (
            message.role is ConsoleMessageRole.ASSISTANT
            and not assistant_state_allows_provider_history(
                state=message.assistant_generation_state,
                has_valid_continuation=(
                    isinstance(
                        message.provider_continuation, ProviderContinuationCheckpoint
                    )
                    and message.provider_continuation.state == "active"
                ),
                content=message.content,
            )
        )
        if (
            not excluded_current
            and message.role is ConsoleMessageRole.ASSISTANT
            and message.status == "stopped"
        ):
            current_ids.add(message.id)
        if not provider_eligible:
            continue
        if not excluded_request:
            request_ids.add(message.id)
        if not excluded_current:
            current_ids.add(message.id)
    return ConsoleSpendHistoryProjection(frozenset(request_ids), frozenset(current_ids))


def build_console_current_cost_messages(
    messages: Sequence[ConsoleChatMessage],
    current_ids: Collection[str],
) -> list[ConsoleChatMessage]:
    """Return settled cost rows without estimating input already in real usage."""
    rows = [
        message
        for message in messages
        if message.id in current_ids
        and getattr(message, "status", "complete") not in {"pending", "streaming"}
    ]
    accounted_users: set[str] = set()
    current_user: ConsoleChatMessage | None = None
    for message in rows:
        if message.role is ConsoleMessageRole.USER:
            current_user = message
        elif (
            message.role is ConsoleMessageRole.ASSISTANT
            and message.usage is not None
            and current_user is not None
        ):
            accounted_users.add(current_user.id)
    return [message for message in rows if message.id not in accounted_users]


def build_console_context_messages(
    messages: Sequence[ConsoleChatMessage],
    request_ids: Collection[str] | None,
    draft_text: str,
) -> list[dict[str, str]]:
    """Return lifecycle-filtered text rows plus the mounted draft."""
    rows = [
        {
            "role": str(getattr(message.role, "value", message.role)),
            "content": message.content,
        }
        for message in messages
        if request_ids is None or message.id in request_ids
    ]
    if draft_text.strip():
        rows.append({"role": "user", "content": draft_text})
    return rows


def build_console_next_send_projection(
    messages: Sequence[ConsoleChatMessage],
    has_pending_attachments: bool,
    request_tokens: int | None,
    input_per_mtok: float | None,
    draft_text: str,
) -> ConsoleNextSendSpendState:
    """Build the input-only forecast from text and attachment metadata."""
    validated_draft, validation_error = validate_console_draft(
        draft_text, allow_empty=True
    )
    if validation_error is not None:
        return ConsoleNextSendSpendState(
            "unavailable",
            "On next send: unavailable\n"
            "This message cannot be sent until the draft is corrected.",
        )
    has_media = has_pending_attachments or any(
        message.attachments or message.image_data is not None for message in messages
    )
    return build_console_next_send_spend_state(
        request_tokens=request_tokens,
        input_per_mtok=input_per_mtok,
        sendable_text=bool(validated_draft.strip()),
        has_media=has_media,
    )


def build_console_spend_cost_state(
    snapshot: ConsoleCostSnapshot,
    cache_state: ConsoleCacheState,
    break_reason: str | None,
    projected_delta_usd: float | None,
    ttl_remaining_s: float | None,
    pricing_as_of: str | None,
    pricing_available: bool,
    context_state: ConsoleContextControlState | None,
    messages: Sequence[ConsoleChatMessage],
    has_pending_attachments: bool,
    input_per_mtok: float | None,
    draft_text: str,
) -> ConsoleCostState:
    """Compose Current and next-send display state from captured pure inputs."""
    empty_priced = (
        snapshot.available
        and snapshot.row_count == 0
        and snapshot.fleet_tokens == 0
        and pricing_available
    )
    if empty_priced:
        snapshot = replace(snapshot, total_usd=0.0, pricing_known=True)
    current = build_cost_state(
        snapshot,
        cache_state=cache_state,
        break_reason=break_reason,
        projected_delta_usd=projected_delta_usd,
        ttl_remaining_s=ttl_remaining_s,
        pricing_as_of=pricing_as_of,
    )
    if empty_priced:
        current = replace(current, label="$0.00", compact_label="$0.00")
    if context_state is None:
        return current
    next_send = build_console_next_send_projection(
        messages,
        has_pending_attachments,
        context_state.request_tokens,
        input_per_mtok,
        draft_text,
    )
    return build_console_context_cost_state(context_state, current, next_send)


@dataclass(slots=True, kw_only=True)
class ConsoleDraftSpendRefresh:
    """Own the single coalesced timer used for idle draft spend refreshes."""

    schedule_timer: Callable[[float, Callable[[], None]], Any]
    sync_settings_summary: Callable[[], None]
    sync_cost_chip: Callable[[], None]
    delay_seconds: float = 0.2
    timer: Any | None = None

    def schedule(self) -> None:
        self.stop()
        self.timer = self.schedule_timer(self.delay_seconds, self.refresh)

    def route_edit(self, *, run_active: bool) -> None:
        if run_active:
            self.stop()
        else:
            self.schedule()

    def stop(self) -> None:
        if self.timer is not None:
            self.timer.stop()
        self.timer = None

    def refresh(self) -> None:
        self.timer = None
        self.sync_settings_summary()
        self.sync_cost_chip()
