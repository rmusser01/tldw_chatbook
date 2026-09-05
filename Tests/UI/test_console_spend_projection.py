"""Focused contracts for Console current and next-send spend projections."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleDispatchRecoveryKind,
    ConsoleDispatchRecoveryState,
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_cost_tracker import (
    ConsoleCacheState,
    ConsoleCostSnapshot,
)
from tldw_chatbook.Chat.provider_continuation import ProviderContinuationCheckpoint
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
    ConsoleDraftSpendRefresh,
    build_console_context_messages,
    build_console_current_cost_messages,
    build_console_next_send_projection,
    build_console_spend_cost_state,
    build_console_spend_history_projection,
    fold_system_prompt,
)
from tldw_chatbook.Widgets.Console.console_context_controls import (
    build_console_context_control_state,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
)


def _context_state(*, request_tokens: int | None = 1_000):
    return build_console_context_control_state(
        settings=ConsoleSessionSettings(
            provider="anthropic", model="claude-sonnet-4-6", max_tokens=1_000
        ),
        estimate=ConsoleSettingsContextEstimate(
            used_tokens=request_tokens,
            token_limit=10_000,
            label="context",
        ),
    )


@pytest.mark.parametrize(
    "kind",
    (
        ConsoleDispatchRecoveryKind.ACCEPTED,
        ConsoleDispatchRecoveryKind.DISPATCH_STARTED,
        ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED,
        ConsoleDispatchRecoveryKind.EPHEMERAL_DISPATCH_STARTED,
    ),
)
def test_accepted_user_remains_in_request_context_but_not_current(kind):
    user = ConsoleChatMessage(ConsoleMessageRole.USER, "accepted", id="user")
    recovery = SimpleNamespace(
        kind=kind,
        checkpoint=SimpleNamespace(user_message_id=user.id),
    )

    projection = build_console_spend_history_projection(
        [user], recovery, None, ConsoleRunStatus.STREAMING, True
    )

    assert user.id in projection.request_ids
    assert user.id not in projection.current_ids


def test_production_shaped_remote_owner_is_not_current():
    history = ConsoleChatMessage(
        ConsoleMessageRole.USER,
        "history",
        id="history",
        persisted_message_id="p-history",
    )
    active = ConsoleChatMessage(
        ConsoleMessageRole.USER,
        "accepted",
        id="active",
        persisted_message_id="p-active",
    )
    assistant = ConsoleChatMessage(
        ConsoleMessageRole.ASSISTANT,
        "",
        id="assistant",
        persisted_message_id="p-assistant",
        parent_message_id="p-active",
        assistant_generation_state="dispatch_started",
    )
    recovery = ConsoleDispatchRecoveryState(
        kind=ConsoleDispatchRecoveryKind.REMOTE_DISPATCH_STARTED,
        assistant_message_id="p-assistant",
        conversation_id="conversation",
        visible_copy="Remote dispatch pending.",
        actions=(),
    )

    projection = build_console_spend_history_projection(
        [history, active, assistant], recovery, None, ConsoleRunStatus.IDLE, False
    )

    assert recovery.checkpoint is None
    assert history.id in projection.current_ids
    assert active.id in projection.request_ids
    assert active.id not in projection.current_ids


def test_failed_user_is_excluded_but_failed_assistant_usage_is_current():
    user = ConsoleChatMessage(ConsoleMessageRole.USER, "sent", id="user")
    failed_echo = ConsoleChatMessage(
        ConsoleMessageRole.USER, "never sent", id="failed-user", status="failed"
    )
    failed_assistant = ConsoleChatMessage(
        ConsoleMessageRole.ASSISTANT,
        "partial",
        id="failed-assistant",
        status="failed",
        usage=ProviderUsage(
            uncached_input=100,
            output=20,
            provider="anthropic",
            model="claude-sonnet-4-6",
        ),
    )

    projection = build_console_spend_history_projection(
        [user, failed_echo, failed_assistant],
        None,
        None,
        ConsoleRunStatus.IDLE,
        False,
    )

    assert failed_echo.id not in projection.request_ids
    assert failed_echo.id not in projection.current_ids
    assert failed_assistant.id not in projection.request_ids
    assert failed_assistant.id in projection.current_ids


def test_stopped_continuation_usage_is_current_but_not_request_context():
    user = ConsoleChatMessage(ConsoleMessageRole.USER, "sent", id="user")
    assistant = ConsoleChatMessage(
        ConsoleMessageRole.ASSISTANT,
        "partial",
        id="assistant",
        status="stopped",
        assistant_generation_state="stopped",
        provider_continuation=ProviderContinuationCheckpoint(
            schema_version=1,
            checkpoint_revision=1,
            provider="deepseek",
            protocol="chat_completions",
            model="deepseek-chat",
            api_base_url="https://api.deepseek.com",
            state="active",
            rounds=(),
        ),
        usage=ProviderUsage(
            uncached_input=100,
            output=20,
            provider="anthropic",
            model="claude-sonnet-4-6",
        ),
    )

    projection = build_console_spend_history_projection(
        [user, assistant], None, None, ConsoleRunStatus.IDLE, False
    )

    assert assistant.id not in projection.request_ids
    assert assistant.id in projection.current_ids


def test_actual_turn_usage_replaces_the_user_row_estimate_in_current():
    user = ConsoleChatMessage(ConsoleMessageRole.USER, "hello", id="user")
    assistant = ConsoleChatMessage(
        ConsoleMessageRole.ASSISTANT,
        "answer",
        id="assistant",
        usage=ProviderUsage(
            uncached_input=1_000,
            output=200,
            provider="anthropic",
            model="claude-sonnet-4-6",
        ),
    )

    rows = build_console_current_cost_messages(
        [user, assistant], {user.id, assistant.id}
    )

    assert rows == [assistant]


def test_context_projection_adds_live_draft_and_seeded_greeting():
    greeting = ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "Hello there", id="g")
    user = ConsoleChatMessage(ConsoleMessageRole.USER, "Prior turn", id="u")

    projected = build_console_context_messages([greeting, user], {user.id}, "New draft")

    assert fold_system_prompt("Be helpful", greeting.content).endswith("Hello there")
    assert projected == [
        {"role": "user", "content": "Prior turn"},
        {"role": "user", "content": "New draft"},
    ]


@pytest.mark.parametrize(
    "message",
    (
        ConsoleChatMessage(
            ConsoleMessageRole.ASSISTANT,
            "generated",
            attachments=(SimpleNamespace(id="image"),),
        ),
        ConsoleChatMessage(
            ConsoleMessageRole.USER,
            "failed image",
            status="failed",
            image_data=b"image",
        ),
    ),
)
def test_any_historical_media_makes_forecast_unavailable(message):
    state = build_console_next_send_projection([message], False, 1_000, 3.0, "sendable")
    assert state.label == "unavailable"
    assert "Media cost is not estimated" in state.tooltip


def test_zero_input_price_is_known_and_empty_draft_is_dash():
    priced = build_console_next_send_projection([], False, 1_000, 0.0, "send")
    empty = build_console_next_send_projection([], False, 1_000, 3.0, "  ")
    assert priced.label == "~+$0.00"
    assert empty.label == "—"


def test_invalid_draft_never_promises_a_next_send_charge():
    state = build_console_next_send_projection([], False, 1_000, 3.0, "x" * 200_001)

    assert state.label == "unavailable"
    assert "cannot be sent" in state.tooltip


@pytest.mark.parametrize(
    ("has_pending", "request_tokens", "input_per_mtok"),
    ((True, 1_000, 3.0), (False, None, 3.0), (False, 1_000, None)),
)
def test_pending_media_or_unknown_forecast_inputs_are_unavailable(
    has_pending, request_tokens, input_per_mtok
):
    state = build_console_next_send_projection(
        [], has_pending, request_tokens, input_per_mtok, "sendable"
    )
    assert state.label == "unavailable"


def test_unknown_current_pricing_does_not_hide_known_next_send_forecast():
    state = build_console_spend_cost_state(
        ConsoleCostSnapshot(None, 1_000, False, False, 1),
        ConsoleCacheState.NONE,
        None,
        None,
        None,
        None,
        True,
        _context_state(),
        [ConsoleChatMessage(ConsoleMessageRole.USER, "history")],
        False,
        3.0,
        "draft",
    )

    assert "Current 1.0k tok" in state.label
    assert "On next send ~+$0.003" in state.label


def test_nonempty_tracker_failure_is_unavailable_but_true_empty_is_zero():
    failed = ConsoleCostSnapshot(None, 0, False, False, 0, available=False)
    empty = ConsoleCostSnapshot(None, 0, False, False, 0)
    context = _context_state()
    message = ConsoleChatMessage(ConsoleMessageRole.USER, "history")
    failed_state = build_console_spend_cost_state(
        failed,
        ConsoleCacheState.WARM,
        "system prompt changed",
        0.01,
        60.0,
        "2026-09-04",
        True,
        context,
        [message],
        False,
        3.0,
        "draft",
    )
    empty_state = build_console_spend_cost_state(
        empty,
        ConsoleCacheState.NONE,
        None,
        None,
        None,
        "2026-09-04",
        True,
        context,
        [],
        False,
        3.0,
        "",
    )

    assert "Current unavailable" in failed_state.label
    assert "On next send ~+$" in failed_state.label
    assert failed_state.alert is True
    assert "system prompt changed" in failed_state.tooltip
    assert empty_state.label == "Context 11% · Current $0.00 · On next send —"


def test_idle_refresh_coalesces_and_uses_late_bound_callbacks():
    scheduled = []
    calls = []

    class Timer:
        stopped = False

        def stop(self):
            self.stopped = True

    owner = SimpleNamespace(summary=lambda: calls.append("old-summary"))
    controller = ConsoleDraftSpendRefresh(
        schedule_timer=lambda delay, callback: (
            scheduled.append((delay, callback, Timer())) or scheduled[-1][2]
        ),
        sync_settings_summary=lambda: owner.summary(),
        sync_cost_chip=lambda: calls.append("cost"),
    )
    controller.route_edit(run_active=False)
    first = scheduled[-1][2]
    controller.route_edit(run_active=False)
    assert first.stopped is True
    owner.summary = lambda: calls.append("new-summary")
    scheduled[-1][1]()
    assert calls == ["new-summary", "cost"]
    controller.route_edit(run_active=False)
    last = scheduled[-1][2]
    controller.route_edit(run_active=True)
    assert last.stopped is True
    assert controller.timer is None
