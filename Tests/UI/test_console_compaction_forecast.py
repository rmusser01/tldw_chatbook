"""TASK-33621.4 AC#3: once a chat is over its Ask trigger, say so before a send.

Live on dev 2026-10-29 (Anthropic claude-haiku-4-5, custom 1,500-token
budget, 160x45): after three exchanges the chat's history was 1,656 tokens,
over the 1,200-token trigger, yet the header read "Ready · not tested",
Inspect's "What happens if I send now?" read "Run: Ready" and the Context
chip read "Context 1%". The next send paused with no warning.

The forecast asks the same question the send preflight does
(``decide_compaction`` over the conversation's own tokens and complete
turns, the draft excluded), so the surfaces cannot disagree with the hold
about whether a send will pause.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_context_compaction import (
    EffectiveMemoryKind,
    EffectiveMemoryResult,
)
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextBudgetMode,
    ContextCompactionMode,
)
from tldw_chatbook.Chat.console_context_repository import ConsoleMemoryRecord
from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostState
from tldw_chatbook.Chat.console_prepared_request import tagged_memory_message
from tldw_chatbook.Chat.console_display_state import (
    ConsoleControlState,
    ConsoleInspectorState,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
)
from tldw_chatbook.Widgets.Console.console_context_controls import (
    COMPACTION_PAUSE_FORECAST_COPY,
    build_console_context_control_state,
    build_console_context_cost_state,
    compaction_forecast_rows,
)
from tldw_chatbook.Widgets.Console.console_send_authority_summary import (
    project_console_send_authority,
)
from tldw_chatbook.Widgets.Console.console_workbench_state import (
    build_console_workbench_state,
)

pytestmark = pytest.mark.bootstrap_profile

_COST = ConsoleCostState(
    label="$0.00", compact_label="$0.00", tooltip="", alert=False, cold=False
)


def _message(role, native_id, *, status="complete", content=None):
    return ConsoleChatMessage(
        role=role,
        content=content if content is not None else f"text of {native_id}",
        id=native_id,
        persisted_message_id=f"p-{native_id}",
        status=status,
    )


def _exchanges():
    user, assistant = ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT
    return [
        _message(user, "u1"),
        _message(assistant, "a1"),
        _message(ConsoleMessageRole.SYSTEM, "notice"),
        _message(user, "u2"),
        _message(assistant, "a2"),
    ]


def _memory(boundary: str) -> ConsoleMemoryRecord:
    return ConsoleMemoryRecord(
        memory_id="memory-1",
        conversation_id="conversation-1",
        boundary_message_id=boundary,
        captured_leaf_message_id=boundary,
        lineage_json="[]",
        summary_text="Earlier: the user asked two questions.",
        provider="anthropic",
        model="claude-haiku-4-5",
        prompt_id="console.rewind_summarize",
        prompt_revision=1,
        prompt_digest="digest",
        selected_units_json="[]",
        summarized_prefix_digest="prefix",
        input_tokens=1,
        output_tokens=1,
        before_tokens=1,
        after_tokens=1,
        created_at="2026-10-29T00:00:00+00:00",
    )


def test_forecast_rows_are_complete_turns_without_notices() -> None:
    rows, units = compaction_forecast_rows(
        _exchanges(), EffectiveMemoryResult(EffectiveMemoryKind.RAW)
    )

    assert [row["content"] for row in rows] == [
        "text of u1",
        "text of a1",
        "text of u2",
        "text of a2",
    ]
    assert units == 2


def test_an_unfinished_reply_is_not_history_or_a_turn() -> None:
    messages = _exchanges()
    messages[-1] = _message(ConsoleMessageRole.ASSISTANT, "a2", status="failed")

    rows, units = compaction_forecast_rows(
        messages, EffectiveMemoryResult(EffectiveMemoryKind.RAW)
    )

    assert "text of a2" not in [row["content"] for row in rows]
    assert units == 1


def test_a_summary_replaces_the_turns_up_to_its_boundary() -> None:
    effective = EffectiveMemoryResult(
        EffectiveMemoryKind.GENERATED_PREFIX, memory=_memory("p-a1")
    )

    rows, units = compaction_forecast_rows(_exchanges(), effective)

    # Wrapped exactly as the send carries it (live 2026-10-29: counting the
    # bare summary forecast "Ready" for a send the preflight held at 1,220
    # of a 1,200-token trigger).
    assert rows[0] == dict(tagged_memory_message(_memory("p-a1").summary_text))
    assert [row["content"] for row in rows[1:]] == ["text of u2", "text of a2"]
    assert units == 1


@pytest.mark.parametrize(
    "effective",
    [
        EffectiveMemoryResult(EffectiveMemoryKind.GENERATED_PREFIX, memory=_memory("gone")),
        EffectiveMemoryResult(EffectiveMemoryKind.GENERATED_RANGE, memory=_memory("p-a1")),
        EffectiveMemoryResult(EffectiveMemoryKind.LEGACY_PREFIX),
    ],
    ids=["boundary-off-path", "range", "legacy"],
)
def test_a_memory_shape_the_forecast_cannot_model_forecasts_nothing(effective) -> None:
    assert compaction_forecast_rows(_exchanges(), effective) is None


def _state(*, tokens, units, mode=ContextCompactionMode.ASK):
    return build_console_context_control_state(
        settings=ConsoleSessionSettings(
            provider="anthropic", model="claude-haiku-4-5", max_tokens=4_096
        ),
        estimate=ConsoleSettingsContextEstimate(
            used_tokens=1_658, token_limit=200_000, label="1,658 / 200,000 tokens"
        ),
        overrides=ConsoleContextPolicyOverrides(
            budget_mode=ContextBudgetMode.CUSTOM,
            custom_budget_tokens=1_500,
            compaction_mode=mode,
        ),
        forecast_tokens=tokens,
        forecast_units=units,
    )


def test_over_the_ask_trigger_the_next_send_is_forecast_to_pause() -> None:
    forecast = _state(tokens=1_656, units=3).compaction_pause_forecast

    assert forecast is not None
    assert (forecast.used_tokens, forecast.trigger_tokens, forecast.budget_tokens) == (
        1_656,
        1_200,
        1_500,
    )


@pytest.mark.parametrize(
    ("tokens", "units", "mode"),
    [
        (1_199, 3, ContextCompactionMode.ASK),  # below the trigger
        (1_656, 0, ContextCompactionMode.ASK),  # nothing to compact
        (1_656, 3, ContextCompactionMode.AUTOMATIC),  # compacts without asking
        (1_656, 3, ContextCompactionMode.OFF),
        (None, 0, ContextCompactionMode.ASK),  # not measured
    ],
)
def test_no_pause_is_forecast_when_the_preflight_would_not_ask(
    tokens, units, mode
) -> None:
    assert _state(tokens=tokens, units=units, mode=mode).compaction_pause_forecast is None


def test_the_context_chip_leads_with_the_forecast_and_explains_it() -> None:
    over = build_console_context_cost_state(_state(tokens=1_656, units=3), _COST)
    under = build_console_context_cost_state(_state(tokens=900, units=3), _COST)

    assert over.label.startswith(COMPACTION_PAUSE_FORECAST_COPY)
    assert over.compact_label.startswith("Pauses to compact")
    assert "Next send pauses for compaction approval" in over.tooltip
    assert "1,656 of 1,500" in over.tooltip
    assert COMPACTION_PAUSE_FORECAST_COPY not in under.label
    assert "compaction approval" not in under.tooltip


def test_inspect_says_the_next_send_pauses_but_not_over_live_or_stuck_work() -> None:
    def run(**values) -> str:
        return project_console_send_authority(
            ConsoleInspectorState.from_values(**values)
        ).run

    assert run(compaction_pause_expected=True) == "Pauses for compaction"
    assert run() == "Ready"
    assert run(compaction_pause_expected=True, run_active=True) == "Running"
    assert run(
        compaction_pause_expected=True, run_blocked_reason="compaction ask"
    ) == "Blocked — compaction ask"
    # It describes the NEXT send, so it outranks the last run's failure.
    assert run(
        compaction_pause_expected=True, run_failed=True, run_failure_reason="x"
    ) == "Pauses for compaction"


def _control_state() -> ConsoleControlState:
    return ConsoleControlState(
        provider_label="Provider: Anthropic",
        model_label="Model: claude-haiku-4-5",
        assistant_label="Assistant: General",
        rag_label="RAG: off",
        sources_label="Sources: 0",
        tools_label="Tools: 0",
        approvals_label="Approvals: 0",
    )


def test_the_header_badge_says_the_next_send_pauses() -> None:
    def badge(readiness_word="Ready · not tested", **values) -> str:
        header = build_console_workbench_state(
            control_state=_control_state(), readiness_word=readiness_word, **values
        ).header
        return header.status_label

    assert badge(compaction_pause_expected=True) == COMPACTION_PAUSE_FORECAST_COPY
    assert badge() == "Ready · not tested"
    assert badge(compaction_pause_expected=True, run_active=True) == ""
    assert badge(compaction_pause_expected=True, blocked_turn="compaction ask") == (
        "Blocked"
    )
    # Single-pane widths drop the readiness word because it overruns the row.
    assert badge(compaction_pause_expected=True, readiness_word="") == ""


def test_both_compaction_run_lines_fit_the_inspect_rail_whole() -> None:
    """Live 2026-10-29 (160x45): Inspect showed "Run: Next send pauses for
    co…" -- the pinned block gets 29 content columns (its own docstring), so
    the line it exists to answer was cut. Both compaction lines fit whole."""
    from rich.cells import cell_len

    def run(**values) -> str:
        return project_console_send_authority(
            ConsoleInspectorState.from_values(**values)
        ).run

    from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
        _BLOCKED_TURN_REASONS,
    )
    from tldw_chatbook.Chat.console_turn_preparation import (
        ConsolePreparationPauseKind,
    )

    held = _BLOCKED_TURN_REASONS[ConsolePreparationPauseKind.CONTEXT_COMPACTION]
    for line in (
        run(compaction_pause_expected=True),
        run(run_blocked_reason=held),
    ):
        assert cell_len(f"Run: {line}") <= 29, line
