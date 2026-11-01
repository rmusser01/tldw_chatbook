"""TASK-33621.4 AC#3: once a chat is over its Ask trigger, say so before a send.

Live on dev 2026-10-29 (Anthropic claude-haiku-4-5, custom 1,500-token
budget, 160x45): after three exchanges the chat's history was 1,656 tokens,
over the 1,200-token trigger, yet the header read "Ready · not tested",
Inspect's "What happens if I send now?" read "Run: Ready" and the Context
chip read "Context 1%". The next send paused with no warning.

The qualified static forecast uses ``decide_compaction`` with estimated
history, current mandatory input and capacity. Agent/tool-dependent next-send
behavior stays unavailable, with a qualified high-context advisory. These
controls preserve the explicit estimate qualification rather than promising
exact future-send parity for dynamic inputs.
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


@pytest.mark.asyncio
async def test_automatic_forecast_matches_real_preflight_with_overhead_and_cap(
    request, monkeypatch, tmp_path
):
    """Current mandatory input and provider caps reduce Automatic capacity."""
    from dataclasses import replace
    from Tests.UI.test_console_compaction_hold_flow import _ask_console, _held_console
    from Tests.UI.test_console_native_chat_flow import _sync_console_control_bar_now
    from tldw_chatbook.Chat import console_prepared_request, console_session_settings
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
    from tldw_chatbook.Utils.token_counter import ContextWindowResolution
    from tldw_chatbook import config as config_module, model_capabilities

    def words(messages, _model, *_args, **_kwargs):
        # Match the shared gpt framing base used by incremental display counts.
        return 3 + sum(len(str(row.get("content", "")).split()) for row in messages)

    monkeypatch.setattr(
        console_prepared_request, "count_console_messages_tokens", words
    )
    monkeypatch.setattr(console_session_settings, "_estimate_tokens_locally", words)
    window = ContextWindowResolution(tokens=10_768, verified=True, source="test")
    caps = {"max_output_tokens": 256}
    catalog = model_capabilities.get_model_capabilities()
    monkeypatch.setattr(catalog, "get_model_capabilities", lambda *_args: dict(caps))
    app, database, host, adapter, adapter_calls = await _held_console(
        monkeypatch, tmp_path
    )
    assert config_module.save_settings_to_cli_config(
        {"console": {"agent_runtime": False}}
    )
    app.app_config = config_module.load_settings(force_reload=True)
    console = None
    try:
        async with host.run_test(size=(160, 45)) as pilot:
            console, controller, session, runtime, tasks = await _ask_console(
                host, pilot, monkeypatch, adapter
            )
            assert not controller._agent_runtime_enabled
            gateway = controller.provider_gateway
            resolve = gateway.resolve_for_send

            async def current_resolution(selection):
                return replace(
                    await resolve(selection), context_window=window, max_tokens=256
                )

            monkeypatch.setattr(gateway, "resolve_for_send", current_resolution)
            monkeypatch.setattr(
                console._context_spend, "_context_window_for_display", lambda _s: window
            )
            for _ in range(2):
                controller.store.append_message(
                    session.id, role=ConsoleMessageRole.USER, content="q", persist=True
                )
                controller.store.append_message(
                    session.id,
                    role=ConsoleMessageRole.ASSISTANT,
                    content="detail " * 2499,
                    persist=True,
                )
            controller.store.set_session_context_policy_overrides(
                session.id,
                ConsoleContextPolicyOverrides(
                    budget_mode=ContextBudgetMode.AUTOMATIC,
                    compaction_mode=ContextCompactionMode.ASK,
                ),
            )
            for mandatory, input_cap in ((5000, None), (3000, 8000)):
                caps.clear()
                caps.update(max_output_tokens=256)
                if input_cap is not None:
                    caps["max_input_tokens"] = input_cap
                    caps["max_output_tokens"] = 128
                session.settings = replace(
                    controller.store.session_settings(session.id),
                    source="user",
                    max_tokens=256,
                    system_prompt="mandatory " * (mandatory - 4),
                )
                session.settings_revision += 1
                await console._sync_native_console_chat_ui()
                console._console_composer_or_none().load_draft("next")
                resolution = await gateway.resolve_for_send(
                    controller._provider_selection_for_session(session.id)
                )
                rows = controller._provider_messages_for_session(
                    session.id,
                    annotate_ids=True,
                    turn_context=controller.resolve_turn_execution_context(session.id),
                )
                assert controller._context_repository is not None
                assert session.persisted_conversation_id is not None
                snapshots = controller._durable_context_snapshots(session.id)
                assert snapshots is not None and len(snapshots) == 4
                prepared = gateway.prepare_chat_request(
                    resolution,
                    [*rows, {"role": "user", "content": "next"}],
                    apply_safety_window=False,
                )
                assert prepared.accounting.compactable_tokens == 5000
                assert prepared.accounting.non_compactable_tokens == mandatory
                assert prepared.capacity.effective_input_ceiling_tokens == (
                    input_cap or 10000
                )
                hold, alert = await controller._assess_context_compaction(
                    session_id=session.id,
                    resolution=resolution,
                    provider_messages=[*rows, {"role": "user", "content": "next"}],
                    uncommitted_user_message_id=None,
                )
                assert alert is None and hold is not None
                assert (hold.used_tokens, hold.budget_tokens, hold.trigger_tokens) == (
                    5000,
                    5000,
                    4000,
                )
                projection = spend.ConsoleReadinessConfigProjection.for_screen(console)
                await _sync_console_control_bar_now(console)
                memory = spend.ConsoleContextReadSnapshot.for_screen(
                    console, max_age=1.0
                )
                assert await memory.warm(controller, session.id)
                states = []
                assert projection.run(
                    lambda: states.append(
                        console._context_spend._console_context_control_state_for_session(
                            session.id
                        )
                    )
                )
                state = states[-1]
                forecast = state.compaction_pause_forecast
                assert (
                    forecast is not None
                ), "Automatic forecast omits current mandatory input/cap"
                # The existing history display counter includes its three-token
                # framing base; preflight attributes that to mandatory input.
                assert forecast.used_tokens == hold.used_tokens + 3
                assert (forecast.budget_tokens, forecast.trigger_tokens) == (
                    hold.budget_tokens,
                    hold.trigger_tokens,
                )
                assert state.request_overhead_tokens == mandatory
                assert (
                    state.response_max_tokens
                    == prepared.capacity.effective_response_tokens
                )
                assert (
                    "estimated"
                    in build_console_context_cost_state(state, _COST).tooltip.lower()
                )
            assert config_module.save_settings_to_cli_config(
                {"console": {"agent_runtime": True}}
            )
            app.app_config = config_module.load_settings(force_reload=True)
            await _sync_console_control_bar_now(console)
            assert controller._agent_runtime_enabled
            assert await memory.warm(controller, session.id)

            def no_preview(*_args, **_kwargs):
                raise AssertionError(
                    "forecast must not invoke permission-bearing preview"
                )

            monkeypatch.setattr(
                controller._agent_bridge, "preview_tool_surface", no_preview
            )
            states = []
            assert projection.run(
                lambda: states.append(
                    console._context_spend._console_context_control_state_for_session(
                        session.id
                    )
                )
            )
            assert states[-1].compaction_pause_forecast is None
            assert states[-1].forecast_unavailable_reason
            assert not states[-1].forecast_high_context
            assert (
                "forecast unavailable"
                in build_console_context_cost_state(states[-1], _COST).label.lower()
            )
            controller.store.set_session_context_policy_overrides(
                session.id,
                ConsoleContextPolicyOverrides(
                    budget_mode=ContextBudgetMode.CUSTOM,
                    custom_budget_tokens=1500,
                    compaction_mode=ContextCompactionMode.ASK,
                ),
            )
            assert await memory.warm(controller, session.id)
            states = []
            assert projection.run(
                lambda: states.append(
                    console._context_spend._console_context_control_state_for_session(
                        session.id
                    )
                )
            )
            advisory = states[-1]
            assert advisory.forecast_high_context
            assert advisory.compaction_pause_forecast is None
            shown = build_console_context_cost_state(advisory, _COST)
            assert "High context" in shown.label
            assert "next-send check unavailable" in shown.tooltip
            assert "Next send pauses" not in shown.tooltip
            assert adapter_calls == []
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


@pytest.mark.asyncio
async def test_bound_skill_ask_gate_uses_its_overhead_before_regenerate_commits(
    request, monkeypatch, tmp_path
):
    """Admitted skill bindings must reach the real preflight, before a sibling."""
    from dataclasses import replace
    from Tests.UI.test_console_compaction_hold_flow import _ask_console, _held_console
    from tldw_chatbook.Chat import console_prepared_request
    from tldw_chatbook.Chat.console_tool_surface_preview import ToolSurfacePreview
    from tldw_chatbook.Utils.token_counter import ContextWindowResolution

    def words(messages, _model, **_kwargs):
        return sum(len(str(row.get("content", "")).split()) for row in messages)

    monkeypatch.setattr(
        console_prepared_request, "count_console_messages_tokens", words
    )
    app, database, host, adapter, adapter_calls = await _held_console(
        monkeypatch, tmp_path
    )
    console = None
    try:
        async with host.run_test(size=(160, 45)) as pilot:
            console, controller, session, runtime, tasks = await _ask_console(
                host, pilot, monkeypatch, adapter
            )
            assert controller._agent_bridge is not None
            gateway = controller.provider_gateway
            resolve = gateway.resolve_for_send

            async def current_resolution(selection):
                return replace(
                    await resolve(selection),
                    max_tokens=256,
                    context_window=ContextWindowResolution(10_768, "test", True),
                )

            monkeypatch.setattr(gateway, "resolve_for_send", current_resolution)
            session.settings = replace(
                controller.store.session_settings(session.id),
                source="user",
                max_tokens=256,
            )
            session.settings_revision += 1
            controller.store.set_session_context_policy_overrides(
                session.id,
                ConsoleContextPolicyOverrides(
                    budget_mode=ContextBudgetMode.AUTOMATIC,
                    compaction_mode=ContextCompactionMode.ASK,
                ),
            )
            for _ in range(2):
                controller.store.append_message(
                    session.id, role=ConsoleMessageRole.USER, content="q", persist=True
                )
                controller.store.append_message(
                    session.id,
                    role=ConsoleMessageRole.ASSISTANT,
                    content="detail " * 2499,
                    persist=True,
                )
            controller.store.append_message(
                session.id, role=ConsoleMessageRole.USER, content="target", persist=True
            )
            reply = controller.store.append_message(
                session.id,
                role=ConsoleMessageRole.ASSISTANT,
                content="old",
                persist=True,
            )
            await console._sync_native_console_chat_ui()
            observed = []

            async def admitted_skill(messages, _context=None):
                return messages, None, [], ("bound-skill",), ""

            def selected_surface(
                _resolution=None, *, turn_skill_bindings=(), **_kwargs
            ):
                observed.append(turn_skill_bindings)
                return ToolSurfacePreview(
                    system_prompt="skill " * 4999 if turn_skill_bindings else ""
                )

            monkeypatch.setattr(controller, "_apply_skill_substitution", admitted_skill)
            monkeypatch.setattr(
                controller._agent_bridge, "preview_tool_surface", selected_surface
            )

            def saved_rows():
                with database.transaction() as cursor:
                    cursor.execute(
                        "SELECT id, parent_message_id, version, assistant_generation_state "
                        "FROM messages WHERE conversation_id = ? ORDER BY id",
                        (session.persisted_conversation_id,),
                    )
                    return tuple(tuple(row) for row in cursor.fetchall())

            saved_before = saved_rows()
            before = tuple(controller.store.active_path_message_ids(session.id))
            result = await controller.regenerate_message(reply.id)
            assert (
                not result.accepted
            ), "bound skill Ask gate must refuse before a sibling"
            assert "Not regenerated" in result.visible_copy
            assert observed == [("bound-skill",)]
            assert saved_rows() == saved_before
            assert (
                tuple(controller.store.active_path_message_ids(session.id))[
                    : len(before)
                ]
                == before
            )
            assert [
                m.id
                for m in controller.store.messages_for_session(session.id)
                if m.role is ConsoleMessageRole.ASSISTANT
            ] == [before[1], before[3], reply.id]
            assert adapter_calls == []
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


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
        EffectiveMemoryResult(
            EffectiveMemoryKind.GENERATED_PREFIX, memory=_memory("gone")
        ),
        EffectiveMemoryResult(
            EffectiveMemoryKind.GENERATED_RANGE, memory=_memory("p-a1")
        ),
        EffectiveMemoryResult(EffectiveMemoryKind.LEGACY_PREFIX),
    ],
    ids=["boundary-off-path", "range", "legacy"],
)
def test_a_memory_shape_the_forecast_cannot_model_forecasts_nothing(effective) -> None:
    assert compaction_forecast_rows(_exchanges(), effective) is None


def test_cold_forecast_does_not_read_history_or_count(monkeypatch) -> None:
    from tldw_chatbook.UI.Console_Modules.context_spend import (
        ConsoleContextSpendController,
    )
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

    controller = ConsoleContextSpendController.__new__(ConsoleContextSpendController)
    controller._screen = object()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cold forecast performed history/count work")

    controller._ensure_console_chat_store = forbidden
    controller._build_context_estimate = forbidden
    monkeypatch.setattr(spend, "console_tokenizer_display_ready", lambda *_args: False)
    assert controller._console_compaction_forecast_inputs(
        "session",
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
        ConsoleContextPolicyOverrides(),
        None,
        EffectiveMemoryResult(EffectiveMemoryKind.RAW),
    ) == (None, 0)


def test_unavailable_forecast_is_honest_without_hiding_live_or_failed_work() -> None:
    from dataclasses import replace

    state = replace(
        _state(tokens=1656, units=3),
        forecast_tokens=None,
        forecast_unavailable_reason="Current tool overhead is unknown.",
    )
    assert (
        "forecast unavailable"
        in build_console_context_cost_state(state, _COST).label.lower()
    )
    for values, expected in (
        ({"compaction_forecast_unavailable": True}, "Forecast unavailable"),
        ({"compaction_forecast_unavailable": True, "run_active": True}, "Running"),
        ({"compaction_forecast_unavailable": True, "run_failed": True}, "Failed"),
        (
            {
                "compaction_forecast_unavailable": True,
                "run_blocked_reason": "recovery needed",
            },
            "Blocked — recovery needed",
        ),
    ):
        assert (
            project_console_send_authority(
                ConsoleInspectorState.from_values(**values)
            ).run
            == expected
        )
    header = build_console_workbench_state(
        control_state=_control_state(),
        readiness_word="Ready",
        compaction_forecast_unavailable=True,
    ).header
    assert header.status_label == "Compaction forecast unavailable"
    high = replace(
        state,
        forecast_high_context=True,
        forecast_unavailable_reason=(
            "Estimated history reaches the displayed compaction trigger; next-send check unavailable."
        ),
    )
    assert high.compaction_pause_forecast is None
    assert "High context" in build_console_context_cost_state(high, _COST).compact_label
    assert (
        "next-send check unavailable"
        in build_console_context_cost_state(high, _COST).tooltip
    )

    blocked = build_console_workbench_state(
        control_state=_control_state(),
        readiness_word="Missing key",
        provider_blocker_copy="Missing API key",
        compaction_forecast_unavailable=True,
    ).header
    assert blocked.status_label == "Missing key"
    from tldw_chatbook.Widgets.Console import console_workbench_state as workbench

    stopped = build_console_workbench_state(
        control_state=replace(_control_state(), emergency_stop_held=True),
        readiness_word="Ready",
        compaction_forecast_unavailable=True,
    ).header
    assert stopped.status_label == workbench.CONSOLE_EMERGENCY_STOP_HEADER_LABEL

    from types import SimpleNamespace
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    store = SimpleNamespace(active_session_id="session")
    controller = SimpleNamespace(
        store=store,
        run_state=SimpleNamespace(
            status=ConsoleRunStatus.FAILED, visible_copy="Response failed."
        ),
        response_rules=None,
        has_pending_approval_round=lambda _sid: False,
    )
    acknowledgement = ["Sending…"]
    screen = SimpleNamespace(
        _console_chat_store=store,
        _console_chat_controller=controller,
        _image=SimpleNamespace(
            _h3_image_edit_registry=lambda: SimpleNamespace(active=lambda _sid: None)
        ),
        _context_spend=SimpleNamespace(
            compaction_pause_expected=lambda: False,
            compaction_forecast_unavailable=lambda: True,
        ),
        _console_send_ack=SimpleNamespace(run_copy=lambda _sid: acknowledgement[0]),
    )
    screen._console_active_run_copy = lambda: ChatScreen._console_active_run_copy(
        screen
    )
    assert screen._console_active_run_copy() == "Sending…"
    acknowledgement[0] = ""
    assert ChatScreen._native_run_status_copy(screen) == "Response failed."
    controller.run_state.status = ConsoleRunStatus.IDLE
    assert screen._console_active_run_copy() == "Forecast unavailable"


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
        request_overhead_tokens=0,
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
    assert (
        _state(tokens=tokens, units=units, mode=mode).compaction_pause_forecast is None
    )


def test_the_context_chip_leads_with_the_forecast_and_explains_it() -> None:
    over = build_console_context_cost_state(_state(tokens=1_656, units=3), _COST)
    under = build_console_context_cost_state(_state(tokens=900, units=3), _COST)

    assert over.label.startswith(COMPACTION_PAUSE_FORECAST_COPY)
    assert over.compact_label.startswith("Pauses to compact")
    assert "Estimated next send pauses for compaction approval:" in over.tooltip
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
    assert (
        run(compaction_pause_expected=True, run_blocked_reason="compaction ask")
        == "Blocked — compaction ask"
    )
    # It describes the NEXT send, so it outranks the last run's failure.
    assert (
        run(compaction_pause_expected=True, run_failed=True, run_failure_reason="x")
        == "Pauses for compaction"
    )


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
