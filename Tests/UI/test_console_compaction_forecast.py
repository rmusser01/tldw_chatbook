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

import json
from types import SimpleNamespace

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
            # These preflight controls call the original synchronous capture.
            trust = await app.ensure_local_skill_trust_service()
            assert app.local_skills_service.trust_service is trust
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
            assert controller._agent_bridge is not None

            def no_preview(*_args, **_kwargs):
                raise AssertionError(
                    "forecast must not invoke permission-bearing preview"
                )

            def set_policy(budget: int | None) -> None:
                controller.store.set_session_context_policy_overrides(
                    session.id,
                    ConsoleContextPolicyOverrides(
                        budget_mode=(
                            ContextBudgetMode.AUTOMATIC
                            if budget is None
                            else ContextBudgetMode.CUSTOM
                        ),
                        custom_budget_tokens=budget,
                        compaction_mode=ContextCompactionMode.ASK,
                    ),
                )

            async def agent_state_and_real_hold():
                """The displayed state, then the real preflight's verdict."""
                projection = spend.ConsoleReadinessConfigProjection.for_screen(console)
                await _sync_console_control_bar_now(console)
                assert await memory.warm(controller, session.id)
                states = []
                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(
                        controller._agent_bridge, "preview_tool_surface", no_preview
                    )
                    assert projection.run(
                        lambda: states.append(
                            console._context_spend._console_context_control_state_for_session(
                                session.id
                            )
                        )
                    )
                resolution = await gateway.resolve_for_send(
                    controller._provider_selection_for_session(session.id)
                )
                rows = controller._provider_messages_for_session(
                    session.id,
                    annotate_ids=True,
                    turn_context=controller.resolve_turn_execution_context(session.id),
                )
                hold, alert = await controller._assess_context_compaction(
                    session_id=session.id,
                    resolution=resolution,
                    provider_messages=[*rows, {"role": "user", "content": "next"}],
                    uncommitted_user_message_id=None,
                )
                assert alert is None
                return states[-1], hold

            # Owner decision 2026-11-20, "ready, then may pause". Review B1
            # (2026-11-22): an agent chat's static input (here a 3,000-token
            # system prompt) used to be dropped with its tools, so the
            # displayed Automatic trigger was 6,400 of 8,000 and the header
            # read Ready while the real preflight held at 4,000 of 5,000.
            set_policy(None)
            over, hold = await agent_state_and_real_hold()
            assert hold is not None and (
                hold.used_tokens,
                hold.budget_tokens,
                hold.trigger_tokens,
            ) == (5000, 5000, 4000), hold
            assert over.compaction_trigger_tokens == hold.trigger_tokens, (
                over.compaction_trigger_tokens,
                hold,
            )
            assert over.forecast_high_context, "Ready shown where the send holds"
            assert over.compaction_pause_forecast is None  # may, not will
            shown = build_console_context_cost_state(over, _COST)
            assert shown.label.startswith("May pause to compact · "), shown.label
            assert shown.compact_label.startswith("May pause to compact · ")
            assert (
                "May pause to compact: the estimated history (~5,003 conversation "
                "tokens) is over the trigger." in shown.tooltip
            ), shown.tooltip
            assert "The agent's tools are only estimated here" in shown.tooltip
            assert "you approve before anything is summarized" in shown.tooltip
            assert "forecast unavailable" not in shown.label.lower()
            assert "Next send pauses" not in shown.tooltip

            # Without the 8,000-token input cap the same chat is under its
            # trigger (5,600 of 7,000): Ready, and the real preflight agrees.
            caps.clear()
            caps.update(max_output_tokens=256)
            session.settings_revision += 1
            await console._sync_native_console_chat_ui()
            below, hold = await agent_state_and_real_hold()
            assert hold is None, hold
            assert below.compaction_trigger_tokens == 5600
            assert below.compaction_pause_forecast is None
            assert not below.forecast_high_context
            shown = build_console_context_cost_state(below, _COST)
            visible = f"{shown.label}\n{shown.compact_label}\n{shown.tooltip}"
            # ("On next send unavailable" is the spend estimate, not this.)
            assert "forecast unavailable" not in visible.lower(), visible
            assert "May pause to compact" not in visible, visible

            set_policy(1500)
            advisory, hold = await agent_state_and_real_hold()
            assert hold is not None and hold.trigger_tokens == 1200
            assert advisory.forecast_high_context
            assert advisory.compaction_pause_forecast is None
            shown = build_console_context_cost_state(advisory, _COST)
            assert shown.label.startswith("May pause to compact · "), shown.label
            assert adapter_calls == []
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "D2-unmeasured",
        "D1-native",
        "D1-fenced",
        "D2-other-session",
        "E-custom-fenced",
        "C3-stale-gate",
        "cannot-fit",
        "may-not-fit",
        "compact-keeps-max",
        "B-persisted",
        "raw-armed-signature",
        "upgrade-figure",
        "internal-prompt-figure",
    ],
)
async def test_an_agent_chats_tool_cost_counts_and_is_never_a_false_ready(
    request, monkeypatch, tmp_path, case, console_tool_cost_file
):
    """Review D1-D3 and follow-ups 1-2 (2026-11-22), against the real check.

    An agent chat (3,000-token system prompt, 5,000 tokens of history,
    Automatic budget, a 10,000-token input ceiling) whose tools are either
    provider-native (2,004 schema tokens) or fenced into the system prompt
    (a 2,000-token rendering delta, as on llama.cpp/vLLM/Ollama):

    - D2: before any request in this run the tools are unmeasured, so the
      chat may pause (the delta review's probe B read Ready at a displayed
      5,600 while the real check held at 3,996).
    - D1/D3: after one real check the displayed trigger is the real hold's
      (probe A read Ready at 5,600 against a fenced hold at 4,000).
    - D2(a): another session falls back to the largest remembered figure.
    - Follow-up 1: a tool-gate change forgets a stale figure (probe C3: a
      remembered 204 read Ready while 2,004 held).
    - Follow-up 2: a system prompt that leaves no room reads "Next send
      won't fit", never Ready (probe D: the real send is refused).
    """
    from dataclasses import replace
    from Tests.UI.test_console_compaction_hold_flow import _ask_console, _held_console
    from Tests.UI.test_console_native_chat_flow import _sync_console_control_bar_now
    from tldw_chatbook.Chat import console_prepared_request, console_session_settings
    from tldw_chatbook.Chat.console_tool_surface_preview import ToolSurfacePreview
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
    from tldw_chatbook.Utils.token_counter import ContextWindowResolution
    from tldw_chatbook import config as config_module, model_capabilities

    def words(messages, _model, *_args, **_kwargs):
        return 3 + sum(len(str(row.get("content", "")).split()) for row in messages)

    monkeypatch.setattr(
        console_prepared_request, "count_console_messages_tokens", words
    )
    monkeypatch.setattr(console_session_settings, "_estimate_tokens_locally", words)
    window = ContextWindowResolution(tokens=10_768, verified=True, source="test")
    catalog = model_capabilities.get_model_capabilities()
    monkeypatch.setattr(
        catalog, "get_model_capabilities", lambda *_args: {"max_output_tokens": 256}
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
            trust = await app.ensure_local_skill_trust_service()
            assert app.local_skills_service.trust_service is trust
            assert controller._agent_runtime_enabled
            assert controller._agent_bridge is not None
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

            def set_policy(budget: int | None) -> None:
                controller.store.set_session_context_policy_overrides(
                    session.id,
                    ConsoleContextPolicyOverrides(
                        budget_mode=(
                            ContextBudgetMode.AUTOMATIC
                            if budget is None
                            else ContextBudgetMode.CUSTOM
                        ),
                        custom_budget_tokens=budget,
                        compaction_mode=ContextCompactionMode.ASK,
                    ),
                )

            async def set_prompt(prompt_words: int) -> None:
                session.settings = replace(
                    controller.store.session_settings(session.id),
                    source="user",
                    max_tokens=256,
                    system_prompt="mandatory " * prompt_words,
                )
                session.settings_revision += 1
                await console._sync_native_console_chat_ui()
                console._console_composer_or_none().load_draft("next")

            set_policy(None)
            await set_prompt(2996)

            def no_preview(*_args, **_kwargs):
                raise AssertionError("the display must not preview the tool surface")

            async def displayed():
                projection = spend.ConsoleReadinessConfigProjection.for_screen(console)
                await _sync_console_control_bar_now(console)
                memory = spend.ConsoleContextReadSnapshot.for_screen(
                    console, max_age=1.0
                )
                assert await memory.warm(controller, session.id)
                states = []
                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(
                        controller._agent_bridge, "preview_tool_surface", no_preview
                    )
                    assert projection.run(
                        lambda: states.append(
                            console._context_spend._console_context_control_state_for_session(
                                session.id
                            )
                        )
                    )
                return states[-1]

            async def real_check(surface):
                resolution = await gateway.resolve_for_send(
                    controller._provider_selection_for_session(session.id)
                )
                rows = controller._provider_messages_for_session(
                    session.id,
                    annotate_ids=True,
                    turn_context=controller.resolve_turn_execution_context(session.id),
                )
                with pytest.MonkeyPatch.context() as patch:
                    if surface is not None:
                        patch.setattr(
                            controller._agent_bridge,
                            "preview_tool_surface",
                            lambda *_args, **_kwargs: surface,
                        )
                    return await controller._assess_context_compaction(
                        session_id=session.id,
                        resolution=resolution,
                        provider_messages=[*rows, {"role": "user", "content": "next"}],
                        uncommitted_user_message_id=None,
                    )

            def remembered() -> dict:
                return getattr(console._context_spend, "_console_tool_overhead_seen", {})

            # As after an app restart: nothing prepared, nothing remembered.
            controller._context_accounting_by_session.clear()
            remembered().clear()

            def native(tool_words: int) -> ToolSurfacePreview:
                return ToolSurfacePreview(
                    tools=(
                        {
                            "type": "function",
                            "function": {
                                "name": "big",
                                "description": "word " * tool_words,
                                "parameters": {"type": "object", "properties": {}},
                            },
                        },
                    )
                )

            prompt = controller.store.session_settings(session.id).system_prompt
            fenced = ToolSurfacePreview(system_prompt=f"{prompt}\n\n" + "fenced " * 2000)

            if case == "D2-unmeasured":
                # Probe B: no request yet in this run. Dev's code read Ready
                # (displayed 5,600) while the real check held at 3,996.
                before = await displayed()
                hold, alert = await real_check(native(2000))
                assert alert is None and hold is not None and hold.trigger_tokens == 3996
                assert before.forecast_high_context, "a Ready that can hold"
                assert "not measured yet" in before.forecast_unavailable_reason
            elif case in {"D1-native", "D1-fenced"}:
                # Probes B2 and A: after one check the displayed trigger is the
                # real one; fenced text (the prompt delta) counts too.
                surface, trigger = (
                    (native(2000), 3996) if case == "D1-native" else (fenced, 4000)
                )
                hold, alert = await real_check(surface)
                assert alert is None and hold is not None
                assert hold.trigger_tokens == trigger, hold
                after = await displayed()
                assert after.compaction_trigger_tokens == hold.trigger_tokens, (
                    after.compaction_trigger_tokens,
                    hold,
                )
                assert after.forecast_high_context
                assert after.compaction_pause_forecast is None
            elif case == "D2-other-session":
                # This tab has no request of its own; another session's largest
                # figure under the same tool configuration stands in.
                hold, _alert = await real_check(native(2000))
                assert hold is not None and hold.trigger_tokens == 3996
                await displayed()  # The display sees the request it follows.
                controller._context_accounting_by_session.clear()
                if session.id in remembered():
                    remembered()["another-session"] = remembered().pop(session.id)
                borrowed = await displayed()
                assert borrowed.compaction_trigger_tokens == 3996, (
                    borrowed.compaction_trigger_tokens
                )
                assert borrowed.forecast_high_context
            elif case == "E-custom-fenced":
                # Probe E: a custom 6,500 budget is capped at 5,000 by the delta.
                set_policy(6500)
                hold, _alert = await real_check(fenced)
                assert hold is not None and hold.trigger_tokens == 4000
                custom = await displayed()
                assert custom.compaction_trigger_tokens == 4000, (
                    custom.compaction_trigger_tokens
                )
                assert custom.forecast_high_context
            elif case == "C3-stale-gate":
                # Probe C3: 204 tool tokens remembered, then a tool gate changes
                # and the next request carries 2,004.
                hold, _alert = await real_check(native(200))
                assert hold is None
                small = await displayed()
                assert not small.forecast_high_context  # 5,003 under 5,436
                assert config_module.save_settings_to_cli_config(
                    {"tools": {"read_file_enabled": True}}
                )
                try:
                    app.app_config = config_module.load_settings(force_reload=True)
                    stale = await displayed()
                    hold, _alert = await real_check(native(2000))
                finally:
                    assert config_module.save_settings_to_cli_config(
                        {"tools": {"read_file_enabled": False}}
                    )
                    app.app_config = config_module.load_settings(force_reload=True)
                assert hold is not None and hold.trigger_tokens == 3996
                assert stale.forecast_high_context, "a stale tool figure read Ready"
                assert "not measured yet" in stale.forecast_unavailable_reason
            elif case == "cannot-fit":
                # Probe D: a system prompt that leaves no room. The send is
                # refused ("cannot make room"); dev's display read Ready.
                await set_prompt(10496)
                full = await displayed()
                hold, alert = await real_check(None)
                assert hold is None and alert is not None, (hold, alert)
                assert getattr(full, "compaction_cannot_fit", False), "read Ready"
                assert not getattr(full, "compaction_fit_uncertain", False)
                shown = build_console_context_cost_state(full, _COST)
                assert shown.label.startswith("Next send won't fit · "), shown.label
                assert "compacting older turns cannot make room" in shown.tooltip
                # The note the header, Inspect and run chip read, for this state.
                console._context_spend._last_console_context_control_state = full
                assert console._context_spend.compaction_forecast_note() == (
                    "Next send won't fit"
                )
            elif case == "may-not-fit":
                # Review 2026-11-23: the static input alone fits (8,004 of
                # 10,000); only the remembered 2,004 tool tokens tip it over,
                # and the send may carry fewer tools. Hedged, not firm.
                await set_prompt(8000)
                await real_check(native(2000))
                tight = await displayed()
                assert tight.compaction_cannot_fit
                assert getattr(tight, "compaction_fit_uncertain", False), "firm"
                shown = build_console_context_cost_state(tight, _COST)
                assert shown.label.startswith("Next send may not fit · "), shown.label
                assert "may carry fewer tools" in shown.tooltip
                console._context_spend._last_console_context_control_state = tight
                assert console._context_spend.compaction_forecast_note() == (
                    "Next send may not fit"
                )
            elif case == "compact-keeps-max":
                # Review 2026-11-23: a later request without the tools (as a
                # /compact summary is) must not lower the remembered figure.
                # Persistence is held off here so only that guard keeps it.
                spend_owner = console._context_spend
                spend_owner._console_tool_costs_disk = {}
                monkeypatch.setattr(
                    spend_owner, "_console_tool_costs_io", lambda _kind: None
                )
                monkeypatch.setattr(
                    spend_owner, "_console_store_tool_cost", lambda *_args: None
                )
                await real_check(native(2000))
                measured = await displayed()
                assert measured.compaction_trigger_tokens == 3996
                await real_check(ToolSurfacePreview(tools=()))
                after = await displayed()
                assert after.compaction_trigger_tokens == 3996, (
                    "a request without tools lowered the remembered cost",
                    after.compaction_trigger_tokens,
                )
            elif case == "B-persisted":
                # Owner decision B (2026-11-23): persisted per configuration,
                # reloaded off the loop after a restart and used; another
                # configuration does not use it; a corrupt file is ignored.
                spend_owner = console._context_spend

                async def settled(predicate) -> None:
                    for _ in range(100):
                        if predicate():
                            return
                        await pilot.pause(0.05)
                    assert predicate()

                hold, _alert = await real_check(native(2000))
                await displayed()
                await settled(lambda: console_tool_cost_file.exists())
                await settled(lambda: not spend_owner._console_tool_costs_busy)
                assert json.loads(console_tool_cost_file.read_text())["costs"][0][1] == 2004

                def restart() -> None:
                    controller._context_accounting_by_session.clear()
                    remembered().clear()
                    spend_owner._console_tool_costs_disk = None

                restart()
                first = await displayed()  # the read is still in flight
                assert first.forecast_high_context
                await settled(lambda: spend_owner._console_tool_costs_disk is not None)
                reloaded = await displayed()
                assert reloaded.compaction_trigger_tokens == hold.trigger_tokens == 3996
                assert reloaded.request_overhead_tokens == 3000 + 2004
                # Another configuration: the stored figure is not used.
                assert config_module.save_settings_to_cli_config(
                    {"tools": {"read_file_enabled": True}}
                )
                try:
                    app.app_config = config_module.load_settings(force_reload=True)
                    other = await displayed()
                finally:
                    assert config_module.save_settings_to_cli_config(
                        {"tools": {"read_file_enabled": False}}
                    )
                    app.app_config = config_module.load_settings(force_reload=True)
                assert other.request_overhead_tokens == 3000
                assert "not measured yet" in other.forecast_unavailable_reason
                # A corrupt file reads as nothing stored: unmeasured, no crash.
                console_tool_cost_file.write_text("{not json")
                restart()
                await displayed()
                await settled(lambda: spend_owner._console_tool_costs_disk is not None)
                corrupt = await displayed()
                assert spend_owner._console_tool_costs_disk == {}
                assert corrupt.request_overhead_tokens == 3000
                assert corrupt.forecast_high_context
            elif case in ("upgrade-figure", "internal-prompt-figure"):
                # Review 2026-11-23 (blocking 3): a figure saved by an older
                # app version, or before the agent's operating prompt was
                # overridden ([internal_prompts]), can be smaller than today's
                # tools. It must not be used (dev's signature ignored both).
                from tldw_chatbook.UI.Console_Modules import context_spend as cs

                spend_owner = console._context_spend
                settings = controller.store.session_settings(session.id)
                key = cs._tool_cost_key(
                    spend_owner._console_tool_signature(settings, session.id)
                )
                controller._context_accounting_by_session.clear()
                remembered().clear()
                spend_owner._console_tool_costs_disk = {key: 2004}
                saved = await displayed()
                assert saved.request_overhead_tokens == 3000 + 2004  # used as is
                if case == "upgrade-figure":
                    import tldw_chatbook

                    monkeypatch.setattr(
                        tldw_chatbook, "__version__", f"{tldw_chatbook.__version__}.1"
                    )
                else:
                    overridden = dict(app.app_config)
                    overridden["internal_prompts"] = {
                        "agents.console_agent_operating": "longer " * 1500
                    }
                    monkeypatch.setattr(app, "app_config", overridden)
                changed = await displayed()
                assert changed.request_overhead_tokens == 3000, (
                    "a figure saved under another app version or operating "
                    "prompt was used",
                    changed.request_overhead_tokens,
                )
                assert changed.forecast_high_context  # unmeasured: may pause
            else:
                # Review 2026-11-23: an armed raw shell grows the light tool
                # set, so it is part of the tool configuration.
                spend_owner = console._context_spend
                settings = controller.store.session_settings(session.id)
                before = spend_owner._console_tool_signature(settings, session.id)
                monkeypatch.setattr(
                    app, "raw_cli_runtime", SimpleNamespace(armed=True), raising=False
                )
                armed = spend_owner._console_tool_signature(settings, session.id)
                assert armed != before
            assert adapter_calls == []
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


@pytest.mark.asyncio
async def test_a_prefill_or_staged_image_is_counted_and_never_reads_a_false_ready(
    request, monkeypatch, tmp_path
):
    """Review B1 (2026-11-22): a response prefill or a staged image used to
    drop the whole static estimate, so the displayed Automatic trigger ignored
    the system prompt and the chat read Ready. Each now counts (an image at
    the send's flat per-part estimate) and only says it *may* pause."""
    from dataclasses import replace
    from Tests.UI.test_console_compaction_hold_flow import _ask_console, _held_console
    from Tests.UI.test_console_native_chat_flow import _sync_console_control_bar_now
    from tldw_chatbook.Chat import console_prepared_request, console_session_settings
    from tldw_chatbook.Chat.attachment_core import PendingAttachment
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
    from tldw_chatbook.Utils.token_counter import (
        NON_TEXT_PART_TOKEN_ESTIMATE,
        ContextWindowResolution,
    )
    from tldw_chatbook import config as config_module, model_capabilities

    def words(messages, _model, *_args, **_kwargs):
        return 3 + sum(len(str(row.get("content", "")).split()) for row in messages)

    monkeypatch.setattr(
        console_prepared_request, "count_console_messages_tokens", words
    )
    monkeypatch.setattr(console_session_settings, "_estimate_tokens_locally", words)
    window = ContextWindowResolution(tokens=10_768, verified=True, source="test")
    catalog = model_capabilities.get_model_capabilities()
    monkeypatch.setattr(
        catalog, "get_model_capabilities", lambda *_args: {"max_output_tokens": 256}
    )
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
            trust = await app.ensure_local_skill_trust_service()
            assert app.local_skills_service.trust_service is trust
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
            session.settings = replace(
                controller.store.session_settings(session.id),
                source="user",
                max_tokens=256,
                system_prompt="mandatory " * 2996,
            )
            session.settings_revision += 1
            await console._sync_native_console_chat_ui()
            console._console_composer_or_none().load_draft("next")

            async def state_and_real_hold():
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
                resolution = await gateway.resolve_for_send(
                    controller._provider_selection_for_session(session.id)
                )
                rows = controller._provider_messages_for_session(
                    session.id,
                    annotate_ids=True,
                    turn_context=controller.resolve_turn_execution_context(session.id),
                )
                hold, _alert = await controller._assess_context_compaction(
                    session_id=session.id,
                    resolution=resolution,
                    provider_messages=[*rows, {"role": "user", "content": "next"}],
                    uncommitted_user_message_id=None,
                )
                return states[-1], hold

            # Measured and under its trigger (5,003 of 5,600): Ready, exact.
            ready, hold = await state_and_real_hold()
            assert hold is None
            assert ready.request_overhead_tokens == 3000
            assert ready.compaction_trigger_tokens == 5600
            assert not ready.forecast_high_context

            # A 2,000-token one-shot prefill is counted: 4,000 of 5,000.
            controller.store.set_session_one_shot_prefill(session.id, "p " * 2000)
            prefilled, hold = await state_and_real_hold()
            assert prefilled.request_overhead_tokens == 5000, prefilled
            assert prefilled.compaction_trigger_tokens == 4000
            assert prefilled.compaction_pause_forecast is None  # may, not will
            assert prefilled.forecast_high_context, "a false Ready with a prefill"
            assert hold is None or hold.trigger_tokens >= 4000, hold
            tooltip = build_console_context_cost_state(prefilled, _COST).tooltip
            assert "The response prefill is only estimated here" in tooltip, tooltip
            controller.store.set_session_one_shot_prefill(session.id, None)

            # Three staged images count at the send's per-part estimate.
            for index in range(3):
                assert controller.store.add_pending_attachment(
                    session.id,
                    PendingAttachment(
                        file_path=f"/tmp/image-{index}.png",
                        display_name=f"image-{index}.png",
                        file_type="image",
                        insert_mode="attachment",
                        data=b"x",
                        mime_type="image/png",
                    ),
                )
            staged, hold = await state_and_real_hold()
            images = 3 * NON_TEXT_PART_TOKEN_ESTIMATE
            assert staged.request_overhead_tokens == 3000 + images, staged
            assert staged.compaction_trigger_tokens == int(0.8 * (7000 - images))
            assert staged.compaction_pause_forecast is None
            assert staged.forecast_high_context, "a false Ready with staged images"
            tooltip = build_console_context_cost_state(staged, _COST).tooltip
            assert "Images are only estimated here" in tooltip, tooltip
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
            # These preflight controls call the original synchronous capture.
            trust = await app.ensure_local_skill_trust_service()
            assert app.local_skills_service.trust_service is trust
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


def _spend_over(monkeypatch, messages):
    """A spend controller over fixed history, counting one token per word."""
    from types import SimpleNamespace
    from tldw_chatbook.UI.Console_Modules.context_spend import (
        ConsoleContextSpendController,
    )
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

    controller = ConsoleContextSpendController.__new__(ConsoleContextSpendController)
    controller._screen = object()
    controller._console_compaction_forecast_cache = None
    controller._ensure_console_chat_store = lambda: object()
    controller._ensure_console_chat_controller = lambda: object()
    controller._console_display_history = lambda *_args: (
        ("history",),
        messages,
        SimpleNamespace(request_ids=None),
    )
    controller._build_context_estimate = lambda rows, *_args, **_kwargs: (
        SimpleNamespace(
            used_tokens=sum(len(str(row["content"]).split()) for row in rows)
        )
    )
    monkeypatch.setattr(spend, "console_tokenizer_display_ready", lambda *_args: True)
    return controller


def _forecast_inputs(controller, memory):
    return controller._console_compaction_forecast_inputs(
        "session",
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
        ConsoleContextPolicyOverrides(compaction_mode=ContextCompactionMode.ASK),
        None,
        memory,
    )


def test_history_images_count_at_the_send_estimate_and_make_it_may(
    monkeypatch,
) -> None:
    """Review B1: historical media used to drop the forecast (a false Ready)."""
    from dataclasses import replace
    from tldw_chatbook.Chat.console_chat_models import MessageAttachment
    from tldw_chatbook.Utils.token_counter import NON_TEXT_PART_TOKEN_ESTIMATE

    messages = _exchanges()
    image = MessageAttachment(
        data=b"x", mime_type="image/png", display_name="a.png", position=0
    )
    messages[3] = replace(messages[3], attachments=(image, image))
    raw = EffectiveMemoryResult(EffectiveMemoryKind.RAW)
    tokens, units, causes = _forecast_inputs(_spend_over(monkeypatch, messages), raw)
    # "text of u1" ... four rows of three words, plus two image parts.
    assert tokens == 12 + 2 * NON_TEXT_PART_TOKEN_ESTIMATE
    assert units == 2
    assert causes == ("media",)
    assert _forecast_inputs(_spend_over(monkeypatch, _exchanges()), raw) == (12, 2, ())


@pytest.mark.parametrize(
    ("effective", "expected"),
    [
        # Planned differently by the preflight: count every turn in full.
        (
            EffectiveMemoryResult(
                EffectiveMemoryKind.GENERATED_RANGE, memory=_memory("p-a1")
            ),
            (12, 2, ("memory",)),
        ),
        (
            EffectiveMemoryResult(
                EffectiveMemoryKind.GENERATED_PREFIX, memory=_memory("gone")
            ),
            (12, 2, ("memory",)),
        ),
        # The preflight never asks under a legacy prefix: nothing to forecast.
        (EffectiveMemoryResult(EffectiveMemoryKind.LEGACY_PREFIX), (None, 0, ())),
    ],
    ids=["range", "boundary-off-path", "legacy"],
)
def test_an_unmodelled_memory_shape_counts_its_turns_in_full(
    monkeypatch, effective, expected
) -> None:
    """Review B2(b): these shapes forecast nothing, so they read Ready at any
    size. A range or an off-path boundary now counts every turn in full (an
    over-count) and may pause; a legacy prefix never asks, so it stays quiet."""
    controller = _spend_over(monkeypatch, _exchanges())
    assert _forecast_inputs(controller, effective) == expected


def test_stored_tool_costs_round_trip_bounded_and_fail_open(tmp_path) -> None:
    """Owner decision B (2026-11-23): the persisted agent tool costs."""
    from tldw_chatbook import console_tool_costs as store
    from tldw_chatbook.UI.Console_Modules import context_spend

    path = tmp_path / "costs.json"
    assert store.load_costs(path) == {}  # missing
    costs = {f"key-{n}": n for n in range(40)}
    store.save_costs(costs, path)
    loaded = store.load_costs(path)
    assert list(loaded) == [f"key-{n}" for n in range(8, 40)]  # newest 32
    assert loaded["key-39"] == 39
    for corrupt in (
        "{not json",
        json.dumps({"version": 2, "costs": [["a", 1]]}),
        json.dumps({"version": 1, "costs": [["a", True], ["b", -1], ["c", "9"]]}),
    ):
        path.write_text(corrupt)
        assert store.load_costs(path) == {}
    # Keys are digests: no config content reaches the file.
    assert len(context_spend._tool_cost_key('[{"api_key": "secret"}]')) == 32


def test_a_save_keeps_another_app_instances_larger_figures(tmp_path) -> None:
    """Review 2026-11-23: two app instances on one profile.

    Each instance writes its whole in-memory copy, so the later write used to
    drop or lower a figure the other had saved. A save re-reads the file and
    keeps the larger figure per key; the saving instance's keys are newest.
    """
    from tldw_chatbook import console_tool_costs as store

    path = tmp_path / "costs.json"
    store.save_costs({"a": 500, "b": 100}, path)  # instance 1
    store.save_costs({"a": 300, "c": 50}, path)  # instance 2, an older copy
    loaded = store.load_costs(path)
    assert loaded == {"b": 100, "a": 500, "c": 50}, loaded
    assert list(loaded) == ["b", "a", "c"]


def test_every_ui_test_saves_tool_costs_to_its_own_file(
    console_tool_cost_file, tmp_path
) -> None:
    """Review 2026-11-23: the Console UI tests share one data folder per
    process, so the shared UI fixture gives each test its own cost file."""
    from tldw_chatbook import console_tool_costs as store

    assert store.tool_cost_path() == console_tool_cost_file
    assert tmp_path in console_tool_cost_file.parents


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
    ) == (None, 0, ())


def test_an_agent_chat_reads_ready_below_the_trigger_and_may_pause_above() -> None:
    """TASK-33621.4 AC#3 as amended (owner decision 2026-11-20).

    The decision: "ready, then may pause". With no exact forecast (an agent
    chat's tools are not counted), dev read "Compaction forecast unavailable"
    in the header and "Forecast unavailable" in Inspect and the run chip as
    soon as the chat had history. Below the displayed trigger it is Ready;
    over it, "May pause to compact". Live, failed, blocked and stopped states
    keep their own copy. The copy is pinned literally, not imported.
    """
    from dataclasses import replace

    may_pause = "May pause to compact"
    # Review follow-up 6: the below fixture is below everywhere (dev's said
    # "~1,658 / 1,500" in the tooltip of a state meant to be under 1,200).
    below = replace(
        _state(tokens=900, units=3, used=900),
        forecast_tokens=None,
        forecast_unavailable_reason="Current tool overhead is unknown.",
    )
    shown = build_console_context_cost_state(below, _COST)
    visible = f"{shown.label}\n{shown.compact_label}\n{shown.tooltip}"
    # ("On next send unavailable" is the spend estimate's word, not this.)
    assert "forecast unavailable" not in visible.lower(), visible
    assert may_pause not in visible
    from tldw_chatbook.Widgets.Console.console_context_controls import (
        compaction_may_pause_reason,
    )

    reason = compaction_may_pause_reason(1_656, ("tools", "limits"))
    assert reason == (
        "the estimated history (~1,656 conversation tokens) is over the trigger. "
        "The agent's tools and this provider's input and output limits are only "
        "estimated here, so the send itself checks."
    )
    assert compaction_may_pause_reason(1_656, ("prefill",)).endswith(
        "The response prefill is only estimated here, so the send itself checks."
    )
    above = replace(
        _state(tokens=None, units=3),
        forecast_high_context=True,
        forecast_unavailable_reason=reason,
    )
    assert above.compaction_pause_forecast is None
    shown = build_console_context_cost_state(above, _COST)
    assert shown.label.startswith(f"{may_pause} · "), shown.label
    assert shown.compact_label.startswith(f"{may_pause} · "), shown.compact_label
    assert f"{may_pause}: {reason}" in shown.tooltip
    assert "forecast unavailable" not in shown.label.lower()

    for values, expected in (
        ({"compaction_forecast_note": may_pause}, may_pause),
        ({"compaction_forecast_note": "Next send won't fit"}, "Next send won't fit"),
        ({}, "Ready"),
        ({"compaction_forecast_note": may_pause, "run_active": True}, "Running"),
        ({"compaction_forecast_note": may_pause, "run_failed": True}, "Failed"),
        (
            {
                "compaction_forecast_note": may_pause,
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
        ), values

    def badge(**values) -> str:
        values.setdefault("control_state", _control_state())
        values.setdefault("readiness_word", "Ready")
        return build_console_workbench_state(**values).header.status_label

    assert badge() == "Ready"
    assert badge(compaction_forecast_note=may_pause) == may_pause
    assert (
        badge(
            readiness_word="Missing key",
            provider_blocker_copy="Missing API key",
            compaction_forecast_note=may_pause,
        )
        == "Missing key"
    )
    assert (
        badge(compaction_forecast_note=may_pause, blocked_turn="compaction ask")
        == "Blocked"
    )
    # Owner decision 2026-11-23 ("short badge word"): below the single-pane
    # width, where no readiness word fits, the badge used to fall back to
    # "Ready" for every note. Now a short word, and "Blocked" for won't fit.
    assert badge(compaction_forecast_note=may_pause, readiness_word="") == "May pause"
    assert (
        badge(compaction_forecast_note="Next send may not fit", readiness_word="")
        == "May not fit"
    )
    narrow = build_console_workbench_state(
        control_state=_control_state(),
        readiness_word="",
        compaction_forecast_note="Next send won't fit",
    ).header
    assert (narrow.status, narrow.status_label) == ("blocked", "Blocked")
    wide = build_console_workbench_state(
        control_state=_control_state(),
        readiness_word="Ready",
        compaction_forecast_note="Next send won't fit",
    ).header
    assert (wide.status, wide.status_label) == ("blocked", "Next send won't fit")
    from tldw_chatbook.Widgets.Console import console_workbench_state as workbench

    assert (
        badge(
            control_state=replace(_control_state(), emergency_stop_held=True),
            compaction_forecast_note=may_pause,
        )
        == workbench.CONSOLE_EMERGENCY_STOP_HEADER_LABEL
    )

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
    over = [True]
    screen = SimpleNamespace(
        _console_chat_store=store,
        _console_chat_controller=controller,
        _image=SimpleNamespace(
            _h3_image_edit_registry=lambda: SimpleNamespace(active=lambda _sid: None)
        ),
        _context_spend=SimpleNamespace(
            compaction_pause_expected=lambda: False,
            compaction_forecast_note=lambda: may_pause if over[0] else "",
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
    assert screen._console_active_run_copy() == may_pause
    over[0] = False
    assert screen._console_active_run_copy() == ""


def _state(*, tokens, units, mode=ContextCompactionMode.ASK, used=1_658):
    return build_console_context_control_state(
        settings=ConsoleSessionSettings(
            provider="anthropic", model="claude-haiku-4-5", max_tokens=4_096
        ),
        estimate=ConsoleSettingsContextEstimate(
            used_tokens=used, token_limit=200_000, label=f"{used:,} / 200,000 tokens"
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
    # The tooltip opens with the label's words (dev's said "Estimated next
    # send pauses for compaction approval:"), then the numbers and who decides.
    assert (
        "Next send pauses to compact (estimated ~1,656 of 1,500 conversation "
        "tokens); you approve before anything is summarized." in over.tooltip
    ), over.tooltip
    assert COMPACTION_PAUSE_FORECAST_COPY not in under.label
    assert COMPACTION_PAUSE_FORECAST_COPY not in under.tooltip
    assert "you approve" not in under.tooltip


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
    # Review 2026-11-23 (blocking 2): single-pane widths pass no readiness
    # word, and a definite pause read "Ready" there. It gets a short word too.
    assert badge(compaction_pause_expected=True, readiness_word="") == "Pauses"
    assert (
        badge(
            compaction_pause_expected=True,
            readiness_word="",
            provider_blocker_copy="Set up a provider",
        )
        != "Pauses"
    )


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
        run(compaction_forecast_note="May pause to compact"),  # owner, 2026-11-20
        run(compaction_forecast_note="Next send won't fit"),  # review, 2026-11-22
        run(run_blocked_reason=held),
    ):
        assert cell_len(f"Run: {line}") <= 29, line
