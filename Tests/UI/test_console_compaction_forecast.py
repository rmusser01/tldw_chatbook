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
        ({"compaction_may_pause": True}, may_pause),
        ({}, "Ready"),
        ({"compaction_may_pause": True, "run_active": True}, "Running"),
        ({"compaction_may_pause": True, "run_failed": True}, "Failed"),
        (
            {"compaction_may_pause": True, "run_blocked_reason": "recovery needed"},
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
    assert badge(compaction_may_pause=True) == may_pause
    assert (
        badge(
            readiness_word="Missing key",
            provider_blocker_copy="Missing API key",
            compaction_may_pause=True,
        )
        == "Missing key"
    )
    assert badge(compaction_may_pause=True, blocked_turn="compaction ask") == "Blocked"
    # Single-pane widths drop the readiness word because it overruns the row.
    assert badge(compaction_may_pause=True, readiness_word="") == ""
    from tldw_chatbook.Widgets.Console import console_workbench_state as workbench

    assert (
        badge(
            control_state=replace(_control_state(), emergency_stop_held=True),
            compaction_may_pause=True,
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
            compaction_may_pause=lambda: over[0],
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
        run(compaction_may_pause=True),  # owner decision 2026-11-20
        run(run_blocked_reason=held),
    ):
        assert cell_len(f"Run: {line}") <= 29, line
