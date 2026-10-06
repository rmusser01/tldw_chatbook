"""TASK-34100.5.3 AC#5: the context meter counts the tools the send carries.

The Console send preflight (``ConsoleCompactionPreflight``) asks the agent
bridge for ``preview_tool_schemas`` and hands those rows to the request
accounting, whose context breakdown shows a "Tool schemas" row. The bridge
never implemented it, so the meter counted no tools while the request carried
the whole agent surface (TASK-34100.5, AC#6 notes).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_light_tool_surface import (
    LIGHT,
    _fresh_tool_gates,  # noqa: F401 -- autouse: fresh gates per test
    _native_names,
    _production_bridge,
    _send,
)
from tldw_chatbook.Chat.console_agent_bridge import build_console_first_request_plan
from tldw_chatbook.Chat.console_context_compaction import ConsoleCompactionPreflight
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.Utils.token_counter import ContextWindowResolution

pytestmark = pytest.mark.bootstrap_profile


class _Captured(Exception):
    """Stops the preflight once ``prepare`` has seen the tool rows."""


def _preflight_tools(bridge, resolution, session_id="s") -> list[dict]:
    """Drive the real capacity preflight up to ``prepare`` and return the
    ``tools=`` rows it would account."""
    names = inspect.signature(ConsoleCompactionPreflight).parameters
    kwargs = {name: (lambda: None) for name in names}
    kwargs["read_controller__agent_bridge"] = lambda: bridge
    preflight = ConsoleCompactionPreflight(**kwargs)
    seen: list[list[dict]] = []

    def prepare(_resolution, _messages, *, tools=(), **_kwargs):
        seen.append(list(tools))
        raise _Captured

    with pytest.raises(_Captured):
        preflight._assess_request_capacity_only(
            session_id=session_id,
            owner=SimpleNamespace(context_policy_overrides=None),
            resolution=resolution,
            provider_messages=[{"role": "user", "content": "Reply with just: hi"}],
            prepare=prepare,
            agent_tools_enabled=True,
            assessment_sink=lambda *_args: None,
        )
    return seen[0]


def _openai(window: ContextWindowResolution):
    return SimpleNamespace(
        provider="openai",
        execution_key="openai",
        model="gpt-4.1-nano",
        max_tokens=4096,
        context_window=window,
        reasoning_replay=None,
    )


def _llama(window: ContextWindowResolution):
    return SimpleNamespace(
        provider="llama_cpp",
        execution_key="llama_cpp",
        model="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        max_tokens=4096,
        context_window=window,
        reasoning_replay=None,
    )


class _AskUserProvider:
    """The local provider as a fresh profile composes it for the light
    surface's purposes: ``ask_user`` with its real schema."""

    def __init__(self):
        from tldw_chatbook.Agents.local_tool_provider import (
            light_surface_local_schemas,
        )

        self._schemas = {
            schema.name: schema for schema in light_surface_local_schemas(("ask_user",))
        }

    def list_catalog(self):
        from tldw_chatbook.Agents.agent_models import ToolCatalogEntry

        return [
            ToolCatalogEntry(
                id=schema.id,
                name=name,
                one_line_description=schema.description,
                source="local",
            )
            for name, schema in self._schemas.items()
        ]

    def load_schema(self, tool_id):
        return next(s for s in self._schemas.values() if s.id == tool_id)

    def invoke(self, tool_id, args):  # pragma: no cover - never dispatched
        raise AssertionError("planning only")


def _row_names(rows) -> set[str]:
    return {row["function"]["name"] for row in rows}


def test_the_preflight_counts_the_light_surface_the_send_carries(tmp_path):
    """A fresh profile, a model with a known large window: the preflight's
    accounting gets exactly the light set's rows."""
    bridge, _gateway, _session, _assistant = _production_bridge(tmp_path, [])

    rows = _preflight_tools(
        bridge, _openai(ContextWindowResolution(1_047_576, "model catalog", True))
    )

    assert _row_names(rows) == LIGHT


def test_a_window_too_small_for_the_light_set_counts_what_the_plan_sends(tmp_path):
    """A llama.cpp server whose window nobody verified is planned as 4,096
    tokens, where the light set does not fit and the request carries no
    tools. The meter agrees with the planner either way."""
    bridge, _gateway, _session, _assistant = _production_bridge(tmp_path, [])
    resolution = _llama(ContextWindowResolution(32_000, "application fallback", False))

    rows = _preflight_tools(bridge, resolution)
    # The same window with a verified, roomy size: the set fits, so the
    # all-or-nothing comparison below is not vacuous.
    roomy = _preflight_tools(
        bridge, _llama(ContextWindowResolution(32_768, "server metadata", True))
    )
    assert _row_names(roomy) == LIGHT
    plan = build_console_first_request_plan(
        shared_registry=bridge._registry,
        shared_allowed_tools=bridge._allowed_tools,
        context={},
        skills_present=False,
        mcp_provider=None,
        builtin_gate=object(),
        local_provider=_AskUserProvider(),
        library_provider=None,
        library_authority=None,
        workspace_id=None,
        ephemeral=False,
        diff_sink=None,
        scratch_root=None,
        scratch_lease=None,
        resolution=resolution,
        fallback_model=resolution.model,
        session_system_prompt="",
        native_tools=True,
        turn_skill_bindings=(),
        turn_bundle_block="",
        install_skill_enabled=False,
        run_skill_script_enabled=False,
        agent_messages=[{"role": "user", "content": "Reply with just: hi"}],
        light_surface=True,
    )
    sent = {
        schema.name
        for schema in (*plan.schemas.runtime_schemas, *plan.schemas.active_schemas)
    }

    assert _row_names(rows) == sent
    # Small local models get the minimal prompt (owner ruling): no tools,
    # and no tool talk in the system prompt.
    assert sent == set()
    assert plan.schemas.system_prompt == "You are a helpful assistant."


def test_a_local_reasoning_resolution_still_counts_the_light_set(tmp_path):
    """Live on llama.cpp (review round 1): the gateway pins a reasoning-replay
    policy on every local resolution, and with one the planner's token count
    needs an active request. Re-run with no messages, it raised inside the
    planner, the planner fell back to no tools, and the meter counted none
    for requests that carried the light set (evidence g53-r1-llama 13)."""
    from dataclasses import replace

    from Tests.Chat.test_console_agent_bridge import _test_resolution
    from tldw_chatbook.Chat.local_reasoning import resolve_reasoning_policy

    bridge, _gateway, session, _assistant = _production_bridge(tmp_path, [])
    resolution = replace(
        _test_resolution(
            model="qwen2.5-0.5b-instruct-q4_k_m.gguf",
            max_tokens=4096,
            context_window=ContextWindowResolution(8_192, "server metadata", True),
        ),
        reasoning_replay=resolve_reasoning_policy("auto", template=None),
    )

    rows = _preflight_tools(bridge, resolution, session_id=session.id)

    assert _row_names(rows) == LIGHT


def _roomy_native():
    """A native-tools resolution with a verified 1M window: the full
    catalog fits, so the planner discloses it directly."""
    from Tests.Chat.test_console_agent_bridge import _test_resolution

    return _test_resolution(
        provider="Groq",
        execution_key="groq",
        max_tokens=4096,
        context_window=ContextWindowResolution(1_047_576, "model catalog", True),
    )


def test_with_full_agent_tools_on_the_meter_counts_the_last_sent_surface(tmp_path):
    """The full surface is composed per send; once one went out, the meter
    counts exactly what it carried for the same window."""
    assert save_setting_to_cli_config("tools", "full_agent_tools_enabled", True)
    bridge, gateway, session, assistant_id = _production_bridge(tmp_path, [["hi"]])
    _send(bridge, session, assistant_id, _roomy_native())

    rows = bridge.preview_tool_schemas(_roomy_native(), session_id=session.id)

    assert _row_names(rows) == _native_names(gateway)
    assert "spawn_subagent" in _row_names(rows)
    assert "web_search" in _row_names(rows)


def test_a_full_surface_preview_is_resized_for_the_current_window(tmp_path):
    """Review round 1 (F2, major): the meter returned the rows the LAST full
    send disclosed, for any model. After a send on a 1M window, a switch to
    an 8,192-token model counted ~9k tokens of direct rows as mandatory
    input, so the preflight refused the send as unable to fit -- though the
    planner would disclose discovery tools, or none, on that window. The
    preview now re-runs the planner against the window it is asked about."""
    assert save_setting_to_cli_config("tools", "full_agent_tools_enabled", True)
    bridge, gateway, session, assistant_id = _production_bridge(tmp_path, [["hi"]])
    _send(bridge, session, assistant_id, _roomy_native())
    big = _preflight_tools(bridge, _roomy_native(), session_id=session.id)

    small = _preflight_tools(
        bridge,
        _llama(ContextWindowResolution(8_192, "server metadata", True)),
        session_id=session.id,
    )

    assert "web_search" in _row_names(big)
    # No directly disclosed catalog tool fits 10% of an 8,192-token window.
    for name in ("web_search", "watchlists_list_sources", "character_save"):
        assert name not in _row_names(small), sorted(_row_names(small))
    assert len(small) < len(big)


def test_the_meter_counts_the_user_mcp_tools_this_session_carried(tmp_path):
    """Review round 1 (C-F5): a tool from an MCP server the user added goes
    out with plain chat's light surface, but the meter counted only the
    config-derived set. It now also counts the user-source tools the
    session's last send carried -- for that session only."""
    from Tests.Chat.test_console_light_tool_surface import _SourcedMCPProvider

    bridge, gateway, session, assistant_id = _production_bridge(tmp_path, [["hi"]])
    _send(
        bridge,
        session,
        assistant_id,
        _openai_native(),
        mcp_provider=_SourcedMCPProvider(user_server=True),
    )
    assert "mcp__weather__forecast" in _native_names(gateway)
    roomy = _openai(ContextWindowResolution(1_047_576, "model catalog", True))

    rows = _preflight_tools(bridge, roomy, session_id=session.id)

    assert _row_names(rows) == LIGHT | {"mcp__weather__forecast"}
    assert _row_names(_preflight_tools(bridge, roomy, session_id="other")) == LIGHT


def _openai_native():
    from Tests.Chat.test_console_agent_bridge import _native_resolution

    return _native_resolution()


def _meter_state(tool_schema_tokens: int):
    """The Context view's state on an 8,192-token llama-server, after one
    send whose accounting carried ``tool_schema_tokens`` of tools."""
    from tldw_chatbook.Chat.console_prepared_request import (
        ConsoleRequestTokenAccounting,
    )
    from tldw_chatbook.Chat.console_session_settings import (
        ConsoleSessionSettings,
        ConsoleSettingsContextEstimate,
    )
    from tldw_chatbook.Widgets.Console.console_context_controls import (
        build_console_context_control_state,
    )

    settings = ConsoleSessionSettings(provider="llama_cpp", model="qwen", max_tokens=4096)
    estimate = ConsoleSettingsContextEstimate(
        used_tokens=14,
        token_limit=8_192,
        label="14 / 8,192 tokens; 4,096 response tokens reserved",
        token_limit_verified=True,
    )
    accounting = ConsoleRequestTokenAccounting(
        total_input_tokens=10 + tool_schema_tokens + 4,
        system_tokens=10,
        memory_tokens=0,
        mandatory_tokens=0,
        compactable_tokens=0,
        active_request_tokens=4,
        tool_schema_tokens=tool_schema_tokens,
    )
    return settings, estimate, build_console_context_control_state(
        settings=settings, estimate=estimate, accounting=accounting
    )


def test_the_settings_meter_shows_the_tool_share_of_the_last_request():
    """The Context view's "Request overhead" row read "unknown" even after a
    send. It now shows the last prepared request's non-conversation tokens
    and names the tool schemas' share."""
    _settings, _estimate, state = _meter_state(489)

    assert state.overhead_label() == "499 tokens (tool schemas 489)"


def test_the_context_meter_counts_the_tools_the_send_carried():
    """Review round 1 (B-F1, major): live on an 8,192-token llama-server, the
    request carried the light tools (733 prompt tokens), yet the status strip
    read "Context 0%" and the model tab "Request estimate · 14 / 8,192": the
    headline numbers left the tool schemas out. They now add the last
    request's tool share, and the percentage moves with the tool surface."""
    from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostState
    from tldw_chatbook.Widgets.Console.console_context_controls import (
        build_console_context_cost_state,
        context_token_row,
    )

    _settings, estimate, with_tools = _meter_state(489)
    _settings, _estimate, without_tools = _meter_state(0)

    assert with_tools.next_request_tokens == 14 + 489
    assert without_tools.next_request_tokens == 14
    # The Context view's "Next request" row (dev retired the dead
    # ``request_row`` property in TASK-33007; the row is built here).
    assert (
        context_token_row("Next request", with_tools.next_request_tokens)
        == "Next request        503 tokens"
    )
    cost = ConsoleCostState(
        label="$0.00", compact_label="$0.00", tooltip="", alert=False, cold=False
    )
    line_with = build_console_context_cost_state(with_tools, cost).tooltip
    line_without = build_console_context_cost_state(without_tools, cost).tooltip
    assert "Context: ~503 / " in line_with
    assert line_with != line_without
    assert with_tools.request_estimate_label(estimate).startswith(
        "503 / 8,192 tokens (tool schemas 489); 4,096 response tokens reserved"
    )


def test_the_tool_share_does_not_change_the_conversation_budget():
    """Review round 1 (F6): deriving the overhead from the accounting also fed
    it into capacity, so the Effective budget subtracted the system prompt
    that the Conversation figure already counts. The derived overhead is
    shown, not subtracted: the budget matches a state with no accounting."""
    from tldw_chatbook.Widgets.Console.console_context_controls import (
        build_console_context_control_state,
    )

    settings, estimate, with_tools = _meter_state(489)
    plain = build_console_context_control_state(settings=settings, estimate=estimate)

    assert with_tools.conversation_budget_tokens == plain.conversation_budget_tokens
    assert with_tools.conversation_tokens == plain.conversation_tokens == 14


@pytest.mark.asyncio
async def test_the_context_view_rows_show_the_tool_share():
    """Review round 1 (C-F3): the "Request overhead" row the user reads was
    untested; so were "Next request" and the model tab's Request estimate."""
    from textual.widgets import Collapsible, Static

    from Tests.UI.test_console_context_controls import _ContextHarness
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        ConsoleSettingsModal,
    )

    settings, estimate, state = _meter_state(489)
    app = _ContextHarness()
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config={"api_settings": {"llama_cpp": {}}},
                providers_models={"llama_cpp": ["qwen"]},
                context_estimate=estimate,
                context_state=state,
                can_save=True,
                focus_context=True,
            ),
            callback=app.capture,
        )
        await pilot.pause()
        screen = app.screen

        def text(selector):
            return str(screen.query_one(selector, Static).render())

        assert text("#console-context-overhead") == (
            "Request overhead    499 tokens (tool schemas 489)"
        )
        assert text("#console-context-next-request") == "Next request        503 tokens"
        from tldw_chatbook.Widgets.Console.console_settings_field_row import (
            REQUEST_ESTIMATE_DISCLOSURE_ID,
        )

        title = str(
            screen.query_one(f"#{REQUEST_ESTIMATE_DISCLOSURE_ID}", Collapsible).title
        )
        assert "503 / 8,192 tokens (tool schemas 489)" in title


# -- one definition: the send, the meter and the Summary agree ---------------


def _composed_local_provider(tmp_path):
    """The local provider as the controller composes it from config: none
    when the ``[console] local_tools_enabled`` master is off."""
    from tldw_chatbook.Agents.builtin_tool_gate import (
        LOCAL_TOOLS_DEFAULT_ENABLED,
        LOCAL_TOOLS_MASTER_KEY,
    )
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
    from tldw_chatbook.config import coerce_bool_setting, get_cli_setting

    if not coerce_bool_setting(
        get_cli_setting("console", LOCAL_TOOLS_MASTER_KEY, LOCAL_TOOLS_DEFAULT_ENABLED),
        LOCAL_TOOLS_DEFAULT_ENABLED,
    ):
        return None
    return LocalToolProvider(workspace_root=tmp_path, ask_user=lambda _q: None)


def _summary_titles_for(names) -> set[str]:
    from tldw_chatbook.Agents.plain_chat_tool_surface import LIGHT_SURFACE_TOOL_TITLES
    from tldw_chatbook.Agents.tool_catalog import _GATEABLE_BUILTINS

    titles = dict(LIGHT_SURFACE_TOOL_TITLES)
    titles.update({entry.tool_name: entry.title for entry in _GATEABLE_BUILTINS})
    titles["web_deep_search"] = "Deep web research"
    return {titles[name] for name in names}


def _three_readings(tmp_path):
    """What the send carried, what the meter counts, what the Summary says."""
    from tldw_chatbook.Agents.plain_chat_tool_surface import summarize_tool_surface
    from tldw_chatbook.config import load_cli_config_and_ensure_existence

    bridge, gateway, session, assistant_id = _production_bridge(tmp_path, [["hi"]])
    _send(
        bridge,
        session,
        assistant_id,
        _openai_native(),
        local_provider=_composed_local_provider(tmp_path),
    )
    roomy = _openai(ContextWindowResolution(1_047_576, "model catalog", True))
    metered = _row_names(_preflight_tools(bridge, roomy, session_id=session.id))
    summary = summarize_tool_surface(load_cli_config_and_ensure_existence())
    return _native_names(gateway), metered, set(summary.tools_on)


@pytest.mark.parametrize(
    ("gates", "expected"),
    [
        ({}, LIGHT),
        ({"tools": {"ask_user_enabled": False}}, LIGHT - {"ask_user"}),
        ({"console": {"local_tools_enabled": False}}, LIGHT - {"ask_user"}),
        ({"tools": {"web_deep_search_enabled": True}}, LIGHT | {"web_deep_search"}),
        ({"tools": {"read_file_enabled": True}}, LIGHT | {"read_file"}),
    ],
    ids=["fresh", "ask-off", "local-master-off", "deep-research-on", "read-file-on"],
)
def test_the_send_the_meter_and_the_summary_agree(tmp_path, gates, expected):
    """Review round 1 (F7): three paths derived the light surface on their
    own -- the planner's narrowing, the meter's config read and the
    Summary's wording. For each gate configuration they now agree, with the
    real local provider composed from the same config."""
    from tldw_chatbook.config import save_settings_to_cli_config

    if gates:
        assert save_settings_to_cli_config(gates)

    sent, metered, summary_titles = _three_readings(tmp_path)

    assert sent == expected, sorted(sent)
    assert metered == sent
    assert summary_titles == _summary_titles_for(sent)


@pytest.mark.parametrize(
    ("attr", "value", "gates", "dropped"),
    [
        (
            "LIGHT_SURFACE_TOOL_NAMES",
            ("get_current_datetime", "calculator"),
            {},
            "ask_user",
        ),
        (
            "LIGHT_SURFACE_LOCAL_OPT_INS",
            (),
            {"tools": {"web_deep_search_enabled": True}},
            "web_deep_search",
        ),
    ],
    ids=["safe-set", "opt-ins"],
)
def test_the_meter_and_the_summary_follow_the_one_named_set(
    tmp_path, monkeypatch, attr, value, gates, dropped
):
    """Review round 1 (F7): the meter hard-coded ``ask_user`` and
    ``web_deep_search`` and the Summary hard-coded "Deep web research", so a
    change to the named set changed what was sent but not what was counted
    or said. Narrowing the set now narrows all three."""
    from tldw_chatbook.Agents import plain_chat_tool_surface as surface
    from tldw_chatbook.config import save_settings_to_cli_config

    if gates:
        assert save_settings_to_cli_config(gates)
    monkeypatch.setattr(surface, attr, value)

    sent, metered, summary_titles = _three_readings(tmp_path)

    assert dropped not in sent
    assert metered == sent
    assert summary_titles == _summary_titles_for(sent)


# -- the MCP hub's tool kill switch -------------------------------------------


def test_with_the_tool_kill_switch_on_the_meter_counts_no_local_tools(tmp_path):
    """Review round 1 (F7, last part): with the MCP hub's kill switch engaged
    the controller composes no local tools provider
    (``_compose_local_tool_provider`` returns ``None`` when
    ``get_kill_switch()`` is true), so ``ask_user`` is not sent. The meter
    read the light set from config alone and counted it anyway; the Tools
    chip did the same. The switch is read per preview, so turning it back
    off counts ``ask_user`` again."""
    from Tests.Chat.test_console_light_tool_surface import (
        ChatScreen,
        _attach_real_dbs,
        _build_test_app,
    )
    from tldw_chatbook.Agents.plain_chat_tool_surface import (
        offered_builtin_tool_count,
    )

    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    engaged = [False]
    app.unified_mcp_service = SimpleNamespace(get_kill_switch=lambda: engaged[0])
    bridge = ChatScreen(app)._ensure_console_agent_bridge()
    roomy = _openai(ContextWindowResolution(1_047_576, "model catalog", True))
    assert _row_names(_preflight_tools(bridge, roomy)) == LIGHT
    assert offered_builtin_tool_count(bridge) == len(LIGHT)

    engaged[0] = True

    assert _row_names(_preflight_tools(bridge, roomy)) == LIGHT - {"ask_user"}
    assert offered_builtin_tool_count(bridge) == len(LIGHT) - 1

    engaged[0] = False

    assert _row_names(_preflight_tools(bridge, roomy)) == LIGHT


def test_the_meter_counts_the_agent_surface_while_the_conversations_children_run(
    tmp_path,
):
    """Review round 1 (F8, meter half): live sub-agents keep the agent
    surface on the next send after Full agent tools is switched off. The
    send looks their fleet up by conversation, but the meter's preview asked
    without one, so it counted the light set while the send carried the
    agent surface. The preview now resolves the session's conversation the
    way the controller does (persisted id, else the session id)."""
    from Tests.Chat.test_console_light_tool_surface import _LiveFleet

    bridge, _gateway, session, _assistant = _production_bridge(tmp_path, [])
    roomy = _openai(ContextWindowResolution(1_047_576, "model catalog", True))
    conversation_id = session.persisted_conversation_id or session.id
    assert _row_names(_preflight_tools(bridge, roomy, session_id=session.id)) == LIGHT

    bridge._fleet_coordinators[conversation_id] = _LiveFleet()

    rows = _row_names(_preflight_tools(bridge, roomy, session_id=session.id))
    assert "spawn_subagent" in rows, sorted(rows)
    assert _row_names(_preflight_tools(bridge, roomy, session_id="other")) == LIGHT
