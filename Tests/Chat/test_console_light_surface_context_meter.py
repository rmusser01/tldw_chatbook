"""TASK-34100.5.3 AC#5: the context meter counts the tools the send carries.

The Console send preflight (``ConsoleCompactionPreflight``) asks the agent
bridge for ``preview_tool_schemas`` and hands those rows to the request
accounting, whose context breakdown shows a "Tool schemas" row. The bridge
never implemented it, so the meter counted no tools while the request carried
the whole agent surface (TASK-34100.5, AC#6 notes).
"""

from __future__ import annotations

import asyncio
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


def _preflight_tools(bridge, resolution, session_id="s", messages=None) -> list[dict]:
    """Drive the real capacity preflight up to ``prepare`` and return the
    ``tools=`` rows it would account for ``messages`` (default: one short
    user message)."""
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
            provider_messages=messages
            or [{"role": "user", "content": "Reply with just: hi"}],
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


@pytest.mark.parametrize(
    ("member", "title", "gate"),
    [
        ("read_file", "Read file", "read_file_enabled"),
        ("web_deep_search", "Deep web research", "web_deep_search_enabled"),
    ],
    ids=["gated-builtin", "local-opt-in"],
)
@pytest.mark.parametrize("gate_on", [False, True], ids=["gate-off", "gate-on"])
def test_a_widened_safe_set_is_sent_counted_and_named_alike(
    tmp_path, monkeypatch, member, title, gate, gate_on
):
    """Review round 2 (R2-3): the F7 parity held only for narrowing. The
    Summary treated ``ask_user`` as the safe set's only conditional member,
    so a gated tool added to the set was named as on with its gate off --
    while the send, which obeys the gate, did not carry it. Widening the set
    now keeps the send, the meter and the Summary in step either way."""
    from tldw_chatbook.Agents import plain_chat_tool_surface as surface
    from tldw_chatbook.config import save_settings_to_cli_config

    monkeypatch.setattr(
        surface, "LIGHT_SURFACE_TOOL_NAMES", (*surface.LIGHT_SURFACE_TOOL_NAMES, member)
    )
    monkeypatch.setattr(
        surface,
        "LIGHT_SURFACE_TOOL_TITLES",
        {**surface.LIGHT_SURFACE_TOOL_TITLES, member: title},
    )
    if gate_on:
        assert save_settings_to_cli_config({"tools": {gate: True}})

    sent, metered, summary_titles = _three_readings(tmp_path)

    assert (member in sent) is gate_on, sorted(sent)
    assert metered == sent
    assert summary_titles == _summary_titles_for(sent)


@pytest.mark.parametrize(
    ("attr", "name"),
    [
        ("LIGHT_SURFACE_TOOL_NAMES", "unspecced_local_tool"),
        ("LIGHT_SURFACE_LOCAL_OPT_INS", "unspecced_opt_in"),
    ],
    ids=["safe-set", "opt-ins"],
)
def test_a_light_tool_nothing_can_describe_fails_loudly(monkeypatch, attr, name):
    """Review round 2 (R2-3): the meter's local table and the Summary's gate
    table silently skipped a name they did not know, so a tool added to the
    named set would be sent but neither counted nor named, and nothing
    failed. Both now refuse it."""
    from tldw_chatbook.Agents import plain_chat_tool_surface as surface

    monkeypatch.setattr(surface, attr, (*getattr(surface, attr), name))
    monkeypatch.setattr(
        surface,
        "LIGHT_SURFACE_TOOL_TITLES",
        {**surface.LIGHT_SURFACE_TOOL_TITLES, name: name},
    )

    with pytest.raises((KeyError, ValueError), match=name):
        surface.light_surface_tool_schemas()
    with pytest.raises((KeyError, ValueError), match=name):
        surface.summarize_tool_surface({})


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


# -- review round 2: the meter sizes the tools against the request itself ----


def _gpt4_8k(window: int = 8_192):
    """gpt-4's shape: native tools, a verified 8,192-token window and 4,096
    tokens reserved for the reply (the live V2-1 model)."""
    from Tests.Chat.test_console_agent_bridge import _test_resolution

    return _test_resolution(
        provider="OpenAI",
        execution_key="openai",
        model="gpt-4",
        max_tokens=4096,
        context_window=ContextWindowResolution(window, "model catalog", True),
    )


def _words(count: int) -> str:
    return " ".join(["word"] * count)


@pytest.mark.parametrize(
    ("full", "window", "system_words", "user_words", "shape"),
    [
        # Planned within the gateway's own ceiling (window - reply reserve -
        # safety margin: 3,584 tokens on 8k), so the full surface's direct
        # set needs a larger window than gpt-4's.
        (True, 32_768, 0, 5, "direct"),
        (True, 8_192, 0, 5, "discovery"),
        # Too long beside the full discovery surface: ADR-253 sends compact
        # discovery (find/load plus the progress reader), not plain chat.
        (True, 8_192, 0, 600, "compact"),
        (False, 8_192, 0, 5, "light"),
        # Under the 3,584-token ceiling alone, over it beside the light set.
        (False, 8_192, 0, 3300, "none"),
        (False, 8_192, 3300, 5, "none"),
    ],
    ids=[
        "full-short-32k",
        "full-short",
        "full-medium",
        "light-short",
        "light-long-message",
        "light-long-session-prompt",
    ],
)
def test_the_preflight_counts_what_the_send_carries_for_this_conversation(
    tmp_path, full, window, system_words, user_words, shape
):
    """Review round 2 (V2-1, major; R2-1): the meter re-planned the tools
    against a one-row stand-in request, so it counted the surface that fits
    an EMPTY conversation. Live on gpt-4 (8k) with Full agent tools on, it
    counted 3,351 tokens of discovery tools and held a four-message chat for
    compaction, while the send -- planned against the real conversation --
    carried only ``read_agent_messages`` (OpenAI reported 510 input tokens).
    The preflight now hands the request it is assessing to the preview, so
    for every conversation size the rows it counts are the rows the send
    carries."""
    if full:
        assert save_setting_to_cli_config("tools", "full_agent_tools_enabled", True)
    bridge, gateway, session, assistant_id = _production_bridge(
        tmp_path, [["hi"], ["hi"]]
    )
    if full:
        _send(bridge, session, assistant_id, _roomy_native())
    session_prompt = _words(system_words) if system_words else ""
    request = [{"role": "user", "content": _words(user_words)}]
    system = [{"role": "system", "content": session_prompt}] if session_prompt else []

    metered = _row_names(
        _preflight_tools(
            bridge,
            _gpt4_8k(window),
            session_id=session.id,
            messages=[*system, *request],
        )
    )
    # A light send composes the real ``ask_user`` schema, as production does
    # (the shared harness's local stand-in has a one-word schema, which fits
    # where the real one does not). A full send keeps the harness provider
    # the remembered full surface was composed from.
    light_provider = {} if full else {"local_provider": _AskUserProvider()}
    _send(
        bridge,
        session,
        assistant_id,
        _gpt4_8k(window),
        agent_messages=request,
        session_system_prompt=session_prompt,
        **light_provider,
    )
    sent = _native_names(gateway, gateway.calls - 1)

    assert metered == sent, (sorted(metered), sorted(sent))
    if shape == "direct":
        assert "spawn_subagent" in sent and "find_tools" not in sent, sorted(sent)
    elif shape == "discovery":
        assert "find_tools" in sent, sorted(sent)
    elif shape == "compact":
        assert sent == {"find_tools", "load_tools", "read_agent_messages"}, sorted(
            sent
        )
    elif shape == "light":
        assert sent == LIGHT, sorted(sent)
    else:
        assert sent == set(), sorted(sent)


def test_a_first_message_that_fits_without_the_tools_is_not_refused(tmp_path):
    """Review round 2 (R2-1): sized against the stand-in, the light set
    (about 780 fenced tokens on a verified 8,192-token llama.cpp window)
    joined a long first message as mandatory input, so the capacity check
    refused it before commit ("This request cannot fit the selected model
    ... reduce mandatory context"). The send's own planner drops the tools
    for that message, and the request fits without them: it must go out."""
    from Tests.Chat.test_console_agent_bridge import _test_resolution
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

    bridge, _gateway, session, _assistant = _production_bridge(tmp_path, [])
    resolution = _test_resolution(
        model="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        max_tokens=4096,
        context_window=ContextWindowResolution(8_192, "server metadata", True),
    )
    controller = ConsoleChatController(store=bridge._store, provider_gateway=None)
    controller._agent_bridge = bridge
    verdicts = []

    controller._assess_request_capacity_only(
        session_id=session.id,
        owner=SimpleNamespace(context_policy_overrides=None),
        resolution=resolution,
        provider_messages=[{"role": "user", "content": _words(3400)}],
        prepare=ConsoleProviderGateway(environ={}).prepare_chat_request,
        agent_tools_enabled=True,
        assessment_sink=lambda hold, decision, alert: verdicts.append(alert),
    )

    assert verdicts == [None]


def test_a_first_message_in_the_gateways_safety_margin_goes_out_without_tools(
    tmp_path,
):
    """Review round 2 (R2-1's residual band; V2-2's cause). The planner kept
    a tool set whenever the request fit ``window - reply reserve``, but the
    gateway refuses anything over ``window - reply reserve - safety margin``
    (``resolve_request_capacity``: at least 512 tokens). A first message
    whose request fit tool-less, but not beside the tools inside that margin,
    was refused before commit; with Full agent tools on an 8k model the same
    gap failed sends outright ("Mandatory Console request material exceeds
    the effective input ceiling (4033 > 3584)"). The planner now plans within
    the gateway's own ceiling, so the tools are dropped and the message
    goes out."""
    from Tests.Chat.test_console_agent_bridge import _test_resolution
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

    bridge, _gateway, session, _assistant = _production_bridge(tmp_path, [])
    resolution = _test_resolution(
        model="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        max_tokens=4096,
        context_window=ContextWindowResolution(8_192, "server metadata", True),
    )
    prepare = ConsoleProviderGateway(environ={}).prepare_chat_request
    # ~3,150 tokens: under the gateway's 3,584-token ceiling alone, over it
    # beside the ~490-token light set, and under the planner's old
    # 4,096-token test either way.
    message = [{"role": "user", "content": _words(3150)}]
    assert not prepare(resolution, message, apply_safety_window=False).known_overflow
    controller = ConsoleChatController(store=bridge._store, provider_gateway=None)
    controller._agent_bridge = bridge
    verdicts = []

    controller._assess_request_capacity_only(
        session_id=session.id,
        owner=SimpleNamespace(context_policy_overrides=None),
        resolution=resolution,
        provider_messages=message,
        prepare=prepare,
        agent_tools_enabled=True,
        assessment_sink=lambda hold, decision, alert: verdicts.append(alert),
    )

    assert verdicts == [None]
    assert bridge.preview_tool_schemas(
        resolution, session_id=session.id, messages=message
    ) == []


# -- review round 3: the safe set survives capacity trimming (R3-3) ---------


def _native_window(window: int):
    """A native-tools resolution with a verified ``window`` and a small reply
    reserve, so the whole request fits and only the 10% direct-disclosure
    share decides which tools go out."""
    from Tests.Chat.test_console_agent_bridge import _test_resolution

    return _test_resolution(
        provider="Groq",
        execution_key="groq",
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        context_window=ContextWindowResolution(window, "model catalog", True),
    )


def _light_schema_tokens(resolution) -> int:
    from tldw_chatbook.Agents.agent_service import catalog_schema_tokens
    from tldw_chatbook.Agents.plain_chat_tool_surface import (
        light_surface_tool_schemas,
    )

    return catalog_schema_tokens(
        light_surface_tool_schemas(),
        model=resolution.model,
        api_endpoint=resolution.execution_key,
        native_tools=True,
    )


def test_switching_a_tool_on_never_leaves_fewer_tools_than_the_default(tmp_path):
    """Review round 3 (R3-3): the light plan's direct disclosure was all or
    nothing, with no discovery fallback, so on a window whose 10% share held
    the safe set but not the safe set plus one switched-on tool, turning that
    tool on left plain chat with NO tools -- fewer than the default. Live on
    an 8,192-token llama.cpp server: the light set measured 780 fenced tokens
    against an 819-token share, and 907 with Read file on (evidence
    g53v3-llama 10, 11). The safe set now survives the trimming ahead of the
    opt-in extras, and the meter counts what the send carries."""
    probe = _native_window(8_192)
    light_tokens = _light_schema_tokens(probe)
    assert save_setting_to_cli_config("tools", "read_file_enabled", True)
    with_read_tokens = _light_schema_tokens(probe)
    assert light_tokens < with_read_tokens
    # The 10% share holds the safe set, but not the safe set plus Read file.
    window = (light_tokens + with_read_tokens) // 2 * 10
    resolution = _native_window(window)
    request = [{"role": "user", "content": "Reply with just: hi"}]
    bridge, gateway, session, assistant_id = _production_bridge(
        tmp_path, [["hi"], ["hi"]]
    )

    _send(
        bridge,
        session,
        assistant_id,
        resolution,
        agent_messages=request,
        local_provider=_AskUserProvider(),
    )
    sent = _native_names(gateway, gateway.calls - 1)
    metered = _row_names(
        _preflight_tools(bridge, resolution, session_id=session.id, messages=request)
    )

    assert sent == LIGHT, sorted(sent)
    assert metered == sent, (sorted(metered), sorted(sent))

    # Control: where the share holds both, the switched-on tool goes out too.
    _send(
        bridge,
        session,
        assistant_id,
        _native_window(with_read_tokens * 10 + 100),
        agent_messages=request,
        local_provider=_AskUserProvider(),
    )
    assert _native_names(gateway, gateway.calls - 1) == LIGHT | {"read_file"}


# -- review round 3: the history preflight -- every send after the first (TE-1)


async def _history_preflight(bridge, resolution, words_per_row, *, memory=False):
    """Drive the real history-path preflight
    (``_apply_conversation_memory_preflight``) for a session holding two
    exchanges of ``words_per_row`` words each plus the current request.

    Returns ``(rows the preflight sized the tools as, messages it hands the
    send)``. Compaction is off, so the preflight returns the request as the
    send gets it; with ``memory`` a stored summary replaces both exchanges.
    """
    from Tests.Chat.test_console_context_compaction import (
        _ControllerGateway,
        _ControllerPersistence,
        _ControllerRepository,
        _memory,
    )
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_context_policy import (
        ConsoleContextPolicyOverrides,
        ContextCompactionMode,
    )

    persistence = _ControllerPersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session()
    store.persist_session_if_needed(session.id)
    store.set_session_context_policy_overrides(
        session.id,
        ConsoleContextPolicyOverrides(compaction_mode=ContextCompactionMode.OFF),
    )
    for index in range(2):
        for role in (ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT):
            store.append_message(
                session.id,
                role=role,
                content=f"{role.value}-{index} " + _words(words_per_row),
                persist=True,
            )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Reply with just: hi",
        persist=True,
    )
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="", persist=True
    )
    gateway = _ControllerGateway(context_window_tokens=resolution.context_window.tokens)
    repository = _ControllerRepository(persistence)
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, context_repository=repository
    )
    controller._agent_bridge = bridge
    if memory:
        snapshots = controller._durable_context_snapshots(session.id)
        repository.memories.append(
            _memory(snapshots, boundary=snapshots[-2].message_id)
        )
    sized: list[list[dict]] = []
    prepare = gateway.prepare_chat_request

    def recording(resolution_, messages, *, tools=None, **kwargs):
        if tools is not None:
            sized.append(list(tools))
        return prepare(resolution_, messages, tools=tools, **kwargs)

    gateway.prepare_chat_request = recording
    output, blocked = await controller._apply_conversation_memory_preflight(
        session_id=session.id,
        resolution=resolution,
        provider_messages=controller._provider_messages_for_session(
            session.id, annotate_ids=True
        ),
        assistant_message_id=assistant.id,
        agent_tools_enabled=True,
    )
    assert blocked is None, blocked
    assert sized, "the preflight never sized the request's tools"
    return sized[0], output


@pytest.mark.parametrize(
    ("full", "words_per_row", "memory", "shape"),
    [
        (False, 20, False, "light"),
        # The history fits the 3,584-token ceiling alone, not beside the
        # light set; the last row alone would fit beside it.
        (False, 800, False, "none"),
        # The stored summary replaces that history: the light set fits again.
        (False, 800, True, "light"),
        # The direct set and discovery no longer fit beside the history; the
        # last row alone leaves room for discovery. ADR-253 then sends
        # compact discovery (find/load plus the progress reader).
        (True, 150, False, "compact"),
    ],
    ids=["light-short-history", "light-long-history", "light-memory", "full-medium"],
)
def test_the_history_preflight_counts_what_the_send_carries(
    tmp_path, full, words_per_row, memory, shape
):
    """Review round 3 (TE-1, major): every send after the first goes through
    the history preflight, and nothing guarded its tool count. Sized against
    one row instead of the conversation -- the stand-in shape V2-1 fixed on
    the first-message path -- it counted 3,351 tokens of discovery tools for
    a four-message gpt-4 chat the send carried only the inbox reader for.
    Here the rows the preflight hands ``prepare(tools=...)`` must be the rows
    the send then carries, for the request the preflight returns: the
    retained history plus the memory row that replaces older turns."""
    if full:
        assert save_setting_to_cli_config("tools", "full_agent_tools_enabled", True)
    bridge, gateway, session, assistant_id = _production_bridge(
        tmp_path, [["hi"], ["hi"]]
    )
    if full:
        _send(bridge, session, assistant_id, _roomy_native())
    resolution = _gpt4_8k()

    sized, request = asyncio.run(
        _history_preflight(bridge, resolution, words_per_row, memory=memory)
    )
    if memory:
        assert "Earlier facts." in str(request)
        assert _words(words_per_row) not in str(request)
    light_provider = {} if full else {"local_provider": _AskUserProvider()}
    _send(
        bridge,
        session,
        assistant_id,
        resolution,
        agent_messages=[dict(row) for row in request],
        **light_provider,
    )
    sent = _native_names(gateway, gateway.calls - 1)

    assert _row_names(sized) == sent, (sorted(_row_names(sized)), sorted(sent))
    if shape == "light":
        assert sent == LIGHT, sorted(sent)
    elif shape == "compact":
        assert sent == {"find_tools", "load_tools", "read_agent_messages"}, sorted(
            sent
        )
    else:
        assert sent == set(), sorted(sent)
