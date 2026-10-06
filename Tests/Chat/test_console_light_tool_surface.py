"""TASK-34100.5.3: plain Console chat offers a light, safe tool surface.

Owner ruling, 2026-10-04 ("Light surface by default"): plain Console chat
keeps a small, safe set (date/time, calculator, ask-user) and offers no
sub-agents, skills, web or other autonomous or costly tools until the user
turns them on.

Live on TASK-34100.5's branch, a fresh profile with every tool gate off
handed gpt-4.1-nano sub-agent, skill, run-log, chat, web, Watchlists, todo
and character tools plus chatbook's own MCP source, and the model answered
"Reply with just: hi" by calling ``spawn_subagent`` (evidence g5-v-oai 03,
g5-r2b-oaihome 04).

These tests drive the bridge the app actually builds
(``ChatScreen._ensure_console_agent_bridge`` -> ``ConsoleRuntime``) on an
isolated fresh profile, so they also prove production wiring: a bridge built
anywhere else would not show the light surface.
"""

from __future__ import annotations

import pytest

from Tests.Chat.test_console_agent_bridge import (
    _ChunkGateway,
    _FakeMCPProvider,
    _fence,
    _native_resolution,
    _test_resolution,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from tldw_chatbook.Agents.agent_models import (
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.MCP.builtin_tool_policy import BUILTIN_MCP_SERVER_KEY
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = pytest.mark.bootstrap_profile

LIGHT = {"get_current_datetime", "calculator", "ask_user"}

#: The autonomous/costly surface a fresh profile used to get.
AGENT_ONLY = {
    "spawn_subagent",
    "fork_chat",
    "new_chat",
    "web_search",
    "web_fetch",
    "watchlists_list_sources",
    "todo_create",
    "character_save",
    "mcp__tldw_chatbook__create_note",
    "mcp__tldw_chatbook__chat_with_character",
}


def fresh_profile_gates() -> dict[str, dict[str, bool]]:
    """Every gate ``all_tool_gates()`` enumerates, at a fresh profile's value.

    Review round 1 (F8): the exact-set assertions here depend on EVERY gate
    (ask_user, the local master, web_deep_search, the built-ins), not only
    the two a test flips, so all of them are reset.
    """
    from tldw_chatbook.Agents.builtin_tool_gate import (
        LOCAL_TOOLS_DEFAULT_ENABLED,
        LOCAL_TOOLS_MASTER_KEY,
        _gate_key_pairs,
    )
    from tldw_chatbook.Agents.local_tool_provider import (
        ASK_USER_DEFAULT_ENABLED,
        ASK_USER_GATE_KEY,
        CHARACTER_TOOLS_DEFAULT_ENABLED,
        CHARACTER_TOOLS_GATE_KEY,
    )

    on_by_default = {
        ("console", LOCAL_TOOLS_MASTER_KEY): LOCAL_TOOLS_DEFAULT_ENABLED,
        ("tools", ASK_USER_GATE_KEY): ASK_USER_DEFAULT_ENABLED,
        ("tools", CHARACTER_TOOLS_GATE_KEY): CHARACTER_TOOLS_DEFAULT_ENABLED,
    }
    sections: dict[str, dict[str, bool]] = {}
    for section, key in _gate_key_pairs():
        sections.setdefault(section, {})[key] = on_by_default.get(
            (section, key), False
        )
    return sections


@pytest.fixture(autouse=True)
def _fresh_tool_gates():
    """Every test starts on a fresh profile's gates and leaves them that way
    (the isolated config file outlives one test)."""
    from tldw_chatbook.config import save_settings_to_cli_config

    assert save_settings_to_cli_config(fresh_profile_gates())
    yield
    assert save_settings_to_cli_config(fresh_profile_gates())


class _LocalLikeProvider:
    """The local workspace/web/Watchlists provider a fresh profile composes:
    ``ask_user`` beside web, Watchlists, todo and character tools."""

    names = (
        "web_search",
        "web_fetch",
        "watchlists_list_sources",
        "todo_create",
        "ask_user",
        "character_save",
    )

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id=f"local:{name}",
                name=name,
                one_line_description=f"{name} tool",
                source="local",
            )
            for name in self.names
        ]

    def load_schema(self, tool_id):
        name = tool_id.split(":", 1)[1]
        return ToolSchema(
            id=tool_id,
            name=name,
            description=f"{name} tool",
            parameters={"type": "object", "properties": {}},
        )

    def invoke(self, tool_id, args):
        return ToolResult(ok=True, content="ok")


class _SourcedMCPProvider(_FakeMCPProvider):
    """chatbook's own built-in MCP source (on every fresh profile), plus
    optionally one server the user added."""

    def __init__(self, *, user_server: bool = False):
        entries = [
            ("mcp__tldw_chatbook__create_note", "Create a note"),
            ("mcp__tldw_chatbook__chat_with_character", "Chat as a character"),
        ]
        if user_server:
            entries.append(("mcp__weather__forecast", "Forecast from the user's server"))
        super().__init__(entries)

    def server_key_for(self, name):
        if name.startswith("mcp__tldw_chatbook__"):
            return BUILTIN_MCP_SERVER_KEY
        if name.startswith("mcp__weather__"):
            return "local:weather"
        return None


def _production_bridge(tmp_path, scripts):
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    screen = ChatScreen(app)
    bridge = screen._ensure_console_agent_bridge()
    assert bridge is not None, "the harness must build the app's real bridge"
    gateway = _ChunkGateway(scripts)
    bridge._gateway = gateway
    store = bridge._store
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    return bridge, gateway, session, assistant.id


def _send(bridge, session, assistant_id, resolution, **over):
    from uuid import uuid4

    kwargs = dict(
        conversation_id="conv-light",
        session_id=session.id,
        resolution=resolution,
        assistant_message_id=assistant_id,
        model="test-model",
        session_system_prompt="",
        agent_messages=[{"role": "user", "content": "Reply with just: hi"}],
        should_cancel=lambda: False,
        mcp_provider=_SourcedMCPProvider(),
        local_provider=_LocalLikeProvider(),
        builtin_gate=object(),
        request_chat_create_confirm=lambda payload: {"approved": False},
        execute_agent_chat_create=lambda payload: {},
        prepare_agent_chat_create=lambda payload: payload,
        request_worktree_merge_confirm=lambda payload: {"approved": False},
        work_chain_id=bridge._db.automatic_work.create_chain(
            "conv-light", root_submission_id=uuid4().hex
        ),
    )
    kwargs.update(over)
    _run_id, outcome = bridge.run_reply(**kwargs)
    return outcome


def _native_names(gateway, call=0) -> set[str]:
    tools = gateway.tools_seen[call] or []
    return {tool["function"]["name"] for tool in tools}


def test_a_fresh_profile_plain_send_offers_exactly_the_light_set_native(tmp_path):
    """gpt-4.1-nano's shape: a native-tools provider gets ``tools=`` with the
    light set and nothing else -- no sub-agent, chat, web, Watchlists, todo,
    character or built-in MCP tool."""
    bridge, gateway, session, assistant_id = _production_bridge(tmp_path, [["hi"]])

    outcome = _send(bridge, session, assistant_id, _native_resolution())

    assert outcome.status == "done"
    names = _native_names(gateway)
    assert names == LIGHT, sorted(names)


def test_a_fresh_profile_plain_send_carries_the_minimal_prompt_fenced(tmp_path):
    """llama.cpp's shape: the system row lists only the light set in the
    fenced protocol, and carries no agent operating prompt (which tells the
    model to "Use spawn_subagent")."""
    bridge, gateway, session, assistant_id = _production_bridge(tmp_path, [["hi"]])

    outcome = _send(bridge, session, assistant_id, _test_resolution())

    assert outcome.status == "done"
    system = str(gateway.messages_seen[0][0]["content"])
    for name in LIGHT:
        assert name in system, system
    for name in AGENT_ONLY:
        assert name not in system, (name, system[:400])
    assert "spawn_subagent" not in system
    assert "find_tools" not in system
    assert "delegate" not in system.lower()
    # The one line that keeps a small model from calling a tool for "hi"
    # (live: gpt-4.1-nano called one 15/15 times without it, 0/15 with it).
    assert system.startswith(
        "You are a helpful assistant.\n\nAnswer directly; call a tool only "
        "when the request needs one."
    ), system[:200]


def test_a_hallucinated_spawn_is_refused_in_the_light_surface(tmp_path):
    """Not offering a tool is not enough: a model that names
    ``spawn_subagent`` anyway must not start a sub-agent."""
    bridge, gateway, session, assistant_id = _production_bridge(
        tmp_path,
        [[_fence("spawn_subagent", {"task": "say hi"})], ["hi"]],
    )

    outcome = _send(bridge, session, assistant_id, _test_resolution())

    spawns = [step for step in outcome.steps if step.kind == "spawn"]
    assert spawns == [], spawns
    assert outcome.subagents_spawned == 0
    children = bridge._db.list_runs("conv-light", agent_kind="subagent")
    assert children == [], children


def test_turning_on_full_agent_tools_offers_the_agent_surface_on_the_next_send(
    tmp_path,
):
    """Nothing is removed: the full agent surface comes back on the very next
    send once the user turns on Full agent tools."""
    bridge, gateway, session, assistant_id = _production_bridge(
        tmp_path, [["hi"], ["hi"]]
    )
    _send(bridge, session, assistant_id, _native_resolution())
    assert _native_names(gateway, 0) == LIGHT

    assert save_setting_to_cli_config("tools", "full_agent_tools_enabled", True)
    _send(bridge, session, assistant_id, _native_resolution())

    names = _native_names(gateway, 1)
    for expected in (
        "spawn_subagent",
        "fork_chat",
        "web_search",
        "watchlists_list_sources",
        "character_save",
        "mcp__tldw_chatbook__create_note",
    ):
        assert expected in names, (expected, sorted(names))


def test_a_tool_the_user_switched_on_is_offered_without_the_full_surface(tmp_path):
    """Setup's Tools step / the MCP Tool gates pane switch one tool on: plain
    chat offers it on the next send, still without the agent surface. A tool
    from an MCP server the user added is offered too; chatbook's own MCP
    source is not. The switch is flipped between two sends on ONE bridge
    (review round 1, F8): the next send sees it without a new bridge."""
    bridge, gateway, session, assistant_id = _production_bridge(
        tmp_path, [["hi"], ["hi"]]
    )

    def send():
        _send(
            bridge,
            session,
            assistant_id,
            _native_resolution(),
            mcp_provider=_SourcedMCPProvider(user_server=True),
        )

    send()
    assert _native_names(gateway, 0) == LIGHT | {"mcp__weather__forecast"}

    assert save_setting_to_cli_config("tools", "read_file_enabled", True)
    send()

    names = _native_names(gateway, 1)
    assert names == LIGHT | {"read_file", "mcp__weather__forecast"}, sorted(names)


def test_a_real_mcp_provider_attributes_user_servers_but_not_the_builtin_source():
    """The narrowing asks the composed MCP provider which server each tool
    comes from: a server the user added is "turned on"; chatbook's own
    built-in source is part of the full agent surface."""
    import asyncio

    from Tests.Agents.test_mcp_tool_provider import (
        FakeMCPService,
        _catalog_record,
        _compose,
        _tool_dict,
    )
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
    from tldw_chatbook.Agents.plain_chat_tool_surface import user_mcp_tool_names

    service = FakeMCPService(
        inventory={"tools": [_tool_dict("create_note", "Create a note.")]},
        catalog_records=[_catalog_record("weather", [_tool_dict("forecast")])],
    )
    provider = MCPToolProvider(service=service, main_loop=asyncio.new_event_loop())
    _compose(provider)

    assert provider.server_key_for("mcp__tldw_chatbook__create_note") == (
        BUILTIN_MCP_SERVER_KEY
    )
    assert user_mcp_tool_names(provider) == frozenset({"mcp__weather__forecast"})


class _PluginOwnedMCPProvider:
    """A plugin's MCP component as ``PluginMCPProvider`` composes it: tools
    renamed into the plugin namespace and sourced ``plugin_mcp``."""

    name = "mcp__plugin_demo__lookup"

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id=self.name,
                name=self.name,
                one_line_description="A plugin's MCP tool",
                source="plugin_mcp",
            )
        ]

    def server_key_for(self, name):
        return "local:plugin-demo" if name == self.name else None


def test_a_plugin_group_keeps_user_servers_but_not_plugin_or_builtin_tools():
    """Review round 1 (F1): with any plugin-owned skill available, the
    controller wraps the MCP provider in ``MCPProviderGroup``. The group had
    no ``server_key_for``, so the narrowing failed closed and dropped every
    tool from the servers the user added. A plugin's own MCP tools come with
    its skills, which stay off in plain chat, so they stay off too."""
    import asyncio

    from Tests.Agents.test_mcp_tool_provider import (
        FakeMCPService,
        _catalog_record,
        _compose,
        _tool_dict,
    )
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
    from tldw_chatbook.Agents.plain_chat_tool_surface import user_mcp_tool_names
    from tldw_chatbook.Plugins.mcp_provider import MCPProviderGroup

    service = FakeMCPService(
        inventory={"tools": [_tool_dict("create_note", "Create a note.")]},
        catalog_records=[_catalog_record("weather", [_tool_dict("forecast")])],
    )
    provider = MCPToolProvider(service=service, main_loop=asyncio.new_event_loop())
    _compose(provider)
    group = MCPProviderGroup([provider, _PluginOwnedMCPProvider()])

    assert group.server_key_for("mcp__weather__forecast") == "local:weather"
    assert group.server_key_for("mcp__plugin_demo__lookup") == "local:plugin-demo"
    assert group.server_key_for("missing") is None
    assert user_mcp_tool_names(group) == frozenset({"mcp__weather__forecast"})


@pytest.mark.parametrize(
    ("tool", "args", "confirm_kw"),
    [
        ("fork_chat", {"title": "Side chat"}, "request_chat_create_confirm"),
        ("new_chat", {"title": "Fresh chat"}, "request_chat_create_confirm"),
        ("install_skill", {"source": "https://example.invalid/s"}, "request_skill_install_confirm"),
        ("run_skill_script", {"skill": "x", "script": "run.py"}, "request_skill_script_confirm"),
    ],
)
def test_a_hallucinated_runtime_tool_never_reaches_its_confirm_card(
    tmp_path, tool, args, confirm_kw
):
    """Review round 1 (C-F1): a light run does not even build the chat-create,
    skill-install or script closures. For ``fork_chat`` that is the only
    barrier: with the closures built, a model naming it anyway raised a real
    fork approval card for a tool plain chat never offered."""
    asked: list[object] = []

    def confirm(*args_, **kwargs_):
        asked.append((args_, kwargs_))
        return {"approved": True} if confirm_kw != "request_skill_install_confirm" else True

    bridge, gateway, session, assistant_id = _production_bridge(
        tmp_path, [[_fence(tool, args)], ["hi"]]
    )

    outcome = _send(
        bridge, session, assistant_id, _test_resolution(), **{confirm_kw: confirm}
    )

    assert asked == [], asked
    results = [step for step in outcome.steps if step.tool_name == tool and step.result]
    assert results, [(s.kind, s.tool_name, s.result[:80]) for s in outcome.steps]
    assert all("not permitted" in step.result for step in results), [
        step.result for step in results
    ]


def test_a_skill_turn_keeps_the_agent_surface(tmp_path):
    """Review round 1 (C-F2): a message that names a skill (``$skill``) gets
    the agent surface for that turn -- skills run as sub-agents."""
    bridge, gateway, session, assistant_id = _production_bridge(tmp_path, [["hi"]])

    _send(
        bridge,
        session,
        assistant_id,
        _native_resolution(),
        turn_skill_bindings=("summarize",),
    )

    assert "spawn_subagent" in _native_names(gateway)


def test_a_light_run_starts_no_fleet_and_spawns_nothing(tmp_path):
    """Review round 1 (C-F7): the light plan's own budget allows no
    sub-agent, and a light run creates no fleet coordinator."""
    bridge, gateway, session, assistant_id = _production_bridge(tmp_path, [["hi"]])

    _send(bridge, session, assistant_id, _native_resolution())

    assert "conv-light" not in bridge._fleet_coordinators


class _LiveFleet:
    def live_count(self):
        return 1


class _Inbox:
    def __init__(self, reports):
        self._reports = reports

    def snapshot(self):
        return self._reports


def test_live_agent_work_keeps_the_agent_surface(tmp_path, monkeypatch):
    """Review round 1 (F8): turning Full agent tools off while sub-agents run
    must not leave the next turn -- the wake turn included -- unable to read
    their reports or reach their fleet. Live children, or progress reports
    nobody has read, keep the agent surface; an idle inbox does not."""
    bridge, gateway, session, assistant_id = _production_bridge(tmp_path, [["hi"]])

    assert bridge.light_surface_for((), session_id=session.id, conversation_id="c")
    bridge._fleet_coordinators["c"] = _LiveFleet()
    assert not bridge.light_surface_for((), session_id=session.id, conversation_id="c")
    del bridge._fleet_coordinators["c"]

    monkeypatch.setattr(bridge, "_session_progress_inbox", lambda _sid: _Inbox(()))
    assert bridge.light_surface_for((), session_id=session.id)
    monkeypatch.setattr(
        bridge, "_session_progress_inbox", lambda _sid: _Inbox(("report",))
    )
    assert not bridge.light_surface_for((), session_id=session.id)


def test_a_send_asks_about_its_own_conversations_agent_work(tmp_path, monkeypatch):
    """The send passes its session and conversation, so live work in THIS
    conversation keeps the agent surface on the very next send."""
    bridge, gateway, session, assistant_id = _production_bridge(tmp_path, [["hi"]])
    asked: list[tuple[object, object]] = []

    def in_flight(session_id, conversation_id):
        asked.append((session_id, conversation_id))
        return conversation_id == "conv-light"

    monkeypatch.setattr(bridge, "_agent_work_in_flight", in_flight)

    _send(bridge, session, assistant_id, _native_resolution())

    assert (session.id, "conv-light") in asked
    assert "spawn_subagent" in _native_names(gateway)


def test_the_tools_chip_counts_what_plain_chat_is_offered(tmp_path):
    """Review round 1 (B-F4): on a light-surface profile the status strip
    read "Tools: 5 ready" and the Inspector "tools 5" -- chatbook's own MCP
    tools, which the light surface does not offer -- while the request
    carried three others. The published counts are now the light surface's:
    its own tools, and only the MCP tools from servers the user added."""
    from types import SimpleNamespace

    from Tests.Chat.test_console_voice_preflight import controller_for
    from tldw_chatbook.Chat.console_display_state import ConsoleControlState

    bridge, _gateway, session, _assistant = _production_bridge(tmp_path, [])
    controller = controller_for(SimpleNamespace())
    controller._agent_bridge = bridge
    controller.app = SimpleNamespace()
    mcp = _SourcedMCPProvider(user_server=True)

    from tldw_chatbook.Agents.plain_chat_tool_surface import offered_mcp_tool_count

    controller._publish_mcp_inspector_counts(
        offered_mcp_tool_count(bridge, mcp, session.id), 0
    )

    assert controller.app.console_mcp_tool_count == 1  # the user's weather server
    assert controller.app.console_tool_count == len(LIGHT)
    chip = ConsoleControlState.from_values(
        tool_count=controller.app.console_tool_count,
        mcp_tool_count=controller.app.console_mcp_tool_count,
    )
    assert chip.tools_label == "Tools: 4 ready"

    # Full agent tools on: the whole composed MCP catalog is offered again,
    # and the built-ins go uncounted as before this task.
    assert save_setting_to_cli_config("tools", "full_agent_tools_enabled", True)
    controller._publish_mcp_inspector_counts(
        offered_mcp_tool_count(bridge, mcp, session.id), 0
    )
    assert controller.app.console_mcp_tool_count == 3
    assert controller.app.console_tool_count == 0
