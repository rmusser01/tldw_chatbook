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


_GATES_TOUCHED = ("full_agent_tools_enabled", "read_file_enabled")


@pytest.fixture(autouse=True)
def _fresh_tool_gates():
    """Every test starts on a fresh profile's gates and leaves them that way
    (the isolated config file outlives one test)."""
    for key in _GATES_TOUCHED:
        save_setting_to_cli_config("tools", key, False)
    yield
    for key in _GATES_TOUCHED:
        save_setting_to_cli_config("tools", key, False)


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
    source is not."""
    assert save_setting_to_cli_config("tools", "read_file_enabled", True)
    bridge, gateway, session, assistant_id = _production_bridge(tmp_path, [["hi"]])

    _send(
        bridge,
        session,
        assistant_id,
        _native_resolution(),
        mcp_provider=_SourcedMCPProvider(user_server=True),
    )

    names = _native_names(gateway)
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
