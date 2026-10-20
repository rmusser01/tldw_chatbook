"""Plain Console chat's light tool surface (TASK-34100.5.3).

Owner ruling, 2026-10-04 ("Light surface by default"): plain Console chat
keeps a small, safe tool set and offers no sub-agents, skills, web or other
autonomous or costly tools until the user turns them on. Setup and its
Summary say exactly what is on, and small local models get the minimal
prompt.

Before this module, a fresh profile with every tool gate off still handed
plain chat the whole agent surface: gpt-4.1-nano listed sub-agent, skill,
run-log, chat, web, Watchlists, todo and character tools, and answered
"Reply with just: hi" by calling ``spawn_subagent`` (TASK-34100.5 evidence
g5-v-oai 03, g5-r2b-oaihome 04).

This module is the ONE place that says what the light surface is:

* ``LIGHT_SURFACE_TOOL_NAMES`` -- the safe set itself.
* ``FULL_AGENT_TOOLS_GATE_KEY`` -- the ``[tools]`` switch that restores the
  full agent surface (off by default).
* ``light_surface_allowed_tools`` -- the narrowing the first-request planner
  applies (``Chat/console_agent_bridge.build_console_first_request_plan``).
* ``summarize_tool_surface`` -- what setup's Tools step and Summary say.

Turning a tool on never needs the full surface: a built-in whose own gate the
user switched on (setup's Tools step, MCP ▸ Servers ▸ built-in row ▸ Tool
gates) is offered in plain chat too, and so are Library tools once a session
allows assistant Library access, tools from MCP servers the user added, and a
raw shell the user armed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping, Sequence
from typing import Any

#: THE light surface: the tools plain Console chat offers before the user
#: turns anything on. The narrowest set that satisfies the ruling's examples
#: (date/time, calculator, ask-user). ``ask_user`` still follows its own
#: ``[tools] ask_user_enabled`` gate, so switching that off removes it here.
LIGHT_SURFACE_TOOL_NAMES: tuple[str, ...] = (
    "get_current_datetime",
    "calculator",
    "ask_user",
)

#: Plain-language names for the light set, in ``LIGHT_SURFACE_TOOL_NAMES``
#: order -- the words setup's Summary and the User Guide use.
LIGHT_SURFACE_TOOL_TITLES: Mapping[str, str] = {
    "get_current_datetime": "date and time",
    "calculator": "calculator",
    "ask_user": "ask you a question",
}

#: Local tools with their OWN default-off ``[tools]`` gate. Switching one on
#: is turning that tool on, so plain chat offers it without the full surface.
#: ``web_deep_search`` is the only one (the rest of the local group sits
#: under the ``[console] local_tools_enabled`` master, which defaults on).
LIGHT_SURFACE_LOCAL_OPT_INS: tuple[str, ...] = ("web_deep_search",)

#: The catalog ``source`` of a plugin's own MCP tools
#: (``Plugins/mcp_provider.PluginMCPProvider``). They come with the plugin's
#: skills, so plain chat leaves them off with the skills.
PLUGIN_MCP_SOURCE = "plugin_mcp"

#: The ``[tools]`` switch for the full agent surface. Off by default.
FULL_AGENT_TOOLS_GATE_KEY = "full_agent_tools_enabled"
FULL_AGENT_TOOLS_DEFAULT_ENABLED = False
FULL_AGENT_TOOLS_TITLE = "Full agent tools"
#: One line for setup's Tools step, whose rows fit one line of description
#: (live 120x40: the full description was clipped after three lines).
FULL_AGENT_TOOLS_BLURB = (
    "Adds sub-agents, skills, web, Watchlists, workspace files, characters, "
    "Canvas and chatbook's own MCP tools."
)
FULL_AGENT_TOOLS_DESCRIPTION = (
    "Lets the Console assistant use sub-agents, skills, web search and fetch, "
    "Watchlists, workspace files, character cards, Canvas, chat forks and "
    "chatbook's own MCP tools. Off by default: plain chat keeps date and time, "
    "calculator and ask-you-a-question, plus any tool you switch on one by "
    "one. Applies on the next send."
)


def full_agent_tools_enabled(tools_section: Mapping[str, Any] | None = None) -> bool:
    """Whether the user turned on the full agent surface.

    Args:
        tools_section: A ``[tools]`` mapping to read, or ``None`` to read the
            live config (fresh on every call, so a switch flipped in the MCP
            hub or setup applies on the next send).

    Returns:
        The coerced ``[tools] full_agent_tools_enabled`` value (default off).
    """
    from tldw_chatbook.config import coerce_bool_setting

    if tools_section is None:
        from tldw_chatbook.config import get_cli_setting

        raw = get_cli_setting(
            "tools", FULL_AGENT_TOOLS_GATE_KEY, FULL_AGENT_TOOLS_DEFAULT_ENABLED
        )
    else:
        raw = tools_section.get(
            FULL_AGENT_TOOLS_GATE_KEY, FULL_AGENT_TOOLS_DEFAULT_ENABLED
        )
    return coerce_bool_setting(raw, FULL_AGENT_TOOLS_DEFAULT_ENABLED)


def light_surface_configured() -> bool:
    """The production reader: plain chat is light unless the full surface is on.

    ``ConsoleRuntime`` hands this to the Console agent bridge, which asks it
    once per send. A read failure keeps the light surface (fail closed: never
    offer autonomous tools because the config could not be read).
    """
    try:
        return not full_agent_tools_enabled()
    except Exception:  # noqa: BLE001 -- an unreadable config stays light
        return True


def light_surface_prompts(session_system_prompt: str | None) -> tuple[str, str]:
    """The light surface's ``(plain, direct)`` system prompts for one send.

    The plain prompt is the session's own, or the neutral one-liner, and goes
    out when no tool does. The direct prompt adds the "answer directly" line
    and goes out with the light tools: live, gpt-4.1-nano called a tool for
    "Reply with just: hi" without it. The send's planner and the meter's
    re-plan both build them here, so they size the same request.
    """
    from tldw_chatbook.Agents.first_request_window import PLAIN_CHAT_SYSTEM_PROMPT
    from tldw_chatbook.Internal_Prompts import get_internal_prompt

    plain = (session_system_prompt or "").strip() or PLAIN_CHAT_SYSTEM_PROMPT
    hint = get_internal_prompt("agents.console_light_surface_hint")
    return plain, f"{plain}\n\n{hint}"


def light_surface_budget(budget: Any) -> Any:
    """Return ``budget`` with sub-agents switched off.

    Zero ``max_subagents`` keeps ``spawn_subagent`` (and the fleet tools that
    share its predicate) out of the request AND refuses a hallucinated call:
    ``AgentService.spawn`` checks the same budget.
    """
    return dataclasses.replace(budget, max_subagents=0)


def light_surface_allowed_tools(
    allowed_tools: Sequence[str],
    *,
    builtin_names: Collection[str],
    local_names: Collection[str],
    user_source_names: Collection[str],
) -> tuple[str, ...]:
    """Narrow a run's allow-list to the light surface, keeping its order.

    Args:
        allowed_tools: The run's composed allow-list.
        builtin_names: Every name ``BuiltinToolProvider`` listed. It lists the
            always-on pair plus only those gated built-ins whose gate the user
            switched on, so all of them stay.
        local_names: The names the local workspace/web/Watchlists provider
            listed. Only ``ask_user`` and the ``LIGHT_SURFACE_LOCAL_OPT_INS``
            stay.
        user_source_names: Names from sources the user turned on themselves:
            Library tools (only composed when the session allows assistant
            Library access), tools from MCP servers the user added, and an
            armed raw shell.

    Returns:
        The allow-list with everything else -- skills, Canvas, profile tools,
        the rest of the local group, the built-in MCP source and
        ``spawn_subagent`` -- removed.
    """
    local = frozenset(local_names)
    keep = (
        frozenset(LIGHT_SURFACE_TOOL_NAMES)
        | frozenset(builtin_names)
        | (local & frozenset(LIGHT_SURFACE_LOCAL_OPT_INS))
        | frozenset(user_source_names)
    )
    return tuple(name for name in allowed_tools if name in keep)


def user_mcp_tool_names(mcp_provider: Any | None) -> frozenset[str]:
    """Names of MCP tools that come from servers the user added.

    The app's own built-in ``tldw_chatbook`` source is part of the full agent
    surface, not something the user turned on, so its tools are left out. So
    are a plugin's own MCP tools (``source == "plugin_mcp"``): they come with
    the plugin's skills, and skills stay off in plain chat. A provider that
    cannot say which server a tool belongs to contributes nothing (fail
    closed); ``MCPProviderGroup`` answers for each of its owners.
    """
    if mcp_provider is None:
        return frozenset()
    server_key_for = getattr(mcp_provider, "server_key_for", None)
    if not callable(server_key_for):
        return frozenset()
    from tldw_chatbook.MCP.builtin_tool_policy import BUILTIN_MCP_SERVER_KEY

    names: set[str] = set()
    try:
        entries = mcp_provider.list_catalog()
    except Exception:  # noqa: BLE001 -- an unreadable catalog offers nothing
        return frozenset()
    for entry in entries:
        if entry.source == PLUGIN_MCP_SOURCE:
            continue
        server_key = server_key_for(entry.name)
        if server_key and server_key != BUILTIN_MCP_SERVER_KEY:
            names.add(entry.name)
    return frozenset(names)


def tool_kill_switch_engaged(app: Any) -> bool:
    """Whether the MCP hub's tool kill switch is on, read now.

    With it on, the controller composes no local tools provider (nor MCP or
    shell), so ``ask_user`` and ``web_deep_search`` are not sent. An app
    without the MCP service has nothing to honor (the Console's own
    kill-switch reader returns ``None`` then); a read that raises counts as
    on, because the controller then composes no local tools either.
    """
    service = getattr(app, "unified_mcp_service", None)
    getter = getattr(service, "get_kill_switch", None)
    if not callable(getter):
        return False
    try:
        return bool(getter())
    except Exception:  # noqa: BLE001 -- the compose path fails closed too
        return True


def _bridge_local_tools_available(bridge: Any | None) -> bool:
    check = getattr(bridge, "light_surface_local_tools_available", None)
    if not callable(check):
        return True
    try:
        return bool(check())
    except Exception:  # noqa: BLE001 -- count as composed, as before
        return True


def _bridge_is_light(bridge: Any | None, session_id: str | None) -> bool:
    check = getattr(bridge, "light_surface_for", None)
    if not callable(check):
        return False
    try:
        return bool(check((), session_id=session_id))
    except Exception:  # noqa: BLE001 -- an unreadable bridge counts as before
        return False


def offered_mcp_tool_count(
    bridge: Any | None, mcp_provider: Any | None, session_id: str | None = None
) -> int:
    """How many of a composed MCP catalog's tools the next send is offered.

    Review round 1 (B-F4): the status strip read "Tools: 5 ready" on a
    light-surface profile -- chatbook's own MCP tools, which plain chat is
    not offered. With the light surface only the user's own servers count.
    """
    if not _bridge_is_light(bridge, session_id):
        return len(provider_names(mcp_provider))
    return len(user_mcp_tool_names(mcp_provider))


def offered_builtin_tool_count(bridge: Any | None, session_id: str | None = None) -> int:
    """The light surface's own (non-MCP) tools, for the Tools chip.

    The chip only ever counted MCP tools (task-1234 F7), so the full surface
    keeps 0 here; the light surface names its own set.
    """
    if not _bridge_is_light(bridge, session_id):
        return 0
    try:
        return len(
            light_surface_tool_schemas(
                local_tools_available=_bridge_local_tools_available(bridge)
            )
        )
    except Exception:  # noqa: BLE001 -- a chip never breaks a send
        return 0


def provider_names(provider: Any | None) -> tuple[str, ...]:
    """The catalog names one composed provider lists (``()`` for none)."""
    if provider is None:
        return ()
    try:
        return tuple(entry.name for entry in provider.list_catalog())
    except Exception:  # noqa: BLE001 -- an unreadable provider offers nothing
        return ()


# -- what the context meter counts ------------------------------------------


def light_surface_local_names() -> tuple[str, ...]:
    """Local-provider tools the light surface can carry, in one list.

    The safe set's local member(s) -- every name that is not a built-in --
    plus ``LIGHT_SURFACE_LOCAL_OPT_INS``; the local provider says how it
    describes each one without being composed, and which gate each follows,
    and refuses a name it cannot (review round 2, R2-3).
    """
    from tldw_chatbook.Agents.tool_catalog import ALWAYS_ON_BUILTIN_NAMES, _GATEABLE_BUILTINS

    builtin = {*ALWAYS_ON_BUILTIN_NAMES, *(entry.tool_name for entry in _GATEABLE_BUILTINS)}
    names = dict.fromkeys((*LIGHT_SURFACE_TOOL_NAMES, *LIGHT_SURFACE_LOCAL_OPT_INS))
    return tuple(name for name in names if name not in builtin)


def light_surface_tool_schemas(
    user_source_schemas: Sequence[Any] = (),
    *,
    local_tools_available: bool = True,
) -> tuple[Any, ...]:
    """The light surface's tool schemas, read fresh from config.

    The candidates are what a send's providers would list -- the always-on
    built-ins plus every gated built-in the user switched on
    (``BuiltinToolProvider`` reads the same gates), the local tools of
    ``light_surface_local_names`` whose gates are on, and
    ``user_source_schemas`` -- and they pass through the SAME
    ``light_surface_allowed_tools`` narrowing the send applies (review round
    1, F7: the meter used to rebuild the set on its own).

    Args:
        user_source_schemas: Tools from the user's own sources (MCP servers
            they added, Library tools, an armed shell) that the session's
            last send carried; they are per-session, so config cannot say.
        local_tools_available: ``False`` when the send composes no local
            tools provider whatever the config says -- the MCP hub's kill
            switch (``tool_kill_switch_engaged``) -- so no local tool counts.
    """
    from tldw_chatbook.Agents.local_tool_provider import light_surface_local_schemas
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

    provider = BuiltinToolProvider()
    builtin = [provider.load_schema(entry.id) for entry in provider.list_catalog()]
    local = (
        list(light_surface_local_schemas(light_surface_local_names()))
        if local_tools_available
        else []
    )
    user = list(user_source_schemas)
    candidates = [*builtin, *local, *user]
    keep = frozenset(
        light_surface_allowed_tools(
            [schema.name for schema in candidates],
            builtin_names=[schema.name for schema in builtin],
            local_names=[schema.name for schema in local],
            user_source_names=[schema.name for schema in user],
        )
    )
    return tuple(schema for schema in candidates if schema.name in keep)


# -- what setup says ---------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ToolSurfaceSummary:
    """What plain Console chat is offered, read from persisted config.

    Attributes:
        full_agent: Whether the full agent surface is on.
        light_names: Plain-language names of the light-set tools that are on.
        switched_on: Titles of the individual tools the user switched on.
        local_group_on: Whether the workspace, web and Watchlists group is on
            (it only matters with the full surface).
        character_tools_on: Whether the character tools are on (same).
        web_research_on: Whether a local web opt-in (Deep web research) is
            on, so plain chat has a web tool.
    """

    full_agent: bool
    light_names: tuple[str, ...]
    switched_on: tuple[str, ...]
    local_group_on: bool
    character_tools_on: bool
    web_research_on: bool = False

    @property
    def user_turned_on(self) -> bool:
        """Whether anything beyond the defaults is on."""
        return self.full_agent or bool(self.switched_on)

    @property
    def tools_on(self) -> tuple[str, ...]:
        """Every tool name that is on, light set first."""
        return self.light_names + self.switched_on


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    return value if isinstance(value, Mapping) else {}


def summarize_tool_surface(app_config: Mapping[str, Any]) -> ToolSurfaceSummary:
    """Read what plain Console chat is offered from a persisted config mapping.

    Pure: reads only ``app_config`` (setup's Summary renders from the
    persisted, re-loaded config, never from step memory).
    """
    from tldw_chatbook.Agents.builtin_tool_gate import (
        LOCAL_TOOLS_DEFAULT_ENABLED,
        local_opt_in_gates,
    )
    from tldw_chatbook.Agents.local_tool_provider import (
        CHARACTER_TOOLS_DEFAULT_ENABLED,
        CHARACTER_TOOLS_GATE_KEY,
        light_surface_local_gate,
    )
    from tldw_chatbook.Agents.tool_catalog import ALWAYS_ON_BUILTIN_NAMES, _GATEABLE_BUILTINS
    from tldw_chatbook.config import coerce_bool_setting

    tools = _section(app_config, "tools")
    console = _section(app_config, "console")

    def flag(section: Mapping[str, Any], key: str, default: bool) -> bool:
        return coerce_bool_setting(section.get(key, default), default)

    local_on = flag(console, "local_tools_enabled", LOCAL_TOOLS_DEFAULT_ENABLED)
    builtin_gates = {entry.tool_name: entry.gate_key for entry in _GATEABLE_BUILTINS}

    def member_on(name: str) -> bool:
        # Each member follows the gate the send obeys (review round 2,
        # R2-3): an always-on built-in, a gated built-in's own switch, or a
        # local tool's switch behind the local master. A name none of these
        # knows raises rather than being named as on.
        if name in ALWAYS_ON_BUILTIN_NAMES:
            return True
        if name in builtin_gates:
            return flag(tools, builtin_gates[name], False)
        gate_key, default = light_surface_local_gate(name)
        return local_on and flag(tools, gate_key, default)

    light_names = tuple(
        LIGHT_SURFACE_TOOL_TITLES[name] for name in LIGHT_SURFACE_TOOL_NAMES if member_on(name)
    )
    switched_on = [
        entry.title
        for entry in _GATEABLE_BUILTINS
        if entry.tool_name not in LIGHT_SURFACE_TOOL_NAMES and flag(tools, entry.gate_key, False)
    ]
    # The local opt-ins come from the one list, named by their own gate rows
    # ("Deep web research (web_deep_search)" -> "Deep web research"), read
    # from the config-free gate table so this stays pure. An opt-in with no
    # gate row raises (R2-3) instead of being sent unnamed.
    gates = local_opt_in_gates()
    opt_ins_on = [
        name
        for name in LIGHT_SURFACE_LOCAL_OPT_INS
        if flag(tools, gates[name][0], False) and local_on
    ]
    switched_on.extend(
        gates[name][1].split(" (", 1)[0]
        for name in opt_ins_on
        if name not in LIGHT_SURFACE_TOOL_NAMES
    )
    return ToolSurfaceSummary(
        full_agent=full_agent_tools_enabled(tools),
        light_names=light_names,
        switched_on=tuple(switched_on),
        local_group_on=local_on,
        character_tools_on=flag(
            tools, CHARACTER_TOOLS_GATE_KEY, CHARACTER_TOOLS_DEFAULT_ENABLED
        ),
        web_research_on="web_deep_search" in opt_ins_on,
    )


def _join(names: Sequence[str]) -> str:
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def full_agent_tool_groups(summary: ToolSurfaceSummary) -> tuple[str, ...]:
    """The tool groups the full agent surface adds, as plain words."""
    groups = ["sub-agents", "skills", "chat forks", "Canvas", "chatbook's own MCP tools"]
    if summary.local_group_on:
        groups[2:2] = ["web", "Watchlists", "workspace files"]
    if summary.character_tools_on:
        groups.insert(-1, "character cards")
    return tuple(groups)


#: Review round 1 (F4): the first-request planner sends no tools when even
#: the light set does not fit the window -- a self-hosted server whose window
#: the app has not read is planned as 4,096 tokens -- so every place that
#: names the light set says so, plainly.
SMALL_WINDOW_NOTE = (
    "A model with a small or not-yet-known context window gets no tools."
)


def _stays_off(summary: ToolSurfaceSummary) -> str:
    """What the full agent surface would add, as plain chat's "stay off" list.

    With Deep web research on, plain chat does have a web tool, so the list
    names the web tools that stay off instead of "web" (review round 1, F4).
    """
    web = "web search and fetch" if summary.web_research_on else "web"
    return f"Sub-agents, skills, {web}, Watchlists, workspace files and characters"


def tool_surface_detail(summary: ToolSurfaceSummary) -> str:
    """The Summary's Tools row: exactly which tools plain chat is offered."""
    from tldw_chatbook.Agents.builtin_tool_gate import TOOL_GATES_PANE_PATH

    if summary.full_agent:
        return (
            f"full agent tools ({_join(full_agent_tool_groups(summary))}), "
            f"plus {_join(summary.tools_on)}. Change them under "
            f"{TOOL_GATES_PANE_PATH}. A model with a small or not-yet-known "
            "context window gets fewer tools, or none."
        )
    return (
        f"on: {_join(summary.tools_on)}, plus tools from MCP servers you add "
        f"and Library tools in chats you allow. {_stays_off(summary)} stay off "
        f"until you turn on {FULL_AGENT_TOOLS_TITLE} under "
        f"{TOOL_GATES_PANE_PATH}. {SMALL_WINDOW_NOTE}"
    )


def tools_step_posture(summary: ToolSurfaceSummary) -> str:
    """Setup's Tools step intro: what chat has before anything is switched on."""
    small_window = SMALL_WINDOW_NOTE[0].lower() + SMALL_WINDOW_NOTE[1:]
    return (
        f"Chat starts with {_join(summary.light_names)}; {small_window} "
        "Everything below is off until you switch it on; tools that read or "
        "change your files ask before they run."
    )


def light_surface_note(summary: ToolSurfaceSummary) -> str:
    """One line naming the light surface, for the MCP hub's Tool gates pane.

    Derived from the pane's own switches (review round 1, F4/B-F3): it names
    only the light tools that are on, and it does not promise "any tool
    switched on below" -- the workspace/web/Watchlists master and Character
    cards read "on" by default yet add nothing to plain chat.
    """
    return (
        f"Plain chat: {_join(summary.light_names)}, plus any Agent built-in or "
        f"Deep web research switched on. {FULL_AGENT_TOOLS_TITLE} adds the "
        "rest: sub-agents, skills, web, Watchlists, files and characters."
    )


def tool_surface_phrase(summary: ToolSurfaceSummary) -> str:
    """A short phrase for the Quick-track note ("light tools only (...)")."""
    if summary.full_agent:
        return "full agent tools on"
    if summary.switched_on:
        return _join(summary.tools_on)
    return f"light tools only ({_join(summary.light_names)})"
