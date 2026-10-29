"""What setup says about the assistant's tools, derived from the gates (TASK-34100.13).

fulltrack-02: the Tools step said "Everything is off by default" while
``all_tool_gates()`` -- the enumerator the MCP hub's Tool gates pane uses --
had three groups on by default that the step never showed. The Summary
counted only truthy ``[tools]`` keys, and the Quick-track note repeated
"tools off". Every one of those sentences now comes from this module, which
reads the same ``ToolGate`` rows as the pane.

The step still offers only the eight opt-in switches (the ``_GATEABLE_BUILTINS``
rows). The local group -- the ``[console]`` master switch and the gates it
masters -- is described, never offered: a reflexive "turn everything off" in
setup would silently remove web search and the Watchlists tools.

Pure: callers pass the ``ToolGate`` list (``all_tool_gates(config=...)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tldw_chatbook.Agents.builtin_tool_gate import (
    LOCAL_TOOLS_MASTER_KEY,
    TOOL_GATES_PANE_PATH,
    ToolGate,
)

#: Plain names for the local group's gates, in the order the line lists them.
#: Keyed by config key; a gate missing here is named by its own title.
_LOCAL_GROUP_NAMES: Mapping[str, str] = {
    LOCAL_TOOLS_MASTER_KEY: "web search and Watchlists",
    "web_deep_search_enabled": "deep web research",
    "character_tools_enabled": "character cards",
    "ask_user_enabled": "questions for you",
}

#: The built-in file tools (`file_operation_tools`), which the Workspace hint
#: sits under.
FILE_TOOL_NAMES = frozenset(
    {"read_file", "list_directory", "write_file", "glob_files", "grep_files"}
)

#: Where file tools can reach, and where a folder is added.
WORKSPACE_HINT = (
    "File tools only reach folders you add as a Workspace (Settings ▸ "
    "Workspaces) and the chat's scratch area."
)

#: The agent runtime's own tools are not registration gates, so
#: ``all_tool_gates()`` cannot describe them. TASK-34100.5 measured what plain
#: chat is offered; the owner's "light surface by default" ruling
#: (TASK-34100.5.3) changes it, and that task changes this one sentence.
#: Measured live (TASK-34100.13, gpt-4.1-mini, every switch off): the request
#: also carried sub-agent, skill, to-do, canvas, run-log, chat-fork,
#: calculator and date tools.
AGENT_RUNTIME_TOOLS_NOTE = (
    "chatbook's own agent tools (sub-agents, skills, to-dos, canvas, run log, "
    "calculator) are offered when the model's context window has room"
)
#: Review V3: the to-do tools live in the local tool provider, which the
#: ``[console]`` master switch drops with the rest of the local group.
AGENT_RUNTIME_TOOLS_NOTE_LOCAL_OFF = AGENT_RUNTIME_TOOLS_NOTE.replace("to-dos, ", "")


@dataclass(frozen=True)
class ToolsPosture:
    """The assistant's registration-gate posture, as setup describes it.

    Attributes:
        switch_total: How many opt-in switches the step offers.
        switched_on: Titles of the opt-in switches that are on.
        also_available: Plain names of the enabled local-group gates.
        local_tools_on: Whether the ``[console]`` master switch is on.
    """

    switch_total: int
    switched_on: tuple[str, ...]
    also_available: tuple[str, ...]
    local_tools_on: bool = True


def tools_posture(gates: Sequence[ToolGate]) -> ToolsPosture:
    """Fold ``all_tool_gates()`` rows into what setup says.

    Args:
        gates: Every gate, as ``all_tool_gates()`` returns them.

    Returns:
        The posture.
    """
    builtin = [gate for gate in gates if gate.group == "builtin"]
    local = {gate.key: gate for gate in gates if gate.group == "local"}
    master = local.get(LOCAL_TOOLS_MASTER_KEY)
    also: list[str] = []
    # Review F3: with the master off the Console composes no local tool
    # provider at all -- ask_user and the character tools go with web search
    # and Watchlists -- so the whole group is off whatever its own gates say.
    local_tools_on = master is None or master.enabled
    if not local_tools_on:
        local = {}
    for key, name in _LOCAL_GROUP_NAMES.items():
        gate = local.get(key)
        if gate is None or not gate.enabled:
            continue
        also.append(name)
    for key, gate in local.items():
        if key not in _LOCAL_GROUP_NAMES and gate.enabled:
            also.append(gate.title)
    return ToolsPosture(
        switch_total=len(builtin),
        switched_on=tuple(gate.title for gate in builtin if gate.enabled),
        also_available=tuple(also),
        local_tools_on=local_tools_on,
    )


def _join(names: Sequence[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def posture_sentence(posture: ToolsPosture) -> str:
    """The step's subtitle: what the switches below do, and their state."""
    on = len(posture.switched_on)
    total = posture.switch_total
    if on == 0:
        return (
            f"These {total} switches add file, note and document tools. All {total} are "
            "off until you turn them on, and each one asks you before it runs."
        )
    return (
        f"{on} of {total} file, note and document switches are on — change any below. "
        "Each one asks you before it runs."
    )


def also_available_line(posture: ToolsPosture) -> str:
    """The read-only line naming the default-on groups the step does not offer."""
    if not posture.also_available:
        # Live (TASK-34100.13 resume): the app's own MCP server still offers
        # list_characters and chat_with_character here (no native twin;
        # TASK-34100.5 decides whether that server belongs in plain chat), so
        # the line names only what goes: every tool that edits a card.
        return (
            "Web search, Watchlists, questions for you and the tools that edit "
            "character cards are switched off. Turn them on under "
            f"{TOOL_GATES_PANE_PATH}."
        )
    # Live (TASK-34100.13): a model whose context window is unknown -- a
    # fresh local server -- gets a plain request with no tools at all
    # (TASK-34100.5), so the line says when these are offered.
    return (
        f"Also available (asks first) when the model's context window has room: "
        f"{_join(posture.also_available)}."
    )


def tools_summary_detail(posture: ToolsPosture) -> str:
    """The Summary's Tools row detail, from the same posture as the step."""
    on = len(posture.switched_on)
    if on:
        switches = f"{on} switched on ({', '.join(posture.switched_on)})"
    else:
        switches = "file, note and document switches off"
    parts = [switches]
    if posture.also_available:
        parts.append(
            "also available, asks first, when the context window has room: "
            f"{_join(posture.also_available)}"
        )
    parts.append(
        AGENT_RUNTIME_TOOLS_NOTE
        if posture.local_tools_on
        else AGENT_RUNTIME_TOOLS_NOTE_LOCAL_OFF
    )
    return " · ".join(parts) + f". Change them under {TOOL_GATES_PANE_PATH}"


def quick_defaults_note(
    posture: ToolsPosture, *, auto_retrieve: bool, default_theme: bool = True
) -> str:
    """The Quick track's "left at defaults" note, derived instead of a literal.

    Review R6: it names the theme only while it is the default one, and says
    nothing about notes folder sync, whose roots live in the Library's
    database where config cannot see them (the Summary has its own row).
    """
    tools = "file, note and document switches off"
    if posture.switched_on:
        tools = f"{len(posture.switched_on)} file, note and document switches on"
    if posture.also_available:
        tools += (
            f" ({_join(posture.also_available)} available when the context "
            "window has room, asking first)"
        )
    search = (
        "Library searched on every message"
        if auto_retrieve
        else "Library searched only when you ask"
    )
    theme = "; default theme" if default_theme else ""
    return (
        f"Left at recommended defaults: {tools}; {search}{theme}. Each one can be "
        "changed later."
    )


def gate_config_view(wizard: Any) -> dict[str, dict[str, Any]]:
    """The ``[tools]``/``[console]`` sections the step reads gates from.

    Args:
        wizard: The Tools step's wizard.

    Returns:
        ``{"tools": {...}, "console": {...}}`` for ``all_tool_gates(config=...)``,
        read from the config as it is now (``wizard_config_sections``).
    """
    from tldw_chatbook.UI.Wizards.first_run_setup_state import wizard_config_sections

    return wizard_config_sections(wizard, "tools", "console")
