# Tests/Internal_Prompts/test_agents_prompt_parity.py
"""Registry defaults must match the agent-runtime literals byte-for-byte;
the tool-protocol template must render exactly what render_tool_protocol's
static scaffold produced pre-migration."""

from tldw_chatbook.Internal_Prompts import CATALOG, render_internal_prompt


def test_subagent_system_matches_source_constant():
    from tldw_chatbook.Agents.agent_service import SUBAGENT_SYSTEM_PROMPT

    assert CATALOG["agents.subagent_system"].default == SUBAGENT_SYSTEM_PROMPT


def test_console_agent_operating_matches_source_constant():
    from tldw_chatbook.Chat.console_agent_bridge import (
        CONSOLE_AGENT_OPERATING_PROMPT,
    )

    assert (
        CATALOG["agents.console_agent_operating"].default
        == CONSOLE_AGENT_OPERATING_PROMPT
    )


def test_tool_protocol_template_renders_original_scaffold():
    # Pre-migration expected output, built exactly as agent_runtime.py:183-193
    # does today (copy the f-string expression verbatim with sample values).
    from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN, _FENCE_CLOSE

    tool_list = '{\n  "name": "demo",\n  "description": "d",\n  "parameters": {}\n}'
    expected = (
        "You can call tools. Available tools:\n"
        f"{tool_list}\n\n"
        "To call a tool, your reply MUST START with the fence as its first "
        "content — no prose before it:\n"
        f'{FENCE_OPEN}\n{{"name": "<tool name>", "arguments": {{...}}}}\n'
        f"{_FENCE_CLOSE}\n"
        "One tool call per reply. After you receive the tool result, either "
        "call another tool the same way or answer the user directly. If no "
        "tool is needed, just answer directly."
    )
    assert render_internal_prompt(
        "agents.tool_protocol",
        tool_list=tool_list,
        fence_open=FENCE_OPEN,
        fence_close=_FENCE_CLOSE,
    ) == expected
