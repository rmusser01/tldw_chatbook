# tldw_chatbook/Internal_Prompts/agents_prompts.py
"""Agent-runtime prompt specs. Defaults moved verbatim from
Agents/agent_service.py (SUBAGENT_SYSTEM_PROMPT), Chat/console_agent_bridge.py
(CONSOLE_AGENT_OPERATING_PROMPT), and the static scaffold of
Agents/agent_runtime.py's render_tool_protocol (converted from an f-string
to a {tool_list}/{fence_open}/{fence_close} template). Parity tests compare
the registry defaults against the live constants and the rendered template
against the original scaffold's output."""

from .catalog import PromptSpec, register

register(
    PromptSpec(
        id="agents.subagent_system",
        subsystem="agents",
        title="Sub-agent system prompt",
        description="System prompt given to a spawned sub-agent run.",
        used_in="Agents/agent_service.py (SUBAGENT_SYSTEM_PROMPT)",
        default=(
            "You are a focused sub-agent. Complete the task you are given and "
            "reply with a concise result. You cannot ask the user questions."
        ),
        contract_note=(
            "The leading text is an identity contract: console_agent_bridge "
            "detects sub-agent turns by prefix-matching this prompt. "
            "Rewording the opening changes detection; the runtime also "
            "matches the shipped default as a fallback."
        ),
    )
)

register(
    PromptSpec(
        id="agents.console_agent_operating",
        subsystem="agents",
        title="Console agent operating prompt",
        description="Operating instructions appended to the Console agent's system prompt.",
        used_in="Chat/console_agent_bridge.py (CONSOLE_AGENT_OPERATING_PROMPT)",
        default=(
            "You are a capable assistant with optional tools. Answer directly when no "
            "tool is needed. When a tool would help, call exactly one tool per reply "
            "using the fenced protocol described below, then continue once you have the "
            "result. Use spawn_subagent to delegate a self-contained sub-task to an "
            "isolated helper. Keep replies concise."
        ),
        contract_note=(
            "References the fenced tool protocol and spawn_subagent; keep "
            "consistent with agents.tool_protocol."
        ),
    )
)

register(
    PromptSpec(
        id="agents.tool_protocol",
        subsystem="agents",
        title="Tool-call fence protocol",
        description="Instructs the model how to call tools via the fenced text protocol.",
        used_in="Agents/agent_runtime.py (render_tool_protocol)",
        default=(
            "You can call tools. Available tools:\n"
            "{tool_list}\n\n"
            "To call a tool, your reply MUST START with the fence as its first "
            "content — no prose before it:\n"
            '{fence_open}\n{"name": "<tool name>", "arguments": {...}}\n'
            "{fence_close}\n"
            "One tool call per reply. After you receive the tool result, either "
            "call another tool the same way or answer the user directly. If no "
            "tool is needed, just answer directly."
        ),
        required_placeholders=("tool_list", "fence_open", "fence_close"),
        contract_note=(
            "Fence markers are injected by code from agent_runtime."
            "FENCE_OPEN/_FENCE_CLOSE and are parsed by the tool-call parser "
            "— the {fence_open}/{fence_close}/{tool_list} tokens are "
            "required. The empty-tools case renders no protocol at all "
            "(code-side)."
        ),
    )
)
