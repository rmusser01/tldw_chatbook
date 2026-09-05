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

register(
    PromptSpec(
        id="agents.ask_user_tool_description",
        subsystem="agents",
        title="ask_user tool description",
        description=(
            "The description the ask_user tool presents to the model; most of "
            "its words say when NOT to ask."
        ),
        used_in="Agents/local_tool_provider.py (_default_specs, via ASK_USER_DESCRIPTION)",
        default=(
            "Ask the user up to 4 multiple-choice questions and wait for the "
            "answers. Use it ONLY for a decision that is genuinely the user's to "
            "make: a preference, a trade-off between valid designs, or something "
            "neither the code nor the conversation can tell you. Do not ask when a "
            "conventional default exists, when the answer is discoverable by "
            "reading the code or running a tool, when you can proceed and state "
            "your assumption, or to confirm a plan you already have. Batch related "
            "questions into ONE call instead of asking several times. Each question "
            "offers 2-4 options; the user can always type a free-text 'Other' "
            "answer instead. The result lists the selected labels per question; "
            "'unanswered' marks questions the user skipped, and 'answered': false "
            "with a reason means no answer will come. If the reason is 'busy', "
            "another question is already waiting for the user: proceed without "
            "asking again this turn."
        ),
        contract_note=(
            "task-31420: the [tools] ask_user_enabled gate defaults ON, so every "
            "user gets whatever this text says -- it is the restraint guidance "
            "(PRD A13). No placeholders. Keep the 'busy' sentence: the tool's "
            "busy result tells the model not to retry, and this is where it "
            "learns what that means."
        ),
    )
)
