"""Pure agent control loop + fence-first text tool protocol.

No Textual, app, DB, or I/O imports. Task 2 adds the protocol; Task 3
appends the loop below it.
"""
from __future__ import annotations

import json

from .agent_models import ToolCall, ToolSchema

FENCE_OPEN = "```tool_call"
_FENCE_CLOSE = "```"

STREAM_TOOL_CALL = "tool_call"
STREAM_TEXT = "text"
STREAM_UNDECIDED = "undecided"


def parse_fenced_tool_call(text: str) -> ToolCall | None:
    """Parse a response whose FIRST non-whitespace content is a tool fence.

    Returns None for anything malformed — never raises.
    """
    stripped = text.lstrip()
    if not stripped.startswith(FENCE_OPEN):
        return None
    after = stripped[len(FENCE_OPEN):]
    newline = after.find("\n")
    if newline == -1:
        return None
    # The character right after FENCE_OPEN must end the tag line (only
    # whitespace may follow before the newline). Otherwise this is a
    # look-alike tag such as ```tool_calls or ```tool_call_schema, not a
    # real fence, and must not be parsed as a tool call.
    tag_line_rest = after[:newline]
    if tag_line_rest.strip():
        return None
    body_and_rest = after[newline + 1:]
    close = body_and_rest.find(_FENCE_CLOSE)
    if close == -1:
        return None
    raw = body_and_rest[:close].strip()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    args = payload.get("arguments", {})
    if not isinstance(name, str) or not name or not isinstance(args, dict):
        return None
    return ToolCall(name=name, args=args)


def split_visible_text_and_tool_call(text: str) -> tuple[str, ToolCall | None]:
    """Handle a disobedient mid-stream fence: visible prefix + parsed call.

    No fence, or a fence that does not parse → the full text stays visible
    and the call is None.

    A look-alike tag (```tool_calls, ```tool_call_schema, ...) that merely
    starts with FENCE_OPEN does not parse, so scan forward past it to see
    if a real fence follows later in the text.
    """
    start = 0
    while True:
        idx = text.find(FENCE_OPEN, start)
        if idx == -1:
            return text, None
        call = parse_fenced_tool_call(text[idx:])
        if call is not None:
            return text[:idx].rstrip(), call
        start = idx + len(FENCE_OPEN)


def stream_prefix_verdict(prefix: str) -> str:
    """Sniff a stream's first tokens: tool_call, text, or undecided."""
    stripped = prefix.lstrip()
    if not stripped:
        return STREAM_UNDECIDED
    if stripped.startswith(FENCE_OPEN):
        # Matching FENCE_OPEN alone is not decisive: the stream could still
        # grow into a look-alike tag like ```tool_calls or
        # ```tool_call_schema. Only a clean line boundary (whitespace then
        # newline) after FENCE_OPEN confirms a real tool-call fence.
        after = stripped[len(FENCE_OPEN):]
        if not after:
            return STREAM_UNDECIDED
        i = 0
        while i < len(after) and after[i] in (" ", "\t"):
            i += 1
        if i == len(after):
            # Nothing but trailing spaces so far — could still become a
            # newline (tool_call) or more characters (look-alike tag).
            return STREAM_UNDECIDED
        if after[i] in ("\n", "\r"):
            return STREAM_TOOL_CALL
        return STREAM_TEXT
    if FENCE_OPEN.startswith(stripped):
        return STREAM_UNDECIDED
    return STREAM_TEXT


def render_tool_protocol(schemas: list[ToolSchema]) -> str:
    """Render the tool-protocol system-prompt section.

    Empty schema list → empty string (no protocol section: answer directly).
    """
    if not schemas:
        return ""
    blocks = []
    for schema in schemas:
        blocks.append(json.dumps(
            {"name": schema.name, "description": schema.description,
             "parameters": schema.parameters}, indent=2))
    tool_list = "\n".join(blocks)
    return (
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


# --- appended below the protocol code in tldw_chatbook/Agents/agent_runtime.py ---

from dataclasses import dataclass
from typing import Callable

from .agent_models import (
    FIND_TOOLS_NAME, LOAD_TOOLS_NAME, LOOP_DETECTION_N, RUN_CANCELLED,
    RUN_DONE, RUN_STUCK, SPAWN_TOOL_NAME, STEP_ERROR, STEP_MODEL,
    STEP_SPAWN, STEP_TOOL_CALL, STEP_TOOL_RESULT, AgentConfig, AgentStep,
    ModelTurn, RunOutcome, ToolCatalogEntry, ToolResult,
)


@dataclass
class LoopDeps:
    """Everything impure, injected. The loop itself stays pure."""

    call_model: Callable[[list, tuple], ModelTurn]
    invoke_tool: Callable[..., ToolResult]
    spawn: Callable[[str], ToolResult]
    find_tools: Callable[[str], list]
    load_schemas: Callable[[list], list]
    should_cancel: Callable[[], bool]
    clock: Callable[[], float]


def _catalog_lines(entries: list) -> str:
    if not entries:
        return "No matching tools."
    return "\n".join(
        f"{e.id} — {e.name}: {e.one_line_description}" for e in entries)


def run_agent_loop(config: AgentConfig, initial_messages: list[dict],
                   active_schemas: list, deps: LoopDeps) -> RunOutcome:
    """Drive think → (tool) → observe until done / stuck / cancelled.

    Message convention (transport-independent): assistant turns append
    verbatim; tool results append as user-role
    ``Tool result for {name}: {content}`` lines.
    """
    budget = config.budget
    steps: list[AgentStep] = []
    messages = list(initial_messages)
    active = list(active_schemas)
    started = deps.clock()
    spawned = 0
    last_key: tuple | None = None
    repeat_count = 0

    def add(kind: str, **kw) -> AgentStep:
        step = AgentStep(index=len(steps), kind=kind, **kw)
        steps.append(step)
        return step

    while True:
        if deps.should_cancel():
            return RunOutcome(RUN_CANCELLED, steps,
                              subagents_spawned=spawned)
        if len(steps) >= budget.max_steps:
            add(STEP_ERROR, summary="step budget exhausted")
            return RunOutcome(RUN_STUCK, steps, subagents_spawned=spawned)
        if deps.clock() - started > budget.max_wall_seconds:
            add(STEP_ERROR, summary="wall-clock budget exhausted")
            return RunOutcome(RUN_STUCK, steps, subagents_spawned=spawned)

        turn = deps.call_model(messages, tuple(active))
        add(STEP_MODEL, summary=turn.text[:200])

        calls = list(turn.tool_calls)
        if not calls:
            _visible, fenced = split_visible_text_and_tool_call(turn.text)
            if fenced is None:
                return RunOutcome(RUN_DONE, steps, final_text=turn.text,
                                  subagents_spawned=spawned)
            calls = [fenced]
        messages.append({"role": "assistant", "content": turn.text})

        for call in calls:
            if deps.should_cancel():
                return RunOutcome(RUN_CANCELLED, steps,
                                  subagents_spawned=spawned)
            key = (call.name, json.dumps(call.args, sort_keys=True))
            repeat_count = repeat_count + 1 if key == last_key else 1
            last_key = key
            if repeat_count >= LOOP_DETECTION_N:
                add(STEP_ERROR,
                    summary=f"loop detected: {call.name} repeated "
                            f"{repeat_count}x with identical args")
                return RunOutcome(RUN_STUCK, steps,
                                  subagents_spawned=spawned)

            if call.name == SPAWN_TOOL_NAME:
                task = str(call.args.get("task", ""))
                if spawned >= budget.max_subagents:
                    result = ToolResult(
                        ok=False, error="sub-agent budget exhausted")
                else:
                    add(STEP_SPAWN, summary=task[:200],
                        tool_name=SPAWN_TOOL_NAME, args=dict(call.args))
                    result = deps.spawn(task)
                    spawned += 1
            elif call.name == FIND_TOOLS_NAME:
                add(STEP_TOOL_CALL, tool_name=call.name,
                    args=dict(call.args))
                entries = deps.find_tools(str(call.args.get("query", "")))
                result = ToolResult(ok=True, content=_catalog_lines(entries))
            elif call.name == LOAD_TOOLS_NAME:
                add(STEP_TOOL_CALL, tool_name=call.name,
                    args=dict(call.args))
                ids = list(call.args.get("ids", []))
                loaded = deps.load_schemas(ids)
                room = budget.max_active_tools - len(active)
                accepted = loaded[:max(room, 0)]
                active.extend(accepted)
                result = ToolResult(ok=True, content="loaded: " + ", ".join(
                    s.name for s in accepted) if accepted else "no room")
            else:
                add(STEP_TOOL_CALL, tool_name=call.name,
                    args=dict(call.args))
                result = deps.invoke_tool(call)

            content = result.content if result.ok else f"ERROR: {result.error}"
            add(STEP_TOOL_RESULT, tool_name=call.name,
                result=content[:2000])
            messages.append({
                "role": "user",
                "content": f"Tool result for {call.name}: {content}"})
