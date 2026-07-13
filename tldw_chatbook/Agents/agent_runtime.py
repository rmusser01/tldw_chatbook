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
