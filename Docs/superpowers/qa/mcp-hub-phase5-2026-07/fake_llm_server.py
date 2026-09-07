#!/usr/bin/env python3
"""Tiny scripted OpenAI-compatible endpoint for MCP Hub Phase 5 live QA.

Stands in for a real LLM so a genuine Console agent turn can be driven
through the app's approval flow end to end. Speaks just enough of the
OpenAI `POST /v1/chat/completions` wire shape for
`tldw_chatbook.LLM_Calls.LLM_API_Calls_Local.chat_with_custom_openai` (via
`_chat_with_openai_compatible_local_server`, non-streaming) to round-trip a
real `ConsoleProviderGateway` / `ConsoleAgentBridge` turn.

## Why this isn't just "first call = tool_calls, subsequent = plain text"

The task brief's own MVP description was a single-round-trip sketch. Two
live facts discovered while wiring this up made a literal implementation of
that sketch unusable, so this file generalizes it (documented here, and
again in the QA README's "fake-LLM design" section):

1. **Progressive tool disclosure.** This build's agent runtime
   (`Agents/tool_catalog.py`/`agent_runtime.py`) only discloses the full
   MCP tool catalog directly when it's small; once over
   `DIRECT_DISCLOSE_THRESHOLD` (this HOME's catalog is -- 10 builtin MCP
   tools + 3 docs-server tools) it instead offers just three meta-tools
   (`find_tools`, `load_tools`, `spawn_subagent`) and the model must call
   `find_tools(query=...)` then `load_tools(ids=[...])` before the real
   tool (e.g. `mcp__tldw_chatbook__list_characters`) ever appears in a
   `tools=` list it can call. That's 3-4 `/v1/chat/completions` round trips
   per turn, not 1 -- confirmed live (see `fake_llm_requests.log` captured
   during the QA round) before writing this version.
2. **Multi-turn captures.** The session-approval-no-reprompt capture needs
   TWO separate fresh turns to call the same tool. A global "first call
   ever" counter can't distinguish "first call of turn 2" from "some later
   call of turn 1".

## Decision policy (stateless; re-derived from the request body every call)

For the most recent **user** message, match a trigger phrase (see
`TRIGGERS`) to pick a target MCP tool + call arguments (default:
`list_characters`). Then:

  * If the request's last message is NOT role=="tool" (a fresh step): call
    the target tool directly if it's already in this request's own
    `tools=` list; otherwise call `find_tools` if that's offered; otherwise
    (neither offered -- e.g. the MCP kill switch cut the target tool from
    the catalog entirely) reply with plain fallback text.
  * If the last message IS role=="tool": look at the immediately preceding
    assistant message's own `tool_calls[0].function.name` to see which
    call this result answers:
      - answers `find_tools` -> if the target tool's name appears in the
        result content, call `load_tools(ids=[target])`; else plain
        fallback text (truly not found).
      - answers `load_tools` -> call the target tool.
      - answers the target tool itself -> plain completion text (the
        turn's real "final answer" -- what capture 2/3/4 actually show in
        the transcript).
      - anything else -> generic "Done." plain text.

No frameworks, stdlib only (`http.server`), single file, no repo imports
(deliberately independent of the app under test).
"""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 8899

# Builtin MCP server key is "builtin:tldw_chatbook"; tool_naming.llm_tool_name
# strips the "builtin:" label prefix and joins with "__", giving
# "mcp__tldw_chatbook__<tool>" (verified live against tool_naming.py before
# writing this file -- see the QA README). MCPToolProvider.list_catalog()
# sets a catalog entry's `id` equal to this SAME llm-facing name (mcp_tool_
# provider.py), so it's also exactly the id `load_tools(ids=[...])` expects.
LIST_CHARACTERS_TOOL = "mcp__tldw_chatbook__list_characters"
SEARCH_NOTES_TOOL = "mcp__tldw_chatbook__search_notes"

FIND_TOOLS = "find_tools"
LOAD_TOOLS = "load_tools"

# (trigger substring in the user's message, case-insensitive) -> (llm tool
# name, tool-call arguments JSON, find_tools query)
TRIGGERS: list[tuple[str, str, str, str]] = [
    ("search the notes", SEARCH_NOTES_TOOL, '{"query": "onboarding"}', "search_notes"),
    ("list the characters", LIST_CHARACTERS_TOOL, "{}", "list_characters"),
]
DEFAULT_TOOL = LIST_CHARACTERS_TOOL
DEFAULT_ARGS = "{}"
DEFAULT_QUERY = "list_characters"

# Plain-text completion line per tool, used once its result (or refusal) is
# fed back as the newest role="tool" message.
COMPLETION_TEXT = {
    LIST_CHARACTERS_TOOL: "Done — listed the characters.",
    SEARCH_NOTES_TOOL: "Done — searched the notes.",
}
NOT_FOUND_TEXT = "No matching MCP tool was found in the catalog for this request."
FALLBACK_TEXT = "No matching MCP tool was offered for this request."

_call_counter_lock = threading.Lock()
_call_counter = 0


def _next_call_id() -> str:
    global _call_counter
    with _call_counter_lock:
        _call_counter += 1
        return f"call_{_call_counter}"


def _offered_tool_names(tools: object) -> set[str]:
    names: set[str] = set()
    if isinstance(tools, list):
        for entry in tools:
            if isinstance(entry, dict):
                fn = entry.get("function")
                if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                    names.add(fn["name"])
    return names


def _last_user_content(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else str(content or "")
    return ""


def _resolve_target(messages: list) -> tuple[str, str, str]:
    """Return (llm_tool_name, arguments_json, find_tools_query) for this turn."""
    haystack = _last_user_content(messages).lower()
    for phrase, tool_name, args_json, query in TRIGGERS:
        if phrase in haystack:
            return tool_name, args_json, query
    return DEFAULT_TOOL, DEFAULT_ARGS, DEFAULT_QUERY


def _prev_call_name(messages: list) -> str | None:
    """Name of the function the trailing role=='tool' message answers."""
    if len(messages) < 2:
        return None
    prev = messages[-2]
    if not isinstance(prev, dict):
        return None
    calls = prev.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    first = calls[0]
    if not isinstance(first, dict):
        return None
    fn = first.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        return name if isinstance(name, str) else None
    return None


def _tool_call_message(name: str, args_json: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": _next_call_id(),
                "type": "function",
                "function": {"name": name, "arguments": args_json},
            }
        ],
    }


def decide(messages: list, offered: set[str]) -> tuple[str, dict]:
    """Return ("tool_calls"|"stop", message_dict) for this request."""
    target_name, target_args, query = _resolve_target(messages)
    last = messages[-1] if messages else None
    last_role = last.get("role") if isinstance(last, dict) else None

    if last_role != "tool":
        # A fresh step (no tool result awaiting a reply yet this turn).
        if target_name in offered:
            return "tool_calls", _tool_call_message(target_name, target_args)
        if FIND_TOOLS in offered:
            return "tool_calls", _tool_call_message(
                FIND_TOOLS, json.dumps({"query": query})
            )
        return "stop", {"role": "assistant", "content": FALLBACK_TEXT}

    prev_name = _prev_call_name(messages)
    if prev_name == FIND_TOOLS:
        content = str(last.get("content") or "")
        if target_name in content:
            return "tool_calls", _tool_call_message(
                LOAD_TOOLS, json.dumps({"ids": [target_name]})
            )
        return "stop", {"role": "assistant", "content": NOT_FOUND_TEXT}
    if prev_name == LOAD_TOOLS:
        return "tool_calls", _tool_call_message(target_name, target_args)
    if prev_name == target_name:
        text = COMPLETION_TEXT.get(target_name, "Done.")
        return "stop", {"role": "assistant", "content": text}
    # Unknown previous call (e.g. spawn_subagent, or a name we don't
    # recognize) -- generic finish rather than looping forever.
    return "stop", {"role": "assistant", "content": "Done."}


def _build_response(model: str, message: dict, finish_reason: str) -> dict:
    return {
        "id": f"fakecmpl-{_next_call_id()}",
        "object": "chat.completion",
        "created": int(datetime.now(tz=timezone.utc).timestamp()),
        "model": model or "fake-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "FakeLLM/1.0"

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - stdlib signature
        sys.stderr.write(
            f"[fake_llm_server {datetime.now().isoformat(timespec='seconds')}] "
            f"{fmt % args}\n"
        )

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        if self.path.startswith("/v1/models"):
            self._send_json(
                {"object": "list", "data": [{"id": "fake-model", "object": "model"}]}
            )
            return
        self._send_json({"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib method name
        if not self.path.startswith("/v1/chat/completions"):
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            payload = {}

        messages = payload.get("messages") or []
        tools = payload.get("tools")
        model = payload.get("model") or "fake-model"
        offered = _offered_tool_names(tools)
        last_role = messages[-1].get("role") if messages and isinstance(messages[-1], dict) else None

        finish_reason, message = decide(messages, offered)

        self.log_message(
            "chat.completions: last_role=%s offered=%s last_user=%r -> %s %r",
            last_role, sorted(offered), _last_user_content(messages)[:60],
            finish_reason,
            message.get("tool_calls") or message.get("content"),
        )
        self._send_json(_build_response(model, message, finish_reason))


def main() -> None:
    port = DEFAULT_PORT
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    sys.stderr.write(f"fake_llm_server listening on 127.0.0.1:{port}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
