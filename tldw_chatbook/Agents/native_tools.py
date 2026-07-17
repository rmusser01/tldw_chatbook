# tldw_chatbook/Agents/native_tools.py
"""Native provider tool-calls: capability check, conversion, parsing.

The fence-first text protocol (``agent_runtime.render_tool_protocol``)
remains the fallback for every provider not listed here — see the
vertical-slice spec and the task-231 tool-call flow review (opportunity 1).

A provider earns a place in ``NATIVE_TOOLS_PROVIDERS`` only when ALL of:

1. ``PROVIDER_PARAM_MAP`` forwards ``tools`` (and the handler accepts it),
2. the handler returns the RAW OpenAI-compatible response dict, so
   ``choices[0].message.tool_calls`` survives verbatim (the anthropic /
   google / cohere handlers normalize and currently DROP tool-use blocks —
   they stay fence-only until their normalizers build
   ``message.tool_calls``; follow-up filed as task-246), and
3. the provider accepts OpenAI-shape ``role: "tool"`` history messages.

Pure module: no I/O, no provider imports.
"""
from __future__ import annotations

import json

from .agent_models import ToolCall, ToolSchema

NATIVE_TOOLS_PROVIDERS = frozenset({
    "openai", "groq", "openrouter", "mistral", "deepseek", "moonshot",
    "custom-openai-api", "custom-openai-api-2",
})


def provider_supports_native_tools(api_endpoint: str | None) -> bool:
    """Return whether ``api_endpoint`` supports native tool-calls end-to-end.

    Args:
        api_endpoint: The ``chat_api_call`` provider key. The Console passes
            ``ConsoleProviderResolution.execution_key`` — the key
            ``PROVIDER_PARAM_MAP`` is indexed by.

    Returns:
        True when the provider forwards ``tools=`` AND returns the raw
        OpenAI-compatible response shape (see module docstring).
    """
    return str(api_endpoint or "").strip().lower() in NATIVE_TOOLS_PROVIDERS


def schemas_to_openai_tools(schemas: list[ToolSchema]) -> list[dict]:
    """Convert ``ToolSchema`` entries to the OpenAI ``tools=`` wire format.

    Args:
        schemas: Disclosed tool schemas (runtime + active), in order.

    Returns:
        One ``{"type": "function", "function": {...}}`` entry per schema;
        an empty ``parameters`` dict is replaced with the minimal valid
        object schema (providers reject ``{}``).
    """
    tools = []
    for schema in schemas:
        tools.append({
            "type": "function",
            "function": {
                "name": schema.name,
                "description": schema.description,
                # A fresh literal per schema: a shared module-level default
                # would leak downstream mutations across conversions through
                # its nested "properties" dict (PR #648 review).
                "parameters": schema.parameters or {"type": "object",
                                                    "properties": {}},
            },
        })
    return tools


def ensure_tool_call_ids(raw_calls: list | None) -> list:
    """Return tool-call entries with every dict entry carrying an id.

    Some OpenAI-compatible servers omit tool-call ids. An id-less call would
    split the history convention — the assistant echo carries the id-less
    entry while its result falls back to a fence-style user-role line —
    which strict providers reject on the next request (PR #648 review).
    Missing ids get a synthesized ``call_<position>`` so the echo and its
    ``role="tool"`` reply always pair; the caller must use the SAME
    normalized list for both the echo and parsing.

    Args:
        raw_calls: The raw ``message.tool_calls`` list (or None).

    Returns:
        A new list where every dict entry has a non-empty ``id``; entries
        with ids and non-dict junk pass through untouched.
    """
    normalized = []
    for position, raw in enumerate(raw_calls or []):
        if isinstance(raw, dict) and not raw.get("id"):
            raw = {**raw, "id": f"call_{position}"}
        normalized.append(raw)
    return normalized


def parse_native_tool_calls(message: dict | None) -> tuple[ToolCall, ...]:
    """Parse OpenAI-shape ``message.tool_calls`` into ``ToolCall`` entries.

    Malformed ``arguments`` JSON yields ``args={}`` rather than dropping
    the call: the downstream tool's own validation error is echoed back to
    the model as a normal tool result, so it can retry with corrected
    arguments. Entries without a function name are dropped.

    Args:
        message: The ``choices[0].message`` dict from a provider response
            (or anything — junk yields no calls).

    Returns:
        Parsed calls in provider order, each carrying its ``call_id``.
    """
    if not isinstance(message, dict):
        return ()
    calls = []
    for raw in message.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        function = raw.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        raw_args = function.get("arguments")
        args: dict = {}
        if isinstance(raw_args, dict):
            args = raw_args
        elif isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                args = parsed
        calls.append(ToolCall(name=name, args=args,
                              call_id=str(raw.get("id") or "")))
    return tuple(calls)
