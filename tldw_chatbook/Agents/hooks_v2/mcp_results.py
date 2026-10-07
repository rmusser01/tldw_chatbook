"""Native hook effects from complete, unchanged, strictly decoded MCP results."""

from __future__ import annotations

import json

from tldw_chatbook.MCP.tool_results import MCPToolResult

from .models import HookHandler, HookResult
from .validation import RESULT_BYTES, decode_result, parse_result


def normalize_hook_result(result: MCPToolResult, handler: HookHandler) -> HookResult:
    """Validate original payload before accepting any native v2 effects.

    Args:
        result: Original typed result from the normal authorized MCP invocation.
        handler: Reviewed handler whose declared effects constrain the result.

    Returns:
        The immutable validated v2 result.

    Raises:
        ValueError: For errors, missing wire evidence, overflow or ambiguous forms.
    """
    if not isinstance(result, MCPToolResult):
        raise ValueError("hook_mcp_result_unqualified")  # noqa: TRY004
    if result.transport_error is not None or result.is_error:
        raise ValueError("hook_mcp_execution_failed")
    raw = result.encoded_payload
    if raw is None or not result.duplicate_keys_checked:
        raise ValueError("hook_mcp_result_unqualified")
    if len(raw) > RESULT_BYTES:
        raise ValueError("hook_mcp_result_overflow")
    # Decode immutable ORIGINAL bytes; mutable public dictionaries are not effects.
    payload = json.loads(raw)
    content = payload.get("content", [])
    structured = payload.get("structuredContent")
    parsed = parse_result(structured, handler) if structured is not None else None
    if not content:
        return parsed or parse_result({"version": 2, "decision": "pass"}, handler)
    if (
        len(content) != 1
        or content[0].get("type") != "text"
        or not isinstance(content[0].get("text"), str)
        or not content[0]["text"].strip()
    ):
        raise ValueError("hook_mcp_result_shape")
    text = content[0]["text"]
    decoded = decode_result(text, handler)
    if structured is not None:
        # Canonical JSON compares types as well as values (True != 1).
        def canonical(value):
            return json.dumps(
                value, sort_keys=True, ensure_ascii=False, allow_nan=False
            )

        if canonical(json.loads(text)) != canonical(structured):
            raise ValueError("hook_mcp_result_conflict")
    return parsed or decoded
