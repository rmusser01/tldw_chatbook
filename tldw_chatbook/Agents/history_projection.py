"""Rewrite accumulated history into another provider's tool protocol.

ADR-110. `agent_runtime._append_tool_result` writes a `role="tool"` message
paired by `tool_call_id` for providers with native tool-calling, and a
`FENCE_TOOL_RESULT_PREFIX` user message for everyone else. Handing one
protocol's history to the other does not fail loudly -- the provider accepts it
and the model gets confused -- so a mid-run provider switch projects first.

Two properties matter more than elegance here:

* **Total.** Every message in equals a message out, in order. A projection that
  silently drops a turn loses exactly the work the fallback exists to preserve.
* **Refusing beats guessing.** Anything that cannot be projected faithfully
  raises `ProjectionError`, and the caller abandons the fallback rather than
  sending a degraded history.
"""

from __future__ import annotations

import json
from typing import Any

from .agent_models import FENCE_TOOL_RESULT_PREFIX

#: Must match `agent_runtime.FENCE_OPEN`. Imported lazily in `_parse_fence` to
#: keep this module free of a runtime import cycle.
_FENCE_OPEN = "```tool_call"
_FENCE_CLOSE = "```"

#: Appended when a call has no recorded result. The alternative -- dropping the
#: call -- would make the model believe it never asked (ADR-110 decision 2).
NO_RESULT_MARKER = "(no result recorded)"


class ProjectionError(ValueError):
    """History cannot be projected faithfully into the target protocol."""


def _is_native_call_turn(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and bool(message.get("tool_calls"))
    )


def _fence_text(name: str, arguments: Any) -> str:
    body = json.dumps({"name": name, "arguments": arguments})
    return f"{_FENCE_OPEN}\n{body}\n{_FENCE_CLOSE}"


def _decode_arguments(raw: Any) -> Any:
    """Native `arguments` is a JSON string; fence carries a real object."""
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Preserve rather than lose it: the fence body keeps the raw text, and
        # the receiving side reports a malformed argument the same way it
        # would have for a native call.
        return raw


def _native_call_parts(message: dict) -> list[tuple[str, str, Any]]:
    """Extract ``(call_id, name, arguments)`` from a native assistant turn."""
    parts: list[tuple[str, str, Any]] = []
    for entry in message.get("tool_calls") or ():
        if not isinstance(entry, dict):
            raise ProjectionError("tool_calls entry is not an object")
        function = entry.get("function")
        if not isinstance(function, dict):
            raise ProjectionError("tool_calls entry has no function body")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ProjectionError("tool_calls entry has no function name")
        call_id = entry.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ProjectionError("tool_calls entry has no id")
        parts.append((call_id, name, _decode_arguments(function.get("arguments"))))
    if not parts:
        raise ProjectionError("assistant turn declares tool_calls but carries none")
    return parts


def _parse_fence(text: str) -> tuple[str, Any] | None:
    """Parse a real tool-call fence, or None for text/look-alikes."""
    from .agent_runtime import parse_fenced_tool_call

    call = parse_fenced_tool_call(text)
    if call is None:
        return None
    return call.name, call.args


def _native_to_fence(messages: list[dict]) -> list[dict]:
    # Results are keyed by call id so a batch keeps its pairing, and an
    # unpaired call is detectable rather than silently absorbed.
    results: dict[str, str] = {
        str(m.get("tool_call_id")): str(m.get("content") or "")
        for m in messages
        if isinstance(m, dict) and m.get("role") == "tool"
    }

    projected: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            projected.append(message)
            continue
        if message.get("role") == "tool":
            call_id = str(message.get("tool_call_id"))
            projected.append(
                {
                    "role": "user",
                    "content": (
                        f"{FENCE_TOOL_RESULT_PREFIX}{_name_for(messages, call_id)}: "
                        f"{message.get('content') or ''}"
                    ),
                }
            )
            continue
        if not _is_native_call_turn(message):
            projected.append(dict(message))
            continue

        parts = _native_call_parts(message)
        chunks = []
        leading = str(message.get("content") or "").strip()
        if leading:
            chunks.append(leading)
        for call_id, name, arguments in parts:
            chunks.append(_fence_text(name, arguments))
            if call_id not in results:
                chunks.append(f"{FENCE_TOOL_RESULT_PREFIX}{name}: {NO_RESULT_MARKER}")
        projected.append({"role": "assistant", "content": "\n".join(chunks)})
    return projected


def _name_for(messages: list[dict], call_id: str) -> str:
    for message in messages:
        if _is_native_call_turn(message):
            for entry in message.get("tool_calls") or ():
                if isinstance(entry, dict) and entry.get("id") == call_id:
                    function = entry.get("function")
                    if isinstance(function, dict):
                        name = function.get("name")
                        if isinstance(name, str) and name:
                            return name
    return "tool"


def _fence_to_native(messages: list[dict]) -> list[dict]:
    projected: list[dict] = []
    pending: list[tuple[str, str]] = []  # (call_id, name) awaiting a result

    for message in messages:
        if not isinstance(message, dict):
            projected.append(message)
            continue

        content = str(message.get("content") or "")

        if message.get("role") == "user" and content.startswith(
            FENCE_TOOL_RESULT_PREFIX
        ):
            remainder = content[len(FENCE_TOOL_RESULT_PREFIX) :]
            name, _, result = remainder.partition(": ")
            call_id = _take_pending(pending, name)
            if call_id is None:
                # A result with no preceding call: keep it as ordinary user
                # text rather than inventing a pairing the model never made.
                projected.append(dict(message))
                continue
            projected.append(
                {"role": "tool", "tool_call_id": call_id, "content": result}
            )
            continue

        if message.get("role") == "assistant":
            parsed = _parse_fence(content)
            if parsed is not None:
                name, arguments = parsed
                call_id = f"proj_{len(projected)}_{name}"
                pending.append((call_id, name))
                projected.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    }
                )
                continue

        projected.append(dict(message))
    return projected


def _take_pending(pending: list[tuple[str, str]], name: str) -> str | None:
    for index, (call_id, pending_name) in enumerate(pending):
        if pending_name == name:
            pending.pop(index)
            return call_id
    return None


def project_history_for_protocol(
    messages: list[dict], *, native: bool
) -> list[dict]:
    """Return ``messages`` rewritten for the target tool protocol.

    Args:
        messages: The accumulated run history.
        native: Whether the TARGET provider supports native tool-calls
            (`native_tools.provider_supports_native_tools`).

    Returns:
        A new list of the same length, in the same order. The input is never
        mutated.

    Raises:
        ProjectionError: The history cannot be projected faithfully. The caller
            must abandon the fallback rather than send a degraded history.
    """
    if not isinstance(messages, list):
        raise ProjectionError("history is not a list")

    has_native = any(_is_native_call_turn(m) for m in messages) or any(
        isinstance(m, dict) and m.get("role") == "tool" for m in messages
    )
    if native:
        # Validate even on the no-op path: a malformed native turn must be
        # reported before a switch, not carried into one.
        for message in messages:
            if _is_native_call_turn(message):
                _native_call_parts(message)
        return messages if has_native else _fence_to_native(messages)

    if not has_native:
        return messages
    return _native_to_fence(messages)
