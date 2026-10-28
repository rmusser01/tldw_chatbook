"""OpenAI function-call recognition and Responses payload projection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _openai_has_chat_function_tools(tools: object) -> bool:
    """Recognize canonical Chat functions without translating or changing schemas."""
    if not isinstance(tools, list) or not tools:
        return False
    for tool in tools:
        if not isinstance(tool, Mapping) or set(tool) != {"type", "function"}:
            return False
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, Mapping):
            return False
        if set(function) - {"name", "description", "parameters", "strict"}:
            return False
        name = function.get("name")
        parameters = function.get("parameters")
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(parameters, Mapping)
            or parameters.get("type") != "object"
        ):
            return False
    return True


def _extract_openai_responses_text(response_data: dict[str, Any]) -> str:
    output_text = response_data.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    text_parts: list[str] = []
    for item in response_data.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text_parts.append(str(part.get("text") or ""))
    return "\n".join(part for part in text_parts if part).strip()


def _normalize_openai_responses_payload(
    response_data: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    """Normalize Responses API output to the chat-completions shape used by Console."""
    from . import LLM_API_Calls as api

    return {
        "id": response_data.get("id", f"resp-openai-{api.time.time_ns()}"),
        "object": "chat.completion",
        "created": int(response_data.get("created_at") or api.time.time()),
        "model": response_data.get("model", model),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": api._extract_openai_responses_text(response_data),
                },
                "finish_reason": "stop"
                if response_data.get("status") in {None, "completed"}
                else response_data.get("status"),
            }
        ],
        "usage": response_data.get("usage"),
    }
