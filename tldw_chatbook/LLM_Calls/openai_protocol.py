"""OpenAI function-call recognition, response projection and stream adapters."""

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


def _responses_stream_to_chat_sse(response, *, model: str):
    from . import LLM_API_Calls as _owner

    completion_id = f"chatcmpl-openai-responses-{_owner.time.time_ns()}"
    created_ts = int(_owner.time.time())
    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            line = (
                raw_line.decode("utf-8")
                if isinstance(raw_line, bytes)
                else str(raw_line)
            )
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            payload_text = line.removeprefix("data:").strip()
            try:
                event = _owner.json.loads(payload_text)
            except _owner.json.JSONDecodeError:
                continue
            event_type = event.get("type")
            if event_type == "response.output_text.delta":
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"content": event.get("delta", "")}},
                    ],
                }
                yield f"data: {_owner.json.dumps(chunk)}\n\n"
            elif event_type == "response.completed":
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                completed_usage = (event.get("response") or {}).get("usage")
                if isinstance(completed_usage, dict):
                    chunk["usage"] = completed_usage
                    _owner._session_ledger().record_provider_payload(
                        completed_usage, provider="openai", model=model
                    )
                yield f"data: {_owner.json.dumps(chunk)}\n\n"
            elif event_type == "error":
                yield f"data: {payload_text}\n\n"
    except _owner.requests.exceptions.RequestException as exc:
        _owner.logger.opt(exception=True).error(
            "OpenAI Responses: stream connection error: %s", exc
        )
        yield (
            "data: "
            + _owner.json.dumps(
                {"error": {"message": f"Stream connection error: {exc}"}}
            )
            + "\n\n"
        )
    finally:
        if response is not None:
            response.close()
    yield "data: [DONE]\n\n"


def _openai_stream_line_usage(line: str) -> dict | None:
    """Extract a usage dict from an OpenAI SSE line, best-effort.

    ``stream_options.include_usage`` is requested for chat-completions
    streams, so the final chunk carries usage; the substring guard at the
    call site avoids parsing every delta chunk (a false substring hit
    parses to a payload without a usage dict and returns None). The caller
    keeps the LAST seen usage and records once at normal loop end --
    OpenAI-compatible gateways sometimes emit cumulative usage per chunk.
    """
    from . import LLM_API_Calls as _owner

    try:
        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            return None
        payload = _owner.json.loads(data)
        usage = payload.get("usage")
        if isinstance(usage, dict) and usage:
            return usage
    except Exception:
        pass
    return None
