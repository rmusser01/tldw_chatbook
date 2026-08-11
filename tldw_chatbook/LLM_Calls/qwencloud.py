"""Pure request translation for QwenCloud's compatible API modes."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Literal
from urllib.parse import urlsplit

from tldw_chatbook.Chat.Chat_Deps import ChatBadRequestError, ChatConfigurationError

QwenCloudAPIMode = Literal["responses", "chat_completions"]

_DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
_DEFAULT_KEY_ENV_VAR = "DASHSCOPE_API_KEY"
_API_MODES: frozenset[str] = frozenset({"responses", "chat_completions"})
_ENDPOINT_SUFFIXES = ("/chat/completions", "/responses")
_REASONING_EFFORTS: frozenset[str] = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)


def _configuration_error(message: str) -> ChatConfigurationError:
    return ChatConfigurationError(provider="qwencloud", message=message)


def _bad_request(message: str) -> ChatBadRequestError:
    return ChatBadRequestError(provider="qwencloud", message=message)


def normalize_qwencloud_api_mode(
    api_mode: str | None,
    *,
    provider_settings: Mapping[str, Any] | None = None,
) -> QwenCloudAPIMode:
    """Resolve and validate the selected QwenCloud API mode.

    Args:
        api_mode: Explicit trusted-caller override, when supplied.
        provider_settings: QwenCloud's isolated provider settings.

    Returns:
        The normalized supported API mode.

    Raises:
        ChatConfigurationError: If the resolved value is not an exact mode.
    """
    configured = provider_settings.get("api_mode") if provider_settings else None
    candidate = api_mode if api_mode is not None else configured
    if candidate is None:
        candidate = "responses"
    if not isinstance(candidate, str):
        raise _configuration_error("QwenCloud API mode must be a string.")

    normalized = candidate.strip().lower()
    if normalized not in _API_MODES:
        raise _configuration_error(
            "QwenCloud API mode must be 'responses' or 'chat_completions'."
        )
    return normalized  # type: ignore[return-value]


def normalize_qwencloud_base_url(api_base_url: str | None) -> str:
    """Normalize a QwenCloud base URL without constructing an endpoint.

    Args:
        api_base_url: Configured compatible-API base or pasted endpoint.

    Returns:
        A validated base URL without a recognized terminal endpoint suffix.

    Raises:
        ChatConfigurationError: If the URL is unsafe or malformed.
    """
    candidate = _DEFAULT_BASE_URL if api_base_url is None else api_base_url
    if not isinstance(candidate, str) or not candidate.strip():
        raise _configuration_error("QwenCloud API base URL is required.")
    candidate = candidate.strip().rstrip("/")
    if (
        any(character.isspace() for character in candidate)
        or "?" in candidate
        or "#" in candidate
    ):
        raise _configuration_error("QwenCloud API base URL is malformed.")

    try:
        parsed = urlsplit(candidate)
        parsed_port = parsed.port
    except ValueError as exc:
        raise _configuration_error("QwenCloud API base URL is malformed.") from exc

    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise _configuration_error(
            "QwenCloud API base URL must be an absolute HTTP(S) URL."
        )
    if parsed.username is not None or parsed.password is not None:
        raise _configuration_error(
            "QwenCloud API base URL must not contain credentials."
        )
    if parsed.query or parsed.fragment:
        raise _configuration_error(
            "QwenCloud API base URL must not contain a query or fragment."
        )
    if parsed.netloc.endswith(":") or (
        parsed_port is not None and not 0 < parsed_port < 65536
    ):
        raise _configuration_error("QwenCloud API base URL is malformed.")
    if (
        "\\" in parsed.path
        or "//" in parsed.path
        or re.search(r"%(?![0-9A-Fa-f]{2})", parsed.path) is not None
        or any(character.isspace() for character in parsed.path)
        or any(segment in {".", ".."} for segment in parsed.path.split("/"))
    ):
        raise _configuration_error("QwenCloud API base URL path is malformed.")

    path = parsed.path.rstrip("/")
    if path.endswith("/models"):
        raise _configuration_error(
            "QwenCloud API base URL must not use the models endpoint."
        )
    if "/responses/" in path or "/chat/completions/" in path:
        raise _configuration_error(
            "QwenCloud API base URL contains a non-terminal request endpoint."
        )
    for suffix in _ENDPOINT_SUFFIXES:
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            if any(path.endswith(other) for other in _ENDPOINT_SUFFIXES):
                raise _configuration_error(
                    "QwenCloud API base URL contains a repeated endpoint suffix."
                )
            break

    authority = parsed.netloc
    return f"{parsed.scheme.lower()}://{authority}{path}".rstrip("/")


def resolve_qwencloud_api_key(
    explicit_api_key: str | None,
    *,
    provider_settings: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve QwenCloud credentials from provider-isolated sources.

    Args:
        explicit_api_key: Credential already resolved by a trusted caller.
        provider_settings: QwenCloud's isolated provider settings.
        environ: Environment mapping, injectable for pure tests.

    Returns:
        The resolved credential value.

    Raises:
        ChatConfigurationError: If no QwenCloud credential is configured.
    """
    if isinstance(explicit_api_key, str) and explicit_api_key.strip():
        return explicit_api_key

    settings = provider_settings or {}
    configured = settings.get("api_key")
    if isinstance(configured, str) and configured.strip():
        return configured

    env_name = settings.get("api_key_env_var", _DEFAULT_KEY_ENV_VAR)
    if not isinstance(env_name, str) or not env_name.strip():
        env_name = _DEFAULT_KEY_ENV_VAR
    environment = os.environ if environ is None else environ
    from_environment = environment.get(env_name.strip())
    if isinstance(from_environment, str) and from_environment.strip():
        return from_environment

    raise _configuration_error("QwenCloud API key is required but not configured.")


def _normalize_message_content(
    role: str,
    content: Any,
    *,
    has_tool_calls: bool,
) -> tuple[Any, Any]:
    if content is None:
        if role == "assistant" and has_tool_calls:
            return None, None
        raise _bad_request("QwenCloud message content must be text or text parts.")
    if isinstance(content, str):
        if role == "assistant":
            return content, [{"type": "output_text", "text": content}]
        return content, content
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        raise _bad_request("QwenCloud message content must be text or text parts.")

    chat_parts: list[dict[str, Any]] = []
    responses_parts: list[dict[str, Any]] = []
    text_values: list[str] = []
    has_image = False
    for part in content:
        if not isinstance(part, Mapping):
            raise _bad_request("QwenCloud message content part is malformed.")
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise _bad_request("QwenCloud text content must be a string.")
            text_values.append(text)
            copied_part = deepcopy(dict(part))
            chat_parts.append(copied_part)
            responses_parts.append({"type": "input_text", "text": text})
            continue
        if part_type == "image_url":
            if role != "user":
                raise _bad_request(
                    "QwenCloud images are supported only for user messages."
                )
            image_url = part.get("image_url")
            if not isinstance(image_url, Mapping):
                raise _bad_request("QwenCloud image URL content is malformed.")
            url = image_url.get("url")
            if not isinstance(url, str) or not url:
                raise _bad_request("QwenCloud image URL must be a non-empty string.")
            has_image = True
            chat_parts.append(deepcopy(dict(part)))
            responses_parts.append({"type": "input_image", "image_url": url})
            continue
        raise _bad_request("QwenCloud message content type is unsupported.")

    if not has_image:
        collapsed = "".join(text_values)
        if role == "assistant":
            return collapsed, [{"type": "output_text", "text": collapsed}]
        return collapsed, collapsed
    return chat_parts, responses_parts


def _validate_tool_calls(
    raw_calls: Any,
    *,
    seen_call_ids: set[str],
) -> list[dict[str, str]]:
    if (
        not isinstance(raw_calls, Sequence)
        or isinstance(raw_calls, (str, bytes))
        or not raw_calls
    ):
        raise _bad_request("QwenCloud assistant tool calls must be a non-empty list.")

    validated: list[dict[str, str]] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping) or raw_call.get("type") != "function":
            raise _bad_request("QwenCloud history supports only function tool calls.")
        call_id = raw_call.get("id")
        function = raw_call.get("function")
        if not isinstance(call_id, str) or not call_id.strip():
            raise _bad_request("QwenCloud tool call IDs must be non-empty strings.")
        if call_id in seen_call_ids:
            raise _bad_request("QwenCloud tool call IDs must be unique.")
        if not isinstance(function, Mapping):
            raise _bad_request("QwenCloud function call history is malformed.")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name.strip():
            raise _bad_request("QwenCloud function call names must be non-empty.")
        if not isinstance(arguments, str):
            raise _bad_request(
                "QwenCloud function call arguments must be a JSON string."
            )
        try:
            decoded_arguments = json.loads(arguments)
        except (TypeError, ValueError) as exc:
            raise _bad_request(
                "QwenCloud function call arguments must contain valid JSON."
            ) from exc
        if not isinstance(decoded_arguments, Mapping):
            raise _bad_request(
                "QwenCloud function call arguments must encode an object."
            )
        seen_call_ids.add(call_id)
        validated.append({"call_id": call_id, "name": name, "arguments": arguments})
    return validated


def _translate_messages(
    system_message: str | None,
    messages_payload: Sequence[Mapping[str, Any]],
) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]]:
    if system_message is not None and not isinstance(system_message, str):
        raise _bad_request("QwenCloud system instructions must be a string.")
    messages: list[dict[str, Any]] = []
    for message in messages_payload:
        if not isinstance(message, Mapping):
            raise _bad_request("QwenCloud messages must be objects.")
        messages.append(deepcopy(dict(message)))

    leading_system: str | None = None
    if messages and messages[0].get("role") == "system":
        system_row = messages.pop(0)
        normalized, _ = _normalize_message_content(
            "system", system_row.get("content"), has_tool_calls=False
        )
        if not isinstance(normalized, str):
            raise _bad_request("QwenCloud system content must contain only text.")
        leading_system = normalized
    if leading_system is not None and system_message is not None:
        if leading_system != system_message:
            raise _bad_request("QwenCloud system instructions conflict.")
    instructions = system_message if system_message is not None else leading_system

    chat_messages: list[dict[str, Any]] = []
    responses_input: list[dict[str, Any]] = []
    if instructions is not None:
        chat_messages.append({"role": "system", "content": instructions})

    seen_call_ids: set[str] = set()
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if not isinstance(role, str) or role not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            raise _bad_request("QwenCloud message role is unsupported.")
        if role == "system":
            raise _bad_request("QwenCloud accepts a system message only at the start.")
        if role == "tool":
            raise _bad_request("QwenCloud tool result has no preceding call batch.")
        if "tool_calls" in message and role != "assistant":
            raise _bad_request("QwenCloud tool calls require the assistant role.")

        has_tool_calls = "tool_calls" in message
        chat_content, responses_content = _normalize_message_content(
            role, message.get("content"), has_tool_calls=has_tool_calls
        )
        chat_message: dict[str, Any] = {"role": role, "content": chat_content}

        if not has_tool_calls:
            chat_messages.append(chat_message)
            responses_input.append({"role": role, "content": responses_content})
            index += 1
            continue

        validated_calls = _validate_tool_calls(
            message.get("tool_calls"), seen_call_ids=seen_call_ids
        )
        chat_message["tool_calls"] = deepcopy(message["tool_calls"])
        chat_messages.append(chat_message)
        if chat_content not in (None, ""):
            responses_input.append({"role": "assistant", "content": responses_content})

        expected_ids = {call["call_id"] for call in validated_calls}
        results: dict[str, str] = {}
        result_index = index + 1
        while (
            result_index < len(messages)
            and messages[result_index].get("role") == "tool"
        ):
            result = messages[result_index]
            call_id = result.get("tool_call_id")
            output = result.get("content")
            if not isinstance(call_id, str) or not call_id.strip():
                raise _bad_request(
                    "QwenCloud tool results require a non-empty call ID."
                )
            if call_id not in expected_ids:
                raise _bad_request(
                    "QwenCloud tool result does not match this call batch."
                )
            if call_id in results:
                raise _bad_request("QwenCloud tool results must be unique per call.")
            if not isinstance(output, str):
                raise _bad_request("QwenCloud tool result content must be a string.")
            results[call_id] = output
            chat_messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": output}
            )
            result_index += 1
        if set(results) != expected_ids:
            raise _bad_request("QwenCloud tool call batch is missing paired results.")

        for call in validated_calls:
            call_id = call["call_id"]
            responses_input.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": call["name"],
                    "arguments": call["arguments"],
                }
            )
            responses_input.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": results[call_id],
                }
            )
        index = result_index

    return instructions, chat_messages, responses_input


def _translate_function_tools(
    tools: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if tools is None:
        return [], []

    chat_tools: list[dict[str, Any]] = []
    responses_tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, Mapping) or tool.get("type") != "function":
            raise _bad_request("QwenCloud supports only existing function tools.")
        function = tool.get("function")
        if not isinstance(function, Mapping):
            raise _bad_request("QwenCloud function tool definition is malformed.")
        unsupported_fields = set(function) - {
            "name",
            "description",
            "parameters",
            "strict",
        }
        if unsupported_fields:
            raise _bad_request("QwenCloud function tool fields are unsupported.")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise _bad_request("QwenCloud function tool names must be non-empty.")
        if name in names:
            raise _bad_request("QwenCloud function tool names must be unique.")
        names.add(name)
        parameters = function.get("parameters")
        if (
            not isinstance(parameters, Mapping)
            or parameters.get("type", "object") != "object"
        ):
            raise _bad_request(
                "QwenCloud function parameters must be an object schema."
            )

        copied_tool = deepcopy(dict(tool))
        copied_function = deepcopy(dict(function))
        chat_tools.append(copied_tool)
        responses_tools.append({"type": "function", **copied_function})
    return chat_tools, responses_tools


def build_qwencloud_payload(
    *,
    api_mode: QwenCloudAPIMode,
    model: str,
    system_message: str | None,
    messages_payload: Sequence[Mapping[str, Any]],
    streaming: bool,
    tools: Sequence[Mapping[str, Any]] | None = None,
    tool_choice: str | Mapping[str, Any] | None = None,
    temp: float | None = None,
    topp: float | None = None,
    topk: int | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    presence_penalty: float | None = None,
    stop: str | Sequence[str] | None = None,
    response_format: Mapping[str, str] | None = None,
    n: int | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Build a fail-closed request payload for one QwenCloud API mode.

    Args:
        api_mode: Validated external API mode.
        model: QwenCloud model identifier.
        system_message: Optional separate leading instruction.
        messages_payload: Canonical Chatbook chat-shaped history.
        streaming: Whether the caller requests a streaming response.
        tools: Existing nested OpenAI function-tool definitions.
        tool_choice: Supported function-tool selection policy.
        temp: Sampling temperature.
        topp: Nucleus-sampling probability.
        topk: Chat Completions top-k sampling value.
        max_tokens: Maximum generated tokens.
        seed: Chat Completions random seed.
        presence_penalty: Chat Completions presence penalty.
        stop: Chat Completions stop sequence or sequences.
        response_format: Chat Completions response-format object.
        n: Chat Completions result count.
        logprobs: Chat Completions log-probability flag.
        top_logprobs: Chat Completions returned log-probability count.
        reasoning_effort: Mode-specific reasoning-effort value.

    Returns:
        A newly allocated request dictionary containing only allowed keys.

    Raises:
        ChatBadRequestError: If request history or parameters are unsupported.
    """
    if tool_choice is not None and tool_choice not in ("auto", "none"):
        raise _bad_request("Unsupported QwenCloud function tool choice.")
    chat_tools, responses_tools = _translate_function_tools(tools)
    instructions, chat_messages, responses_input = _translate_messages(
        system_message, messages_payload
    )

    if api_mode == "chat_completions":
        chat_payload: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "stream": streaming,
            "preserve_thinking": False,
        }
        optional_values = (
            ("temperature", temp),
            ("top_p", topp),
            ("top_k", topk),
            ("max_completion_tokens", max_tokens),
            ("seed", seed),
            ("presence_penalty", presence_penalty),
            ("stop", stop),
            ("logprobs", logprobs),
            ("top_logprobs", top_logprobs),
            ("reasoning_effort", reasoning_effort),
        )
        for key, value in optional_values:
            if value is not None:
                chat_payload[key] = value
        if response_format is not None:
            copied_format = deepcopy(dict(response_format))
            if copied_format not in ({"type": "text"}, {"type": "json_object"}):
                raise _bad_request("Unsupported QwenCloud response format.")
            chat_payload["response_format"] = copied_format
        if chat_tools:
            if n is not None and n != 1:
                raise _bad_request("QwenCloud tool requests require n=1.")
            chat_payload["n"] = 1
            chat_payload["tools"] = chat_tools
            if tool_choice is not None:
                chat_payload["tool_choice"] = deepcopy(tool_choice)
        elif n is not None:
            chat_payload["n"] = n
        if streaming:
            chat_payload["stream_options"] = {"include_usage": True}
        return chat_payload
    if api_mode != "responses":
        raise _bad_request("Unsupported QwenCloud API mode for request translation.")

    payload: dict[str, Any] = {
        "model": model,
        "input": responses_input,
        "stream": streaming,
        "store": False,
    }
    if instructions is not None:
        payload["instructions"] = instructions
    if temp is not None:
        payload["temperature"] = temp
    if topp is not None:
        payload["top_p"] = topp
    if max_tokens is not None:
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens < 16
        ):
            raise _bad_request("QwenCloud max output tokens must be at least 16.")
        payload["max_output_tokens"] = max_tokens
    if reasoning_effort is not None:
        if (
            not isinstance(reasoning_effort, str)
            or reasoning_effort not in _REASONING_EFFORTS
        ):
            raise _bad_request("Unsupported QwenCloud reasoning effort.")
        payload["reasoning"] = {"effort": reasoning_effort}
    if responses_tools:
        payload["tools"] = responses_tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    return payload
