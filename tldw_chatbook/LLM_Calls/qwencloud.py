"""QwenCloud dual-API transport, translation, and response normalization."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal, Never, cast

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError as RequestsConnectionError,
    ContentDecodingError,
    HTTPError,
    InvalidJSONError,
    JSONDecodeError as RequestsJSONDecodeError,
    RequestException,
    Timeout as RequestsTimeout,
)
from urllib3.exceptions import InvalidHeader as Urllib3InvalidHeader
from urllib3.util import Retry

from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_chatbook.LLM_Calls.qwencloud_url import (
    QwenCloudBaseURLValidationError,
    normalize_qwencloud_base_url as _normalize_qwencloud_base_url,
)
from tldw_chatbook.config import (
    ProviderSettingsError,
    get_runtime_config_snapshot,
    provider_settings_for_key,
    resolve_provider_api_key,
)
from tldw_chatbook.Utils.egress import create_default_session
from tldw_chatbook.Utils.sensitive_llm_logging import llm_retry_count

logger = logger.bind(module="qwencloud")

QwenCloudAPIMode = Literal["responses", "chat_completions"]

_DEFAULT_KEY_ENV_VAR = "DASHSCOPE_API_KEY"
_API_MODES: frozenset[str] = frozenset({"responses", "chat_completions"})
_REASONING_EFFORTS: frozenset[str] = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _configuration_error(message: str) -> ChatConfigurationError:
    return ChatConfigurationError(provider="qwencloud", message=message)


def _bad_request(message: str) -> ChatBadRequestError:
    return ChatBadRequestError(provider="qwencloud", message=message)


def _provider_error(message: str, *, status_code: int = 502) -> ChatProviderError:
    return ChatProviderError(
        provider="qwencloud", message=message, status_code=status_code
    )


def _best_effort_close(resource: Any) -> None:
    """Attempt one close without allowing cleanup failures to mask results."""
    try:
        resource.close()
    except Exception:
        pass


def _retry_configuration(
    provider_settings: Mapping[str, Any],
) -> tuple[int, float]:
    configured_retries = provider_settings.get("retries", 3)
    if isinstance(configured_retries, bool):
        raise _configuration_error("QwenCloud retries must be an integer.")
    try:
        retries = max(0, int(configured_retries))
    except (TypeError, ValueError) as exc:
        raise _configuration_error("QwenCloud retries must be an integer.") from exc

    configured_delay = provider_settings.get("retry_delay", 1)
    if isinstance(configured_delay, bool):
        raise _configuration_error("QwenCloud retry delay must be numeric.")
    try:
        retry_delay = float(configured_delay)
    except (TypeError, ValueError) as exc:
        raise _configuration_error("QwenCloud retry delay must be numeric.") from exc
    if not math.isfinite(retry_delay) or retry_delay < 0:
        raise _configuration_error(
            "QwenCloud retry delay must be non-negative and finite."
        )
    return llm_retry_count(retries), retry_delay


class _RetryStatusResponse:
    def __init__(self, response: requests.Response) -> None:
        self.status = int(response.status_code)
        self.headers = response.headers

    def get_redirect_location(self) -> None:
        return None


def _build_retry_policy(*, retries: int, retry_delay: float) -> Retry:
    return Retry(
        total=retries,
        backoff_factor=retry_delay,
        status_forcelist=_RETRYABLE_STATUS_CODES,
        allowed_methods=frozenset({"POST"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )


def _advance_retry_policy(
    retry_policy: Retry,
    *,
    api_url: str,
    response: requests.Response | None = None,
    error: RequestException | None = None,
) -> tuple[Retry, float]:
    retry_after: float | None = None
    if response is not None:
        retry_response = _RetryStatusResponse(response)
        try:
            retry_after = retry_policy.get_retry_after(cast(Any, retry_response))
        except Urllib3InvalidHeader:
            retry_after = None
        next_policy = retry_policy.increment(
            method="POST",
            url=api_url,
            response=cast(Any, retry_response),
        )
    else:
        next_policy = retry_policy.increment(
            method="POST",
            url=api_url,
            error=error,
        )
    delay = retry_after if retry_after is not None else next_policy.get_backoff_time()
    return next_policy, delay


def _transport_without_hidden_retries() -> HTTPAdapter:
    retry_policy = Retry(
        total=0,
        connect=0,
        read=0,
        redirect=0,
        status=0,
        other=0,
        allowed_methods=frozenset({"POST"}),
        raise_on_status=False,
    )
    return HTTPAdapter(max_retries=retry_policy)


def _is_mode_model_mismatch(response: requests.Response) -> bool:
    detail = ""
    try:
        payload = response.json()
    except RequestException:
        return False
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            message = error.get("message")
            if isinstance(message, str):
                detail = message
    if not detail:
        raw_text = getattr(response, "text", "")
        detail = raw_text if isinstance(raw_text, str) else ""
    lowered = detail.lower()
    incompatible = any(
        marker in lowered
        for marker in ("not supported", "unsupported", "incompatible", "not compatible")
    )
    mentions_mode = any(
        marker in lowered
        for marker in ("responses api", "chat completions", "api mode", "api_mode")
    )
    return "model" in lowered and incompatible and mentions_mode


def _raise_qwencloud_http_error(response: requests.Response) -> Never:
    status_code = int(getattr(response, "status_code", 0) or 0)
    logger.error(
        "QwenCloud request failed; status={}; error_type=http_error",
        status_code,
    )
    if status_code in {401, 403}:
        raise ChatAuthenticationError(
            provider="qwencloud",
            message="QwenCloud authentication failed. Check the QwenCloud API key.",
        ) from None
    if status_code == 429:
        raise ChatRateLimitError(
            provider="qwencloud",
            message="QwenCloud rate limit exceeded. Retry after the provider delay.",
        ) from None
    if 400 <= status_code < 500:
        if _is_mode_model_mismatch(response):
            message = (
                "QwenCloud model is not compatible with the selected API mode; "
                "choose a compatible model or switch api_mode."
            )
        else:
            message = f"QwenCloud rejected the request (status {status_code})."
        raise ChatBadRequestError(provider="qwencloud", message=message) from None
    if 500 <= status_code < 600:
        raise _provider_error(
            f"QwenCloud service failed (status {status_code}).",
            status_code=status_code,
        ) from None
    raise _provider_error("QwenCloud returned an unexpected HTTP failure.") from None


def _validate_optional_number(name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _bad_request(f"QwenCloud {name} must be numeric.")
    if isinstance(value, float) and not math.isfinite(value):
        raise _bad_request(f"QwenCloud {name} must be finite.")


def _validate_optional_integer(name: str, value: Any) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise _bad_request(f"QwenCloud {name} must be an integer.")


def _normalize_stop(stop: Any) -> str | list[str] | None:
    if stop is None or isinstance(stop, str):
        return stop
    if not isinstance(stop, Sequence) or isinstance(stop, (str, bytes)):
        raise _bad_request("QwenCloud stop must be a string or sequence of strings.")
    if any(not isinstance(item, str) for item in stop):
        raise _bad_request("QwenCloud stop sequences must contain only strings.")
    return list(stop)


def _validate_scalar_parameters(
    *,
    model: Any,
    streaming: Any,
    temp: Any,
    topp: Any,
    topk: Any,
    max_tokens: Any,
    seed: Any,
    presence_penalty: Any,
    stop: Any,
    n: Any,
    logprobs: Any,
    top_logprobs: Any,
    reasoning_effort: Any,
) -> str | list[str] | None:
    if not isinstance(model, str) or not model.strip():
        raise _bad_request("QwenCloud model must be a non-empty string.")
    if not isinstance(streaming, bool):
        raise _bad_request("QwenCloud streaming must be a boolean.")
    if logprobs is not None and not isinstance(logprobs, bool):
        raise _bad_request("QwenCloud logprobs must be a boolean.")
    if reasoning_effort is not None and not isinstance(reasoning_effort, str):
        raise _bad_request("QwenCloud reasoning effort must be a string.")

    for name, value in (
        ("temperature", temp),
        ("top_p", topp),
        ("presence penalty", presence_penalty),
    ):
        _validate_optional_number(name, value)
    for name, value in (
        ("top_k", topk),
        ("max tokens", max_tokens),
        ("seed", seed),
        ("n", n),
        ("top_logprobs", top_logprobs),
    ):
        _validate_optional_integer(name, value)
    return _normalize_stop(stop)


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is not supported: {value}")


def normalize_qwencloud_api_mode(
    api_mode: object | None,
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
    configured = None
    if api_mode is None:
        if provider_settings is not None and not isinstance(provider_settings, Mapping):
            raise _configuration_error("QwenCloud provider settings must be an object.")
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
    return cast(QwenCloudAPIMode, normalized)


def normalize_qwencloud_base_url(api_base_url: str | None) -> str:
    """Normalize a QwenCloud base URL without constructing an endpoint.

    Args:
        api_base_url: Configured compatible-API base or pasted endpoint.

    Returns:
        A validated base URL without a recognized terminal endpoint suffix.

    Raises:
        ChatConfigurationError: If the URL is unsafe or malformed.
    """
    try:
        return _normalize_qwencloud_base_url(api_base_url)
    except QwenCloudBaseURLValidationError as exc:
        raise _configuration_error(str(exc)) from exc


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
    explicit = resolve_provider_api_key(explicit_api_key)
    if explicit is not None:
        return explicit
    if provider_settings is not None and not isinstance(provider_settings, Mapping):
        raise _configuration_error("QwenCloud provider settings must be an object.")

    settings = provider_settings or {}
    configured = resolve_provider_api_key(settings.get("api_key"))
    if configured is not None:
        return configured

    if environ is not None and not isinstance(environ, Mapping):
        raise _configuration_error(
            "QwenCloud credential environment must be an object."
        )

    env_name = settings.get("api_key_env_var", _DEFAULT_KEY_ENV_VAR)
    if not isinstance(env_name, str) or not env_name.strip():
        env_name = _DEFAULT_KEY_ENV_VAR
    environment = os.environ if environ is None else environ
    from_environment = resolve_provider_api_key(environment.get(env_name.strip()))
    if from_environment is not None:
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
            decoded_arguments = json.loads(
                arguments, parse_constant=_reject_non_finite_json_constant
            )
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
    if not isinstance(messages_payload, Sequence) or isinstance(
        messages_payload, (str, bytes)
    ):
        raise _bad_request("QwenCloud messages must be a list.")
    messages: list[dict[str, Any]] = []
    for message in messages_payload:
        if not isinstance(message, Mapping):
            raise _bad_request("QwenCloud messages must be objects.")
        messages.append(deepcopy(dict(message)))

    leading_system: str | None = None
    if messages and messages[0].get("role") == "system":
        system_row = messages.pop(0)
        if "tool_calls" in system_row:
            raise _bad_request("QwenCloud tool calls require the assistant role.")
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
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        raise _bad_request("QwenCloud function tools must be a list.")

    chat_tools: list[dict[str, Any]] = []
    responses_tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, Mapping) or tool.get("type") != "function":
            raise _bad_request("QwenCloud supports only existing function tools.")
        if set(tool) != {"type", "function"}:
            raise _bad_request("QwenCloud function tool fields are unsupported.")
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

        copied_function = deepcopy(dict(function))
        chat_tools.append({"type": "function", "function": copied_function})
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
    normalized_stop = _validate_scalar_parameters(
        model=model,
        streaming=streaming,
        temp=temp,
        topp=topp,
        topk=topk,
        max_tokens=max_tokens,
        seed=seed,
        presence_penalty=presence_penalty,
        stop=stop,
        n=n,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        reasoning_effort=reasoning_effort,
    )
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
            ("logprobs", logprobs),
            ("top_logprobs", top_logprobs),
            ("reasoning_effort", reasoning_effort),
        )
        for key, value in optional_values:
            if value is not None:
                chat_payload[key] = value
        if normalized_stop is not None:
            chat_payload["stop"] = normalized_stop
        if response_format is not None:
            if not isinstance(response_format, Mapping):
                raise _bad_request("QwenCloud response format must be an object.")
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


def _normalize_response_usage(payload: Mapping[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage")
    if usage is None:
        return {}
    if not isinstance(usage, Mapping):
        raise _provider_error("QwenCloud returned malformed token usage.")
    return deepcopy(dict(usage))


def _normalize_response_tool_call(raw_call: Mapping[str, Any]) -> dict[str, Any]:
    call_id = raw_call.get("call_id")
    name = raw_call.get("name")
    arguments = raw_call.get("arguments")
    if (
        not isinstance(call_id, str)
        or not call_id.strip()
        or not isinstance(name, str)
        or not name.strip()
        or not isinstance(arguments, str)
    ):
        raise _provider_error("QwenCloud returned an incomplete function call.")
    try:
        decoded_arguments = json.loads(
            arguments, parse_constant=_reject_non_finite_json_constant
        )
    except (TypeError, ValueError) as exc:
        raise _provider_error(
            "QwenCloud returned malformed function-call arguments."
        ) from exc
    if not isinstance(decoded_arguments, Mapping):
        raise _provider_error("QwenCloud returned non-object function-call arguments.")
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _normalize_responses_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    status = payload.get("status")
    if status in {"failed", "cancelled"}:
        raise _provider_error("QwenCloud did not complete the response.")
    if status not in {"completed", "incomplete"}:
        raise _provider_error("QwenCloud returned a malformed response status.")

    output = payload.get("output")
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        raise _provider_error("QwenCloud returned a malformed response envelope.")

    text_fragments: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    call_ids: set[str] = set()
    for raw_item in output:
        if not isinstance(raw_item, Mapping):
            raise _provider_error("QwenCloud returned a malformed output item.")
        item_type = raw_item.get("type")
        if item_type == "reasoning":
            continue
        if item_type == "message":
            content = raw_item.get("content")
            if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
                raise _provider_error("QwenCloud returned malformed message content.")
            for raw_part in content:
                if not isinstance(raw_part, Mapping):
                    raise _provider_error(
                        "QwenCloud returned a malformed message content part."
                    )
                part_type = raw_part.get("type")
                if part_type == "output_text":
                    text = raw_part.get("text")
                    if not isinstance(text, str):
                        raise _provider_error(
                            "QwenCloud returned malformed output text."
                        )
                    text_fragments.append(text)
                    continue
                if part_type == "refusal":
                    raise _provider_error("QwenCloud refused the response.")
                raise _provider_error(
                    "QwenCloud returned an unsupported message content part."
                )
            continue
        if item_type == "function_call":
            if status != "completed" or raw_item.get("status") not in {
                None,
                "completed",
            }:
                raise _provider_error("QwenCloud returned an incomplete function call.")
            normalized_call = _normalize_response_tool_call(raw_item)
            call_id = normalized_call["id"]
            if call_id in call_ids:
                raise _provider_error("QwenCloud returned duplicate function-call IDs.")
            call_ids.add(call_id)
            tool_calls.append(normalized_call)
            continue
        raise _provider_error("QwenCloud returned an unsupported output item.")

    content = "".join(text_fragments)
    has_usable_text = bool(content.strip())
    if status == "incomplete":
        incomplete_details = payload.get("incomplete_details")
        reason = (
            incomplete_details.get("reason")
            if isinstance(incomplete_details, Mapping)
            else None
        )
        if reason != "max_output_tokens" or not has_usable_text or tool_calls:
            raise _provider_error("QwenCloud returned an incomplete response.")
        finish_reason = "length"
    elif tool_calls:
        finish_reason = "tool_calls"
    else:
        finish_reason = "stop"

    if not has_usable_text and not tool_calls:
        raise _provider_error("QwenCloud returned no usable response content.")
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content if has_usable_text else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": _normalize_response_usage(payload),
    }


def _normalize_chat_completions_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    choices = payload.get("choices")
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes))
        or not choices
        or not isinstance(choices[0], Mapping)
    ):
        raise _provider_error("QwenCloud returned a malformed choices envelope.")
    choice = choices[0]
    raw_message = choice.get("message")
    if not isinstance(raw_message, Mapping):
        raise _provider_error("QwenCloud returned a malformed assistant message.")

    content = raw_message.get("content")
    if content is not None and not isinstance(content, str):
        raise _provider_error("QwenCloud returned malformed assistant text.")
    raw_calls = raw_message.get("tool_calls")
    tool_calls: list[dict[str, Any]] = []
    call_ids: set[str] = set()
    if raw_calls is not None:
        if (
            not isinstance(raw_calls, Sequence)
            or isinstance(raw_calls, (str, bytes))
            or not raw_calls
        ):
            raise _provider_error("QwenCloud returned malformed function calls.")
        for raw_call in raw_calls:
            if (
                not isinstance(raw_call, Mapping)
                or raw_call.get("type") != "function"
                or not isinstance(raw_call.get("function"), Mapping)
            ):
                raise _provider_error("QwenCloud returned a malformed function call.")
            function = raw_call["function"]
            normalized_call = _normalize_response_tool_call(
                {
                    "call_id": raw_call.get("id"),
                    "name": function.get("name"),
                    "arguments": function.get("arguments"),
                }
            )
            call_id = normalized_call["id"]
            if call_id in call_ids:
                raise _provider_error("QwenCloud returned duplicate function-call IDs.")
            call_ids.add(call_id)
            tool_calls.append(normalized_call)

    has_usable_text = isinstance(content, str) and bool(content.strip())
    if not has_usable_text and not tool_calls:
        raise _provider_error("QwenCloud returned no usable response content.")
    raw_finish_reason = choice.get("finish_reason")
    if not isinstance(raw_finish_reason, str) or raw_finish_reason not in {
        "stop",
        "length",
        "tool_calls",
    }:
        raise _provider_error("QwenCloud returned an invalid completion finish reason.")
    if bool(tool_calls) != (raw_finish_reason == "tool_calls"):
        raise _provider_error(
            "QwenCloud returned an inconsistent completion finish reason."
        )
    finish_reason = raw_finish_reason

    message: dict[str, Any] = {
        "role": "assistant",
        "content": content if has_usable_text else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": _normalize_response_usage(payload),
    }


def normalize_qwencloud_response(
    payload: Mapping[str, Any], *, api_mode: QwenCloudAPIMode
) -> dict[str, Any]:
    """Normalize a successful QwenCloud response to Chatbook's chat contract.

    Args:
        payload: Decoded provider JSON response.
        api_mode: The API mode used for the request.

    Returns:
        A standard choices/message/finish/usage mapping.

    Raises:
        ChatProviderError: If the successful HTTP response is malformed or empty.
    """
    if not isinstance(payload, Mapping):
        raise _provider_error("QwenCloud response envelope must be an object.")
    if api_mode == "responses":
        return _normalize_responses_payload(payload)
    if api_mode == "chat_completions":
        return _normalize_chat_completions_payload(payload)
    raise _provider_error("QwenCloud response used an unknown API mode.")


def chat_with_qwencloud(
    input_data: list[dict[str, Any]],
    model: str | None = None,
    api_key: str | None = None,
    system_message: str | None = None,
    temp: float | None = None,
    streaming: bool | None = False,
    topp: float | None = None,
    topk: int | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    stop: str | list[str] | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    presence_penalty: float | None = None,
    response_format: dict[str, str] | None = None,
    n: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    api_base_url: str | None = None,
    api_mode: str | None = None,
) -> dict[str, Any] | Iterator[dict[str, Any]]:
    """Send a QwenCloud request through the selected compatible API mode.

    Transport timeout and retry policy are owned by QwenCloud's modern provider
    settings rather than the generic dispatcher.
    """
    config_values = get_runtime_config_snapshot().values
    api_settings = config_values.get("api_settings", {})
    if not isinstance(api_settings, Mapping):
        raise _configuration_error("QwenCloud API settings must be an object.")
    try:
        provider_settings = cast(
            Mapping[str, Any], provider_settings_for_key(api_settings, "qwencloud")
        )
    except ProviderSettingsError as exc:
        raise _configuration_error(
            "QwenCloud provider settings must be a configuration table."
        ) from exc

    final_mode = normalize_qwencloud_api_mode(
        api_mode, provider_settings=provider_settings
    )
    configured_base = provider_settings.get("api_base_url")
    final_base = normalize_qwencloud_base_url(
        api_base_url if api_base_url is not None else configured_base
    )
    final_api_key = resolve_qwencloud_api_key(
        api_key, provider_settings=provider_settings
    )
    configured_model = provider_settings.get("model")
    final_model = (
        model
        if model is not None
        else configured_model
        if isinstance(configured_model, str)
        else "qwen3.8-max"
    )
    final_streaming = False if streaming is None else streaming

    payload = build_qwencloud_payload(
        api_mode=final_mode,
        model=final_model,
        system_message=system_message,
        messages_payload=input_data,
        streaming=final_streaming,
        tools=tools,
        tool_choice=tool_choice,
        temp=temp,
        topp=topp,
        topk=topk,
        max_tokens=max_tokens,
        seed=seed,
        stop=stop,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        presence_penalty=presence_penalty,
        response_format=response_format,
        n=n,
        reasoning_effort=reasoning_effort,
    )
    timeout_value = provider_settings.get("timeout", 120)
    try:
        timeout = float(timeout_value)
    except (TypeError, ValueError) as exc:
        raise _configuration_error("QwenCloud timeout must be numeric.") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise _configuration_error("QwenCloud timeout must be positive and finite.")
    retries, retry_delay = _retry_configuration(provider_settings)
    retry_policy = _build_retry_policy(retries=retries, retry_delay=retry_delay)
    adapter = _transport_without_hidden_retries()

    suffix = "/responses" if final_mode == "responses" else "/chat/completions"
    api_url = f"{final_base}{suffix}"
    headers = {
        "Authorization": f"Bearer {final_api_key}",
        "Content-Type": "application/json",
    }

    session = create_default_session()
    stream_owns_session = False
    try:
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        for attempt_index in range(retries + 1):
            response: requests.Response | None = None
            retry_sleep: float | None = None
            try:
                response = session.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                    stream=True,
                )
                status_code = int(response.status_code)
                if status_code in _RETRYABLE_STATUS_CODES and attempt_index < retries:
                    retry_policy, retry_sleep = _advance_retry_policy(
                        retry_policy,
                        api_url=api_url,
                        response=response,
                    )
                else:
                    response.raise_for_status()
                    if final_streaming:
                        from tldw_chatbook.LLM_Calls.qwencloud_streaming import (
                            QwenCloudStream,
                        )

                        stream = QwenCloudStream(
                            response=response,
                            session=session,
                            api_mode=final_mode,
                        )
                        response = None
                        stream_owns_session = True
                        return stream
                    result = response.json()
                    if not isinstance(result, Mapping):
                        raise _provider_error(
                            "QwenCloud response envelope must be an object."
                        )
                    return normalize_qwencloud_response(result, api_mode=final_mode)
            except HTTPError as exc:
                failed_response = exc.response if exc.response is not None else response
                if failed_response is None:
                    logger.error(
                        "QwenCloud request failed; "
                        "status=unknown; error_type=http_error"
                    )
                    raise _provider_error(
                        "QwenCloud returned an HTTP failure without a response."
                    ) from None
                _raise_qwencloud_http_error(failed_response)
            except (
                RequestsJSONDecodeError,
                InvalidJSONError,
                ContentDecodingError,
            ):
                logger.error(
                    "QwenCloud request failed; status={}; "
                    "error_type=malformed_response",
                    getattr(response, "status_code", "unknown"),
                )
                raise _provider_error(
                    "QwenCloud returned malformed provider JSON or content."
                ) from None
            except ChunkedEncodingError as exc:
                if attempt_index < retries:
                    retry_policy, retry_sleep = _advance_retry_policy(
                        retry_policy,
                        api_url=api_url,
                        error=exc,
                    )
                else:
                    logger.error(
                        "QwenCloud request failed; "
                        "status=none; error_type=incomplete_body"
                    )
                    raise _provider_error(
                        "QwenCloud network response was incomplete."
                    ) from None
            except (RequestsConnectionError, RequestsTimeout) as exc:
                if attempt_index < retries:
                    retry_policy, retry_sleep = _advance_retry_policy(
                        retry_policy,
                        api_url=api_url,
                        error=exc,
                    )
                else:
                    logger.error(
                        "QwenCloud request failed; status=none; error_type={}",
                        type(exc).__name__,
                    )
                    raise _provider_error(
                        "QwenCloud network request failed.",
                        status_code=504 if isinstance(exc, RequestsTimeout) else 502,
                    ) from None
            except RequestException as exc:
                logger.error(
                    "QwenCloud request failed; status=none; error_type={}",
                    type(exc).__name__,
                )
                raise _provider_error("QwenCloud network request failed.") from None
            except (TypeError, ValueError) as exc:
                logger.error(
                    "QwenCloud request failed; status={}; error_type={}",
                    getattr(response, "status_code", "unknown"),
                    type(exc).__name__,
                )
                raise _provider_error("QwenCloud returned malformed JSON.") from None
            finally:
                if response is not None:
                    _best_effort_close(response)

            if retry_sleep is None:
                raise _provider_error("QwenCloud retry state was incomplete.")
            if retry_sleep > 0:
                time.sleep(retry_sleep)
    finally:
        if not stream_owns_session:
            _best_effort_close(session)

    raise _provider_error("QwenCloud request attempts were exhausted.")
