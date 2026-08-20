"""Strict Moonshot/Kimi Chat-Completions adapter policy."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast

from tldw_chatbook.Chat.Chat_Deps import (
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
)
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRound,
    ContinuationRestoreTarget,
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
    parse_provider_continuation_json,
    validate_continuation_restore,
)
from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedChatProtocolError,
    HostedChatStream,
    HostedChatTurn,
    HostedHTTPTransportConfig,
    hosted_chat_request,
    normalize_hosted_chat_base_url,
)
from tldw_chatbook.config import (
    ProviderSettingsError,
    get_runtime_config_snapshot,
    provider_settings_for_key,
    resolve_provider_api_key,
)
from tldw_chatbook.model_capabilities import (
    moonshot_model_rejects_sampling_params,
    moonshot_model_requires_min_temperature_for_multiple_choices,
    moonshot_model_supports_reasoning_effort,
)


_DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
_CHINA_BASE_URL = "https://api.moonshot.cn/v1"
_DEFAULT_MODEL = "kimi-k3"
_FUNCTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{2,63}$")
# "medium" probe-verified accepted on kimi-k3 and kimi-k2.6 (TASK-18803,
# chatcmpl-6a872b62bea2d202c1d3f6fa / chatcmpl-6a872affa508b3005f3a5c0b).
_KIMI_REASONING_EFFORTS = frozenset({"low", "medium", "high", "max"})
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 50_000
_MAX_JSON_STRING_CHARS = 16 * 1024 * 1024


@dataclass(frozen=True)
class MoonshotResolution:
    """Immutable resolved Moonshot request identity and transport policy."""

    provider: str
    model: str
    api_key: str = field(repr=False, compare=False)
    base_url: str
    timeout: float
    retries: int
    retry_delay: float
    streaming: bool


class MoonshotFinishPolicy:
    """Validate Moonshot finish and reasoning fields."""

    def validate_finish(
        self,
        *,
        finish_reason: object,
        has_text: bool,
        has_calls: bool,
    ) -> str:
        if finish_reason not in {"stop", "tool_calls", "length"}:
            raise HostedChatProtocolError("Moonshot finish state is malformed.")
        if finish_reason == "tool_calls":
            if not has_calls:
                raise HostedChatProtocolError("Moonshot finish state is inconsistent.")
        elif has_calls or not has_text:
            raise HostedChatProtocolError("Moonshot finish state is inconsistent.")
        return cast(str, finish_reason)

    def validate_reasoning_content(self, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise HostedChatProtocolError("Moonshot reasoning content is malformed.")
        return value


class MoonshotStream(Iterator[dict[str, Any]]):
    """Expose visible Moonshot chunks while retaining private terminal reasoning."""

    def __init__(
        self,
        stream: HostedChatStream,
        *,
        resolution: MoonshotResolution | None = None,
        provider_continuations: Sequence[ProviderContinuationCheckpoint] = (),
    ) -> None:
        self._stream = stream
        self._resolution = resolution
        self._provider_continuations = tuple(provider_continuations)

    def __iter__(self) -> MoonshotStream:
        return self

    def __next__(self) -> dict[str, Any]:
        event = deepcopy(next(self._stream))
        for choice in event.get("choices", ()):
            if isinstance(choice, dict) and isinstance(choice.get("delta"), dict):
                choice["delta"].pop("reasoning_content", None)
        return event

    @property
    def terminal_turn(self) -> HostedChatTurn:
        """Return private terminal state after clean stream exhaustion."""
        return self._stream.terminal_turn

    @property
    def provider_continuation(self) -> ProviderContinuationCheckpoint | None:
        """Return a canonical candidate only after clean stream exhaustion."""
        if self._resolution is None:
            raise HostedChatProtocolError("Moonshot stream metadata is incomplete.")
        return _moonshot_continuation_candidate(
            self.terminal_turn,
            resolution=self._resolution,
            provider_continuations=self._provider_continuations,
        )

    def close(self) -> None:
        """Close the owned response/session pair."""
        self._stream.close()


class MoonshotResponse(dict[str, Any]):
    """Public response mapping with private terminal state kept out of repr."""

    def __init__(
        self,
        value: Mapping[str, Any],
        *,
        terminal_turn: HostedChatTurn,
        provider_continuation: ProviderContinuationCheckpoint | None,
    ) -> None:
        super().__init__(value)
        self._terminal_turn = terminal_turn
        self._provider_continuation = provider_continuation

    @property
    def terminal_turn(self) -> HostedChatTurn:
        return self._terminal_turn

    @property
    def provider_continuation(self) -> ProviderContinuationCheckpoint | None:
        return self._provider_continuation


_FINISH_POLICY = MoonshotFinishPolicy()


def chat_with_moonshot(
    input_data: list[dict[str, Any]],
    model: str | None = None,
    api_key: str | None = None,
    system_message: str | None = None,
    temp: float | None = None,
    maxp: float | None = None,
    streaming: bool | None = False,
    frequency_penalty: float | None = None,
    max_tokens: int | None = None,
    n: int | None = None,
    presence_penalty: float | None = None,
    response_format: dict[str, str] | None = None,
    seed: int | None = None,
    stop: str | list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    user: str | None = None,
    custom_prompt_arg: str | None = None,
    api_base_url: str | None = None,
    reasoning_effort: str | None = None,
    provider_continuations: Sequence[ProviderContinuationCheckpoint] = (),
    request_timeout: float | None = None,
    request_retries: int | None = None,
    request_retry_delay: float | None = None,
) -> dict[str, Any] | MoonshotStream:
    """Send one Moonshot Chat-Completions request through the hosted boundary."""
    del custom_prompt_arg
    resolution = resolve_moonshot_request(
        explicit_api_key=api_key,
        explicit_base_url=api_base_url,
        explicit_model=model,
        explicit_timeout=request_timeout,
        explicit_retries=request_retries,
        explicit_retry_delay=request_retry_delay,
    )
    payload = build_moonshot_chat_payload(
        resolution=resolution,
        messages_payload=input_data,
        system_message=system_message,
        streaming=streaming,
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=reasoning_effort,
        provider_continuations=provider_continuations,
        max_tokens=max_tokens,
        stop=stop,
        response_format=response_format,
        temperature=temp,
        top_p=maxp,
        n=n,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        seed=seed,
        user=user,
    )
    try:
        result = hosted_chat_request(
            config=HostedHTTPTransportConfig(
                provider="moonshot",
                base_url=resolution.base_url,
                api_key=resolution.api_key,
                timeout=resolution.timeout,
                retries=resolution.retries,
                retry_delay=resolution.retry_delay,
            ),
            payload=payload,
            streaming=cast(bool, payload["stream"]),
            finish_policy=_FINISH_POLICY,
        )
    except HostedChatProtocolError:
        raise ChatProviderError(
            provider="moonshot",
            message="Moonshot returned a malformed successful response.",
            status_code=502,
        ) from None
    if isinstance(result, HostedChatStream):
        return MoonshotStream(
            result,
            resolution=resolution,
            provider_continuations=provider_continuations,
        )
    return _moonshot_turn_response(
        result,
        resolution=resolution,
        provider_continuations=provider_continuations,
    )


def _moonshot_turn_response(
    turn: HostedChatTurn,
    *,
    resolution: MoonshotResolution,
    provider_continuations: Sequence[ProviderContinuationCheckpoint],
) -> MoonshotResponse:
    message = deepcopy(turn.assistant_message)
    if message is None:
        raise HostedChatProtocolError("Moonshot response message is incomplete.")
    message.pop("reasoning_content", None)
    response: dict[str, Any] = {
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": turn.finish_reason,
            }
        ]
    }
    if turn.usage is not None:
        response["usage"] = deepcopy(turn.usage)
    return MoonshotResponse(
        response,
        terminal_turn=turn,
        provider_continuation=_moonshot_continuation_candidate(
            turn,
            resolution=resolution,
            provider_continuations=provider_continuations,
        ),
    )


def _moonshot_continuation_candidate(
    turn: HostedChatTurn,
    *,
    resolution: MoonshotResolution,
    provider_continuations: Sequence[ProviderContinuationCheckpoint],
) -> ProviderContinuationCheckpoint | None:
    active = tuple(
        checkpoint
        for checkpoint in provider_continuations
        if checkpoint.state == "active"
    )
    if len(active) > 1:
        raise HostedChatProtocolError("Moonshot continuation state is ambiguous.")
    current = active[0] if active else None
    if turn.tool_calls:
        round_ = ContinuationRound(
            assistant_content=turn.text,
            reasoning_blocks=(turn.reasoning_content,)
            if turn.reasoning_content is not None
            else (),
            calls=tuple(
                ContinuationCall(
                    call_id=cast(str, call["id"]),
                    name=cast(str, call["function"]["name"]),
                    arguments=cast(str, call["function"]["arguments"]),
                    state="pending",
                )
                for call in turn.tool_calls
            ),
        )
        candidate = ProviderContinuationCheckpoint(
            schema_version=1,
            checkpoint_revision=(current.checkpoint_revision + 1 if current else 1),
            provider="moonshot",
            protocol="chat_completions",
            model=resolution.model,
            api_base_url=resolution.base_url,
            state="active",
            rounds=((*current.rounds, round_) if current else (round_,)),
        )
    elif current is not None:
        rounds = current.rounds
        if resolution.model == _DEFAULT_MODEL:
            rounds = (
                *rounds,
                ContinuationRound(
                    assistant_content=turn.text,
                    reasoning_blocks=(turn.reasoning_content or "",),
                    calls=(),
                ),
            )
        candidate = ProviderContinuationCheckpoint(
            schema_version=1,
            checkpoint_revision=current.checkpoint_revision + 1,
            provider="moonshot",
            protocol="chat_completions",
            model=resolution.model,
            api_base_url=resolution.base_url,
            state="complete",
            rounds=rounds,
        )
    elif resolution.model == _DEFAULT_MODEL and turn.reasoning_content:
        candidate = ProviderContinuationCheckpoint(
            schema_version=1,
            checkpoint_revision=1,
            provider="moonshot",
            protocol="chat_completions",
            model=resolution.model,
            api_base_url=resolution.base_url,
            state="complete",
            rounds=(
                ContinuationRound(
                    assistant_content=turn.text,
                    reasoning_blocks=(turn.reasoning_content,),
                    calls=(),
                ),
            ),
        )
    else:
        return None
    return parse_provider_continuation_json(dump_provider_continuation_json(candidate))


def _configuration_error(message: str) -> ChatConfigurationError:
    return ChatConfigurationError(provider="moonshot", message=message)


def _bad_request(message: str) -> ChatBadRequestError:
    return ChatBadRequestError(provider="moonshot", message=message)


def resolve_moonshot_request(
    *,
    explicit_api_key: object = None,
    explicit_base_url: object = None,
    explicit_model: object = None,
    explicit_timeout: object = None,
    explicit_retries: object = None,
    explicit_retry_delay: object = None,
    app_config: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> MoonshotResolution:
    """Resolve one Moonshot request from canonical immutable sources.

    Args:
        explicit_api_key: Trusted call-supplied credential override.
        explicit_base_url: Trusted call-supplied endpoint override.
        explicit_model: Trusted call-supplied model override.
        app_config: Loaded application configuration; runtime snapshot when absent.
        environ: Environment mapping; ``os.environ`` when absent.

    Returns:
        A frozen Moonshot resolution.

    Raises:
        ChatConfigurationError: If any authoritative source is malformed or a
            usable credential cannot be resolved.
    """
    config: object = (
        get_runtime_config_snapshot().values if app_config is None else app_config
    )
    if not isinstance(config, Mapping):
        raise _configuration_error("Moonshot application configuration is invalid.")
    api_settings = config.get("api_settings", {})
    if not isinstance(api_settings, Mapping):
        raise _configuration_error(
            "Moonshot api_settings must be a configuration table."
        )
    try:
        settings = provider_settings_for_key(api_settings, "moonshot")
    except ProviderSettingsError:
        raise _configuration_error(
            "Moonshot api_settings.moonshot must be one unambiguous configuration table."
        ) from None
    transport_settings = dict(settings)
    if explicit_timeout is not None:
        transport_settings["timeout"] = explicit_timeout
    if explicit_retries is not None:
        transport_settings["retries"] = explicit_retries
    if explicit_retry_delay is not None:
        transport_settings["retry_delay"] = explicit_retry_delay
    environment = os.environ if environ is None else environ

    api_key = _resolve_api_key(explicit_api_key, settings, environment)
    model = _resolve_nonblank_string(
        explicit_model,
        settings,
        "model",
        _DEFAULT_MODEL,
        "model",
    )
    base_url = _resolve_base_url(explicit_base_url, settings)
    timeout = _positive_number(transport_settings, "timeout", 90.0)
    retries = _nonnegative_integer(transport_settings, "retries", 3)
    retry_delay = _nonnegative_number(transport_settings, "retry_delay", 1.0)
    streaming = settings.get("streaming", True)
    if type(streaming) is not bool:
        raise _configuration_error("Moonshot streaming must be a boolean.")
    return MoonshotResolution(
        provider="moonshot",
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        streaming=streaming,
    )


def build_moonshot_chat_payload(
    *,
    resolution: MoonshotResolution,
    messages_payload: Sequence[Mapping[str, Any]],
    system_message: object = None,
    streaming: object = None,
    tools: object = None,
    tool_choice: object = None,
    reasoning_effort: object = None,
    provider_continuations: Sequence[ProviderContinuationCheckpoint] = (),
    max_tokens: object = None,
    stop: object = None,
    response_format: object = None,
    temperature: object = None,
    top_p: object = None,
    n: object = None,
    presence_penalty: object = None,
    frequency_penalty: object = None,
    seed: object = None,
    user: object = None,
    **_generic: object,
) -> dict[str, Any]:
    """Build one validated Moonshot request without mutating inputs."""
    if type(resolution) is not MoonshotResolution or resolution.provider != "moonshot":
        raise _bad_request("Moonshot resolution is invalid.")
    stream = resolution.streaming if streaming is None else streaming
    if type(stream) is not bool:
        raise _bad_request("Moonshot streaming must be a boolean.")

    messages = _normalize_messages(messages_payload, system_message=system_message)
    _apply_continuations(messages, provider_continuations, resolution)
    validated_tools = _normalize_tools(tools)
    validated_choice = _normalize_tool_choice(tool_choice, validated_tools)
    _validate_optional_generic_values(
        temperature=temperature,
        top_p=top_p,
        n=n,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        seed=seed,
        user=user,
    )

    payload: dict[str, Any] = {
        "model": resolution.model,
        "messages": messages,
        "stream": stream,
    }
    if max_tokens is not None:
        payload["max_completion_tokens"] = _positive_integer("max_tokens", max_tokens)
    if stop is not None:
        payload["stop"] = _normalize_stop(stop)
    if response_format is not None:
        payload["response_format"] = _normalize_response_format(response_format)
    if validated_tools is not None:
        payload["tools"] = validated_tools
    if validated_choice is not None:
        payload["tool_choice"] = validated_choice

    # Per-model request capabilities come from `model_capabilities`
    # predicates, not model-name checks kept here, so a new release in a
    # covered family never needs a marker edited (TASK-18803; probe evidence
    # in the predicate docstrings).
    if reasoning_effort is not None:
        if not moonshot_model_supports_reasoning_effort(resolution.model):
            raise _bad_request(
                "Moonshot reasoning effort is unsupported for this model."
            )
        if (
            not isinstance(reasoning_effort, str)
            or reasoning_effort not in _KIMI_REASONING_EFFORTS
        ):
            raise _bad_request("Moonshot Kimi reasoning effort is invalid.")
        payload["reasoning_effort"] = reasoning_effort

    if not moonshot_model_rejects_sampling_params(resolution.model):
        # The versioned kimi reasoning family answers a value-level 400 for
        # non-default sampling values, so those models keep the historical
        # omission; everything else (legacy moonshot-v1, kimi-latest, unknown
        # future families) passes the caller's values through for the server
        # to adjudicate.
        if moonshot_model_requires_min_temperature_for_multiple_choices(
            resolution.model
        ):
            legacy_n = cast(int | None, n)
            legacy_temperature = cast(int | float | None, temperature)
            if (
                legacy_n is not None
                and legacy_n > 1
                and (legacy_temperature is None or legacy_temperature < 0.3)
            ):
                raise _bad_request(
                    "Moonshot multiple choices require temperature of at least 0.3."
                )
        for key, value in (
            ("temperature", temperature),
            ("top_p", top_p),
            ("n", n),
            ("presence_penalty", presence_penalty),
            ("frequency_penalty", frequency_penalty),
        ):
            if value is not None:
                payload[key] = value
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _resolve_api_key(
    explicit: object,
    settings: Mapping[str, object],
    environ: Mapping[str, str],
) -> str:
    if explicit is not None:
        resolved = resolve_provider_api_key(explicit)
        if resolved is None:
            raise _configuration_error("Moonshot explicit API key is invalid.")
        return resolved
    if "api_key" in settings:
        resolved = resolve_provider_api_key(settings.get("api_key"))
        if resolved is None:
            raise _configuration_error(
                "Moonshot api_settings.moonshot.api_key is invalid."
            )
        return resolved
    env_name = settings.get("api_key_env_var", "MOONSHOT_API_KEY")
    if not isinstance(env_name, str) or not env_name.strip():
        raise _configuration_error(
            "Moonshot api_settings.moonshot.api_key_env_var is invalid."
        )
    for candidate_name in dict.fromkeys((env_name.strip(), "MOONSHOT_API_KEY")):
        resolved = resolve_provider_api_key(environ.get(candidate_name))
        if resolved is not None:
            return resolved
    raise _configuration_error("Moonshot API key is required.")


def _resolve_nonblank_string(
    explicit: object,
    settings: Mapping[str, object],
    field_name: str,
    default: str,
    label: str,
) -> str:
    value = explicit if explicit is not None else settings.get(field_name, default)
    if not isinstance(value, str) or not value.strip():
        raise _configuration_error(f"Moonshot {label} is invalid.")
    return value.strip()


def _resolve_base_url(
    explicit: object,
    settings: Mapping[str, object],
) -> str:
    if explicit is not None:
        candidate = explicit
    elif "api_base_url" in settings:
        candidate = settings.get("api_base_url")
    else:
        region = settings.get("api_region", "international")
        if not isinstance(region, str) or region not in {"international", "china"}:
            raise _configuration_error("Moonshot API region is invalid.")
        candidate = _CHINA_BASE_URL if region == "china" else _DEFAULT_BASE_URL
    try:
        return normalize_hosted_chat_base_url(candidate, default=_DEFAULT_BASE_URL)
    except ValueError:
        raise _configuration_error("Moonshot API base URL is invalid.") from None


def _positive_number(
    settings: Mapping[str, object], name: str, default: float
) -> float:
    value = settings.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _configuration_error(f"Moonshot {name} must be numeric.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise _configuration_error(f"Moonshot {name} must be positive and finite.")
    return normalized


def _nonnegative_number(
    settings: Mapping[str, object], name: str, default: float
) -> float:
    value = settings.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _configuration_error(f"Moonshot {name} must be numeric.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise _configuration_error(f"Moonshot {name} must be non-negative.")
    return normalized


def _nonnegative_integer(
    settings: Mapping[str, object], name: str, default: int
) -> int:
    value = settings.get(name, default)
    if type(value) is not int or value < 0:
        raise _configuration_error(f"Moonshot {name} must be a non-negative integer.")
    return value


def _normalize_messages(
    value: object,
    *,
    system_message: object,
) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _bad_request("Moonshot messages must be a sequence.")
    result: list[dict[str, Any]] = []
    has_system = any(
        isinstance(message, Mapping) and message.get("role") == "system"
        for message in value
    )
    if system_message is not None:
        if not isinstance(system_message, str) or has_system:
            raise _bad_request("Moonshot system message ownership is invalid.")
        result.append({"role": "system", "content": system_message})

    call_ids: set[str] = set()
    pending_ids: list[str] = []
    for raw_message in value:
        if not isinstance(raw_message, Mapping):
            raise _bad_request("Moonshot message is malformed.")
        role = raw_message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise _bad_request("Moonshot message role is invalid.")
        if role == "tool":
            if not pending_ids:
                raise _bad_request("Moonshot tool result is orphaned.")
            if set(raw_message) != {"role", "tool_call_id", "content"}:
                raise _bad_request("Moonshot tool result is malformed.")
            call_id = raw_message.get("tool_call_id")
            content = raw_message.get("content")
            if call_id != pending_ids[0] or not isinstance(content, str):
                raise _bad_request("Moonshot tool result ordering is invalid.")
            pending_ids.pop(0)
            result.append(deepcopy(dict(raw_message)))
            continue
        if pending_ids:
            raise _bad_request("Moonshot tool call batch is incomplete.")
        allowed = {"role", "content"} | (
            {"tool_calls"} if role == "assistant" else set()
        )
        if set(raw_message) - allowed:
            raise _bad_request("Moonshot message fields are unsupported.")
        content = raw_message.get("content")
        if role == "assistant":
            if content is not None and not isinstance(content, str):
                raise _bad_request("Moonshot assistant content is invalid.")
        elif not isinstance(content, str):
            raise _bad_request("Moonshot message content is invalid.")
        safe = {"role": role, "content": "" if content is None else content}
        if role == "assistant" and "tool_calls" in raw_message:
            calls = _normalize_call_batch(raw_message.get("tool_calls"), call_ids)
            safe["tool_calls"] = list(calls)
            pending_ids = [cast(str, call["id"]) for call in calls]
        result.append(deepcopy(safe))
    if pending_ids:
        raise _bad_request("Moonshot tool call batch is incomplete.")
    return result


def _normalize_call_batch(
    value: object,
    prior_ids: set[str],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise _bad_request("Moonshot tool call batch is invalid.")
    calls: list[dict[str, Any]] = []
    for raw_call in value:
        if not isinstance(raw_call, Mapping) or set(raw_call) != {
            "id",
            "type",
            "function",
        }:
            raise _bad_request("Moonshot tool call is malformed.")
        call_id = raw_call.get("id")
        function = raw_call.get("function")
        if (
            not isinstance(call_id, str)
            or not call_id
            or call_id in prior_ids
            or raw_call.get("type") != "function"
            or not isinstance(function, Mapping)
            or set(function) != {"name", "arguments"}
            or not isinstance(function.get("name"), str)
            or not _FUNCTION_NAME.fullmatch(cast(str, function.get("name")))
            or not isinstance(function.get("arguments"), str)
        ):
            raise _bad_request("Moonshot tool call is malformed.")
        try:
            arguments = json.loads(cast(str, function.get("arguments")))
        except (TypeError, ValueError):
            raise _bad_request("Moonshot tool call arguments are invalid.") from None
        if not isinstance(arguments, dict) or not _json_shape_is_bounded(arguments):
            raise _bad_request("Moonshot tool call arguments are invalid.")
        prior_ids.add(call_id)
        calls.append(deepcopy(dict(raw_call)))
    return tuple(calls)


def _normalize_tools(value: object) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise _bad_request("Moonshot tools are malformed.")
    names: set[str] = set()
    result: list[dict[str, Any]] = []
    for raw_tool in value:
        if not isinstance(raw_tool, Mapping) or set(raw_tool) != {"type", "function"}:
            raise _bad_request("Moonshot supports function tools only.")
        function = raw_tool.get("function")
        if raw_tool.get("type") != "function" or not isinstance(function, Mapping):
            raise _bad_request("Moonshot supports function tools only.")
        if set(function) != {"name", "description", "parameters"}:
            raise _bad_request("Moonshot function tool is malformed.")
        name = function.get("name")
        description = function.get("description")
        parameters = function.get("parameters")
        if (
            not isinstance(name, str)
            or not _FUNCTION_NAME.fullmatch(name)
            or name in names
            or not isinstance(description, str)
            or not description.strip()
            or not isinstance(parameters, Mapping)
            or parameters.get("type") != "object"
            or not _json_shape_is_bounded(parameters)
        ):
            raise _bad_request("Moonshot function tool is malformed.")
        names.add(name)
        result.append(deepcopy(dict(raw_tool)))
    return result


def _normalize_tool_choice(
    value: object,
    tools: Sequence[Mapping[str, Any]] | None,
) -> object:
    if value is None:
        return None
    if isinstance(value, str) and value in {"auto", "none", "required"}:
        if value != "none" and tools is None:
            raise _bad_request("Moonshot tool choice requires tools.")
        return value
    if isinstance(value, Mapping) and set(value) == {"type", "function"}:
        function = value.get("function")
        if (
            value.get("type") == "function"
            and isinstance(function, Mapping)
            and set(function) == {"name"}
            and isinstance(function.get("name"), str)
            and tools is not None
            and function.get("name")
            in {cast(Mapping[str, Any], tool["function"])["name"] for tool in tools}
        ):
            return deepcopy(dict(value))
    raise _bad_request("Moonshot tool choice is unsupported.")


def _validate_optional_generic_values(**values: object) -> None:
    for name in ("temperature", "top_p"):
        value = values[name]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 <= float(value) <= 1
        ):
            raise _bad_request(f"Moonshot {name} is invalid.")
    n = values["n"]
    if n is not None and (type(n) is not int or n <= 0):
        raise _bad_request("Moonshot n is invalid.")
    for name in ("presence_penalty", "frequency_penalty"):
        value = values[name]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not -2 <= float(value) <= 2
        ):
            raise _bad_request(f"Moonshot {name} is invalid.")
    seed = values["seed"]
    if seed is not None and type(seed) is not int:
        raise _bad_request("Moonshot seed is invalid.")
    user = values["user"]
    if user is not None and (not isinstance(user, str) or not user.strip()):
        raise _bad_request("Moonshot user is invalid.")


def _positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise _bad_request(f"Moonshot {name} is invalid.")
    return value


def _normalize_stop(value: object) -> object:
    if isinstance(value, str) and value:
        return value
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and 1 <= len(value) <= 4
        and all(isinstance(item, str) and item for item in value)
    ):
        return list(value)
    raise _bad_request("Moonshot stop is invalid.")


def _normalize_response_format(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not _json_shape_is_bounded(value):
        raise _bad_request("Moonshot response format is invalid.")
    format_type = value.get("type")
    if format_type in {"text", "json_object"}:
        if set(value) != {"type"}:
            raise _bad_request("Moonshot response format is invalid.")
    elif format_type == "json_schema":
        if set(value) != {"type", "json_schema"} or not isinstance(
            value.get("json_schema"), Mapping
        ):
            raise _bad_request("Moonshot response format is invalid.")
    else:
        raise _bad_request("Moonshot response format is invalid.")
    return deepcopy(dict(value))


def _json_shape_is_bounded(value: object) -> bool:
    stack: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            return False
        if current is None or type(current) in {bool, int}:
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                return False
            continue
        if isinstance(current, str):
            if len(current) > _MAX_JSON_STRING_CHARS:
                return False
            continue
        if isinstance(current, Mapping):
            for key, item in current.items():
                if not isinstance(key, str) or len(key) > _MAX_JSON_STRING_CHARS:
                    return False
                stack.append((item, depth + 1))
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            stack.extend((item, depth + 1) for item in current)
            continue
        return False
    return True


def _apply_continuations(
    messages: list[dict[str, Any]],
    checkpoints: Sequence[ProviderContinuationCheckpoint],
    resolution: MoonshotResolution,
) -> None:
    if not isinstance(checkpoints, Sequence):
        raise _bad_request("Moonshot provider continuation is invalid.")
    cursor = 0
    for checkpoint in checkpoints:
        try:
            validate_continuation_restore(
                checkpoint,
                ContinuationRestoreTarget(
                    provider="moonshot",
                    protocol="chat_completions",
                    model=resolution.model,
                    api_base_url=resolution.base_url,
                ),
            )
        except Exception:
            raise _bad_request("Moonshot provider continuation is invalid.") from None
        if checkpoint.state == "complete" and resolution.model == _DEFAULT_MODEL:
            final_round = checkpoint.rounds[-1]
            match_index = _find_round_owner(
                messages,
                assistant_content=final_round.assistant_content,
                call_ids=(),
                start=cursor,
            )
            if match_index is None:
                raise _bad_request("Moonshot continuation owner is missing.")
            expanded: list[dict[str, Any]] = []
            for round_ in checkpoint.rounds:
                assistant: dict[str, Any] = {
                    "role": "assistant",
                    "content": round_.assistant_content,
                }
                if round_.reasoning_blocks:
                    assistant["reasoning_content"] = "".join(round_.reasoning_blocks)
                if round_.calls:
                    assistant["tool_calls"] = [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                        for call in round_.calls
                    ]
                expanded.append(assistant)
                for call in round_.calls:
                    if call.result is None:
                        raise _bad_request("Moonshot provider continuation is invalid.")
                    expanded.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.call_id,
                            "content": call.result.value,
                        }
                    )
            messages[match_index : match_index + 1] = expanded
            cursor = match_index + len(expanded)
            continue
        for round_ in checkpoint.rounds:
            call_ids = tuple(call.call_id for call in round_.calls)
            match_index = _find_round_owner(
                messages,
                assistant_content=round_.assistant_content,
                call_ids=call_ids,
                start=cursor,
            )
            if match_index is None:
                raise _bad_request("Moonshot continuation owner is missing.")
            if round_.reasoning_blocks:
                messages[match_index]["reasoning_content"] = "".join(
                    round_.reasoning_blocks
                )
            cursor = match_index + 1


def _find_round_owner(
    messages: Sequence[Mapping[str, Any]],
    *,
    assistant_content: str,
    call_ids: tuple[str, ...],
    start: int,
) -> int | None:
    for index in range(start, len(messages)):
        message = messages[index]
        if (
            message.get("role") != "assistant"
            or message.get("content") != assistant_content
        ):
            continue
        raw_calls = message.get("tool_calls", ())
        message_ids = tuple(
            call.get("id") for call in raw_calls if isinstance(call, Mapping)
        )
        if message_ids == call_ids:
            return index
    return None
