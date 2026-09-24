"""Generic strict hosted Chat-Completions engine for preset providers (ADR-179).

Phase 1, Tasks 4-5: the RESOLUTION layer (``resolve_hosted_request``) and
the PAYLOAD layer (``build_hosted_chat_payload``). Both port their
``LLM_Calls/zai.py`` counterparts, parameterized by a
``provider_registry.ProviderRecord`` preset instead of hardcoded Z.ai
constants: error copy is ``record.display_name``-prefixed, provider identity
is ``record.key``, env candidates are ``record.api_key_env_candidates``,
numeric/streaming defaults come from ``record.settings_defaults``, and the
payload surface is gated by ``record.payload_flags``.

Deliberate divergences from the zai template (spec: Generic hosted provider
engine and presets design, "Data flow" step 2):

- A record may ship no default base URL (Databricks workspace hosts are
  per-account): a missing URL raises an actionable "workspace base URL is
  required" error instead of falling back to a builtin URL.
- When the configured URL's path is empty or ``/`` and the record carries a
  ``base_url_suffix`` (e.g. Databricks ``/openai/v1``), the suffix is
  appended; any other path is validated as-is.
- A pasted terminal ``/chat/completions`` URL is REJECTED with actionable
  copy. (``normalize_hosted_chat_base_url`` accepts-and-strips that suffix --
  zai relies on the strip -- but the engine preset contract rejects the
  paste outright, so the check lives here, before the shared normalizer.)
- A preset may ship a blank default model (Databricks has none: gateway
  models are workspace-configured, and readiness requires key and URL, not a
  model). An unset model therefore resolves to ``""`` and is gated by the
  payload layer; a user-supplied blank model still fails closed here.
- Credentials and endpoint are resolved before the model so missing-key and
  missing-URL errors surface with their actionable copy first.

Payload-layer (Task 5) divergences from the zai template:

- A blank-default record may resolve ``model`` to ``""`` (Databricks ships
  none); the payload builder fails closed on an empty model instead of
  sending one.
- Optional body fields are gated by ``record.payload_flags``: a flag-off
  field with a caller-supplied value is a bad request (never silently
  dropped -- hidden caller intent is the failure the strict engine exists
  to surface), and ``reasoning_effort`` is gated by ``record.reasoning_effort``
  with the key from ``record.reasoning_effort_key``.
- ``record.extra_body_fields`` merge into the payload last, each validated
  bounded (preset authoring data, not caller input).
- No provider-invented fields: zai's ``thinking``/``request_id``/``user_id``
  quirks do not exist here -- the engine emits exactly the OpenAI body the
  record describes. Unknown keyword arguments are swallowed (``**_generic``)
  so one shared per-provider param map can drive every preset.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.parse import urlsplit

from tldw_chatbook.Chat.provider_continuation import (
    ContinuationRestoreTarget,
    ProviderContinuationCheckpoint,
    validate_continuation_restore,
)
from tldw_chatbook.LLM_Calls.hosted_chat import (
    ProviderPayloadValidators,
    TOOL_FUNCTION_NAME,
    normalize_hosted_chat_base_url,
)
from tldw_chatbook.config import (
    ProviderSettingsError,
    get_runtime_config_snapshot,
    provider_settings_for_key,
    resolve_provider_api_key,
)
from tldw_chatbook.provider_registry import ProviderRecord


@dataclass(frozen=True)
class HostedProviderResolution:
    """Immutable resolved engine request identity and transport policy."""

    provider: str
    model: str
    api_key: str = field(repr=False, compare=False)
    base_url: str
    timeout: float
    retries: int
    retry_delay: float
    streaming: bool


def resolve_hosted_request(
    record: ProviderRecord,
    *,
    explicit_api_key: object = None,
    explicit_base_url: object = None,
    explicit_model: object = None,
    explicit_timeout: object = None,
    explicit_retries: object = None,
    explicit_retry_delay: object = None,
    app_config: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> HostedProviderResolution:
    """Resolve one engine-driven hosted request from canonical immutable sources.

    Args:
        record: Registry preset driving every default, env candidate, and
            error message.
        explicit_api_key: Caller-supplied credential; wins over settings/env.
        explicit_base_url: Caller-supplied base URL; wins over settings.
        explicit_model: Caller-supplied model; wins over settings.
        explicit_timeout: Caller-supplied timeout override.
        explicit_retries: Caller-supplied retry override.
        explicit_retry_delay: Caller-supplied retry-delay override.
        app_config: Canonical config mapping; defaults to the runtime
            snapshot. Never mutated.
        environ: Environment mapping; defaults to ``os.environ``.

    Returns:
        The immutable resolved request identity and transport policy.

    Raises:
        ChatConfigurationError: On any missing or malformed credential,
            base URL, model, or transport setting. Credential and endpoint
            failures surface before model failures (see module docstring).
    """
    validators = _validators_for(record)
    config: object = (
        get_runtime_config_snapshot().values if app_config is None else app_config
    )
    if not isinstance(config, Mapping):
        raise validators.configuration_error(
            f"{record.display_name} application configuration is invalid."
        )
    api_settings = config.get("api_settings", {})
    if not isinstance(api_settings, Mapping):
        raise validators.configuration_error(
            f"{record.display_name} api_settings must be a configuration table."
        )
    try:
        settings = provider_settings_for_key(api_settings, record.key)
    except ProviderSettingsError:
        raise validators.configuration_error(
            f"{record.display_name} api_settings.{record.key} must be one "
            "unambiguous configuration table."
        ) from None
    defaults = record.settings_defaults
    transport_settings = dict(settings)
    if explicit_timeout is not None:
        transport_settings["timeout"] = explicit_timeout
    if explicit_retries is not None:
        transport_settings["retries"] = explicit_retries
    if explicit_retry_delay is not None:
        transport_settings["retry_delay"] = explicit_retry_delay
    environment = os.environ if environ is None else environ
    api_key = _resolve_api_key(
        record, explicit=explicit_api_key, settings=settings, environ=environment
    )
    base_url = _resolve_base_url(record, explicit=explicit_base_url, settings=settings)
    return HostedProviderResolution(
        provider=record.key,
        model=_resolve_string(
            record,
            explicit=explicit_model,
            settings=settings,
            name="model",
            default=_settings_default(defaults, "model", ""),
        ),
        api_key=api_key,
        base_url=base_url,
        timeout=validators.positive_number(
            transport_settings, "timeout", _settings_default(defaults, "timeout", 90.0)
        ),
        retries=validators.nonnegative_integer(
            transport_settings, "retries", _settings_default(defaults, "retries", 3)
        ),
        retry_delay=validators.nonnegative_number(
            transport_settings,
            "retry_delay",
            _settings_default(defaults, "retry_delay", 5.0),
        ),
        streaming=_resolve_streaming(
            record,
            settings=settings,
            default=_settings_default(defaults, "streaming", True),
        ),
    )


def _validators_for(record: ProviderRecord) -> ProviderPayloadValidators:
    """Build the shared error/validator family for one record."""
    return ProviderPayloadValidators(record.key, record.display_name)


def _settings_default(
    defaults: Mapping[str, object], name: str, fallback: object
) -> object:
    """Return a preset's shipped default, or the fallback when absent."""
    value = defaults.get(name, fallback)
    return fallback if value is None else value


def _resolve_string(
    record: ProviderRecord,
    *,
    explicit: object,
    settings: Mapping[str, object],
    name: str,
    default: object,
) -> str:
    validators = _validators_for(record)
    if explicit is not None:
        value: object = explicit
        supplied = True
    elif name in settings:
        value = settings.get(name)
        supplied = True
    else:
        value = default
        supplied = False
    if not isinstance(value, str):
        raise validators.configuration_error(
            f"{record.display_name} {name} is invalid."
        )
    stripped = value.strip()
    if not stripped:
        # A preset may ship a blank default (Databricks has no default
        # model); an unset value then passes through blank for the payload
        # layer to gate. A user-supplied blank is still an error.
        if supplied:
            raise validators.configuration_error(
                f"{record.display_name} {name} is invalid."
            )
        return ""
    return stripped


def _resolve_api_key(
    record: ProviderRecord,
    *,
    explicit: object,
    settings: Mapping[str, object],
    environ: Mapping[str, str],
) -> str:
    validators = _validators_for(record)
    if explicit is not None:
        resolved = resolve_provider_api_key(explicit)
        if resolved is None:
            raise validators.configuration_error(
                f"{record.display_name} explicit API key is invalid."
            )
        return resolved
    if "api_key" in settings:
        resolved = resolve_provider_api_key(settings.get("api_key"))
        if resolved is None:
            raise validators.configuration_error(
                f"{record.display_name} api_settings.{record.key}.api_key is invalid."
            )
        return resolved
    env_name = settings.get("api_key_env_var", record.api_key_env_var)
    if not isinstance(env_name, str) or not env_name.strip():
        raise validators.configuration_error(
            f"{record.display_name} api_settings.{record.key}.api_key_env_var "
            "is invalid."
        )
    for candidate in dict.fromkeys(
        (env_name.strip(), *record.api_key_env_candidates)
    ):
        resolved = resolve_provider_api_key(environ.get(candidate))
        if resolved is not None:
            return resolved
    raise validators.configuration_error(
        f"{record.display_name} API key is required."
    )


def _resolve_base_url(
    record: ProviderRecord,
    *,
    explicit: object,
    settings: Mapping[str, object],
) -> str:
    validators = _validators_for(record)
    candidate: object = (
        explicit if explicit is not None else settings.get("api_base_url")
    )
    if candidate is None and isinstance(record.default_base_url, str):
        candidate = record.default_base_url
    if candidate is None:
        raise validators.configuration_error(
            f"{record.display_name} workspace base URL is required."
        )
    if (
        isinstance(candidate, str)
        and isinstance(record.base_url_suffix, str)
        and _url_path(candidate) in ("", "/")
    ):
        candidate = f"{candidate.rstrip('/')}{record.base_url_suffix}"
    if isinstance(candidate, str) and _names_terminal_chat_completions(candidate):
        raise validators.configuration_error(
            f"{record.display_name} API base URL must not include a terminal "
            "/chat/completions endpoint path."
        )
    default_url = (
        record.default_base_url
        if isinstance(record.default_base_url, str)
        else ""
    )
    try:
        return normalize_hosted_chat_base_url(candidate, default=default_url)
    except ValueError:
        raise validators.configuration_error(
            f"{record.display_name} API base URL is invalid."
        ) from None


def _resolve_streaming(
    record: ProviderRecord,
    *,
    settings: Mapping[str, object],
    default: object,
) -> bool:
    validators = _validators_for(record)
    value = settings.get("streaming", default)
    if type(value) is not bool:
        raise validators.configuration_error(
            f"{record.display_name} streaming must be a boolean."
        )
    return value


def _url_path(candidate: str) -> str | None:
    """Return the URL path, or None when the candidate cannot be split."""
    try:
        return urlsplit(candidate).path
    except ValueError:
        return None


def _names_terminal_chat_completions(candidate: str) -> bool:
    """Check whether the URL path ends with a pasted ``/chat/completions``."""
    path = _url_path(candidate)
    if path is None:
        return False
    segments = tuple(
        segment.casefold() for segment in path.strip("/").split("/")
    )
    return len(segments) >= 2 and segments[-2:] == ("chat", "completions")


# --- payload layer (Task 5; ports zai.py's payload-side helpers) ---


def build_hosted_chat_payload(
    record: ProviderRecord,
    *,
    resolution: HostedProviderResolution,
    messages_payload: Sequence[Mapping[str, Any]],
    system_message: object = None,
    streaming: object = None,
    tools: object = None,
    tool_choice: object = None,
    reasoning_effort: object = None,
    provider_continuations: Sequence[ProviderContinuationCheckpoint] = (),
    temperature: object = None,
    top_p: object = None,
    max_tokens: object = None,
    stop: object = None,
    response_format: object = None,
    seed: object = None,
    n: object = None,
    user: object = None,
    **_generic: object,
) -> dict[str, Any]:
    """Build one validated engine request without mutating inputs.

    Args:
        record: Registry preset driving flags, error copy, and extra body.
        resolution: Resolved request identity from ``resolve_hosted_request``.
        messages_payload: Chat-Completions message history.
        system_message: Optional system prompt; errors when the history
            already carries a system message.
        streaming: Transport streaming override; defaults to the resolution.
        tools: OpenAI function-tool descriptors.
        tool_choice: Only ``"auto"`` (with tools) is supported.
        reasoning_effort: Reasoning-effort request; requires
            ``record.reasoning_effort``.
        provider_continuations: Durable continuation checkpoints to restore
            onto the message history.
        temperature: Sampler in [0, 1]; requires the ``temperature`` flag.
        top_p: Sampler in [0, 1]; requires the ``top_p`` flag.
        max_tokens: Positive integer; requires the ``max_tokens`` flag.
        stop: Stop sequence(s); requires the ``stop`` flag.
        response_format: ``text``/``json_object``/``json_schema`` mapping;
            requires the ``response_format`` flag.
        seed: Integer seed; requires the ``seed`` flag.
        n: Positive completion count; requires the ``n`` flag.
        user: Bounded caller identity; requires the ``user`` flag.
        **_generic: Unknown keywords are ignored (one shared per-provider
            param map drives every preset).

    Returns:
        The validated request body.

    Raises:
        ChatBadRequestError: On an empty model, a resolution/record
            mismatch, or any malformed, unsupported, or flag-off-supplied
            value.
    """
    validators = _validators_for(record)
    bad_request = validators.bad_request
    if (
        type(resolution) is not HostedProviderResolution
        or resolution.provider != record.key
    ):
        raise bad_request(f"{record.display_name} resolution is invalid.")
    if not resolution.model:
        # Task 4 contract: a blank-default record resolves model to "" and
        # the payload layer gates it (readiness requires key and URL, not a
        # model) instead of sending a modelless request.
        raise bad_request(f"{record.display_name} model is required.")
    stream = resolution.streaming if streaming is None else streaming
    if type(stream) is not bool:
        raise bad_request(f"{record.display_name} streaming must be a boolean.")
    _validate_sampler(record, "temperature", temperature)
    _validate_sampler(record, "top_p", top_p)
    messages = _normalize_messages(
        record, messages_payload, system_message=system_message
    )
    validated_tools = _normalize_tools(record, tools)
    validated_choice = _normalize_tool_choice(record, tool_choice, validated_tools)
    _apply_continuations(record, messages, provider_continuations, resolution)

    payload: dict[str, Any] = {
        "model": resolution.model,
        "messages": messages,
        "stream": stream,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if max_tokens is not None:
        _require_payload_flag(record, "max_tokens")
        payload["max_tokens"] = validators.positive_integer("max_tokens", max_tokens)
    if stop is not None:
        _require_payload_flag(record, "stop")
        payload["stop"] = validators.normalize_stop(stop)
    if response_format is not None:
        _require_payload_flag(record, "response_format")
        payload["response_format"] = validators.normalize_response_format(
            response_format
        )
    if seed is not None:
        _require_payload_flag(record, "seed")
        if type(seed) is not int:
            raise bad_request(f"{record.display_name} seed is invalid.")
        payload["seed"] = seed
    if n is not None:
        _require_payload_flag(record, "n")
        payload["n"] = validators.positive_integer("n", n)
    if user is not None:
        _require_payload_flag(record, "user")
        payload["user"] = _bounded_identifier(record, "user", user)
    if validated_tools is not None:
        payload["tools"] = validated_tools
    if validated_choice is not None:
        payload["tool_choice"] = validated_choice
    if reasoning_effort is not None:
        if not record.reasoning_effort:
            raise bad_request(
                f"{record.display_name} reasoning effort is unsupported."
            )
        payload[record.reasoning_effort_key or "reasoning_effort"] = (
            _bounded_identifier(record, "reasoning effort", reasoning_effort)
        )
    for key, value in record.extra_body_fields.items():
        if not validators.json_shape_is_bounded(value):
            raise bad_request(f"{record.display_name} extra body field {key} is invalid.")
        payload[key] = value
    return payload


def _require_payload_flag(record: ProviderRecord, flag: str) -> None:
    """Fail closed when a preset lacks a payload flag but a value arrived."""
    if flag not in record.payload_flags:
        raise _validators_for(record).bad_request(
            f"{record.display_name} {flag} is unsupported."
        )


def _normalize_messages(
    record: ProviderRecord,
    value: object,
    *,
    system_message: object,
) -> list[dict[str, Any]]:
    validators = _validators_for(record)
    bad_request = validators.bad_request
    display = record.display_name
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise bad_request(f"{display} messages must be a sequence.")
    result: list[dict[str, Any]] = []
    has_system = any(
        isinstance(message, Mapping) and message.get("role") == "system"
        for message in value
    )
    if system_message is not None:
        if not isinstance(system_message, str) or has_system:
            raise bad_request(f"{display} system message ownership is invalid.")
        result.append({"role": "system", "content": system_message})

    call_ids: set[str] = set()
    pending_ids: list[str] = []
    for raw_message in value:
        if not isinstance(raw_message, Mapping):
            raise bad_request(f"{display} message is malformed.")
        role = raw_message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise bad_request(f"{display} message role is invalid.")
        if role == "tool":
            if not pending_ids or set(raw_message) != {
                "role",
                "tool_call_id",
                "content",
            }:
                raise bad_request(f"{display} tool result is malformed or orphaned.")
            call_id = raw_message.get("tool_call_id")
            content = raw_message.get("content")
            if call_id != pending_ids[0] or not isinstance(content, str):
                raise bad_request(f"{display} tool result ordering is invalid.")
            pending_ids.pop(0)
            result.append(deepcopy(dict(raw_message)))
            continue
        if pending_ids:
            raise bad_request(f"{display} tool call batch is incomplete.")
        allowed = {"role", "content"} | (
            {"tool_calls"} if role == "assistant" else set()
        )
        if set(raw_message) - allowed:
            raise bad_request(f"{display} message fields are unsupported.")
        content = raw_message.get("content")
        if role == "assistant":
            if content is not None and not isinstance(content, str):
                raise bad_request(f"{display} assistant content is invalid.")
        elif not isinstance(content, str):
            raise bad_request(f"{display} message content is invalid.")
        safe: dict[str, Any] = {
            "role": role,
            "content": "" if content is None else content,
        }
        if role == "assistant" and "tool_calls" in raw_message:
            calls = validators.normalize_call_batch(raw_message.get("tool_calls"), call_ids)
            safe["tool_calls"] = list(calls)
            pending_ids = [cast(str, call["id"]) for call in calls]
        result.append(safe)
    if pending_ids:
        raise bad_request(f"{display} tool call batch is incomplete.")
    return result


def _normalize_tools(
    record: ProviderRecord, value: object
) -> list[dict[str, Any]] | None:
    validators = _validators_for(record)
    bad_request = validators.bad_request
    display = record.display_name
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise bad_request(f"{display} tools are malformed.")
    names: set[str] = set()
    result: list[dict[str, Any]] = []
    for raw_tool in value:
        if not isinstance(raw_tool, Mapping) or set(raw_tool) != {"type", "function"}:
            raise bad_request(f"{display} supports function tools only.")
        function = raw_tool.get("function")
        if raw_tool.get("type") != "function" or not isinstance(function, Mapping):
            raise bad_request(f"{display} supports function tools only.")
        if set(function) != {"name", "description", "parameters"}:
            raise bad_request(f"{display} function tool is malformed.")
        name = function.get("name")
        parameters = function.get("parameters")
        if (
            not isinstance(name, str)
            or not TOOL_FUNCTION_NAME.fullmatch(name)
            or name in names
            or not isinstance(function.get("description"), str)
            or not cast(str, function.get("description")).strip()
            or not isinstance(parameters, Mapping)
            or parameters.get("type") != "object"
            or not validators.json_shape_is_bounded(parameters)
        ):
            raise bad_request(f"{display} function tool is malformed.")
        names.add(name)
        result.append(deepcopy(dict(raw_tool)))
    return result


def _normalize_tool_choice(
    record: ProviderRecord,
    value: object,
    tools: Sequence[Mapping[str, Any]] | None,
) -> str | None:
    if value is None:
        return None
    if value == "auto" and tools is not None:
        return "auto"
    raise _validators_for(record).bad_request(
        f"{record.display_name} tool choice is unsupported."
    )


def _apply_continuations(
    record: ProviderRecord,
    messages: list[dict[str, Any]],
    checkpoints: Sequence[ProviderContinuationCheckpoint],
    resolution: HostedProviderResolution,
) -> None:
    bad_request = _validators_for(record).bad_request
    display = record.display_name
    if not isinstance(checkpoints, Sequence):
        raise bad_request(f"{display} provider continuation is invalid.")
    cursor = 0
    for checkpoint in checkpoints:
        try:
            validate_continuation_restore(
                checkpoint,
                ContinuationRestoreTarget(
                    provider=record.key,
                    protocol=record.continuation_protocol or "chat_completions",
                    model=resolution.model,
                    api_base_url=resolution.base_url,
                ),
            )
        except Exception:
            raise bad_request(f"{display} provider continuation is invalid.") from None
        for round_ in checkpoint.rounds:
            call_ids = tuple(call.call_id for call in round_.calls)
            match_index = _find_owner(
                messages,
                assistant_content=round_.assistant_content,
                call_ids=call_ids,
                start=cursor,
            )
            if match_index is None:
                raise bad_request(f"{display} continuation owner is missing.")
            if round_.reasoning_blocks:
                messages[match_index]["reasoning_content"] = "".join(
                    round_.reasoning_blocks
                )
            cursor = match_index + 1


def _find_owner(
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
        ids = tuple(call.get("id") for call in raw_calls if isinstance(call, Mapping))
        if ids == call_ids:
            return index
    return None


def _validate_sampler(record: ProviderRecord, name: str, value: object) -> None:
    if value is None:
        return
    validators = _validators_for(record)
    if name not in record.payload_flags:
        raise validators.bad_request(f"{record.display_name} {name} is unsupported.")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise validators.bad_request(f"{record.display_name} {name} is invalid.")


def _bounded_identifier(record: ProviderRecord, name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
    ):
        raise _validators_for(record).bad_request(
            f"{record.display_name} {name} is invalid."
        )
    return value
