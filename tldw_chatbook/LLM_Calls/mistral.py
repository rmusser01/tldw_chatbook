"""Mistral hosted Chat-Completions provider profile (TASK-32852, ADR-062).

Transport, bounded retries, strict SSE framing, streamed usage capture, and
exactly-once resource closure live in :mod:`tldw_chatbook.LLM_Calls.hosted_chat`;
this module owns Mistral's request resolution, payload allowlist (including
the provider-specific ``random_seed``/``safe_prompt`` keys, the
``Accept: application/json`` header, and the system-message dedup), finish
policy, and the legacy consumer surfaces (the shared ``LegacyLineStream``
and the legacy ``MistralResponse`` choices/usage dict).

Provider-specific behavior preserved from the pre-migration handler: Mistral
uses ``random_seed`` (not ``seed``), does not support ``top_k``/``stop``/
penalties/``logprobs``/``n``/``user``/``logit_bias`` (none are forwarded),
prepends the system message only when the input does not already carry one,
and resolves its endpoint under the ``mistralai`` provider key.

Defects the migration fixes, pinned in
``Tests/LLM_Calls/test_groq_openrouter_migration_characterization.py``:
the sentinel-in-``finally`` Stop leak, unrequested/unforwarded streamed
usage, the duplicated ``[DONE]`` on normal completion, and the wrong
streaming metric label (the old streaming path logged
``openrouter_api_response_time`` for Mistral).
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_chatbook.Chat.console_provider_endpoints import builtin_provider_endpoint
from tldw_chatbook.config import get_runtime_config_snapshot
from tldw_chatbook.LLM_Calls import recovery_review as _provider_recovery
from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedChatProtocolError,
    HostedChatStream,
    HostedChatTurn,
    HostedHTTPTransportConfig,
    hosted_chat_request,
)
from tldw_chatbook.LLM_Calls.legacy_line_stream import LegacyLineStream
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram
from tldw_chatbook.Utils.sensitive_llm_logging import (
    is_sensitive_llm_request,
    llm_retry_count,
    safe_llm_request_payload_summary,
)


class MistralFinishPolicy:
    """Validate Mistral finish and reasoning fields.

    Mistral emits the standard OpenAI-family finish reasons; the engine's
    stream validator accepts only OpenAI-shaped deltas, which Mistral's
    Chat-Completions surface provides.
    """

    reasoning_disposition = "proprietary"

    def validate_finish(
        self,
        *,
        finish_reason: object,
        has_text: bool,
        has_calls: bool,
    ) -> str:
        if finish_reason not in {"stop", "tool_calls", "length", "content_filter"}:
            raise HostedChatProtocolError("Mistral finish state is malformed.")
        if finish_reason == "tool_calls":
            if not has_calls:
                raise HostedChatProtocolError("Mistral finish state is inconsistent.")
        elif has_calls or not has_text:
            raise HostedChatProtocolError("Mistral finish state is inconsistent.")
        return finish_reason  # type: ignore[return-value]

    def validate_reasoning_content(self, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise HostedChatProtocolError("Mistral reasoning content is malformed.")
        return value


_FINISH_POLICY = MistralFinishPolicy()


class MistralResponse(dict):
    """Public legacy response dict with the normalized terminal turn attached."""

    def __init__(
        self, value: dict[str, Any], *, terminal_turn: HostedChatTurn
    ) -> None:
        super().__init__(value)
        self._terminal_turn = terminal_turn

    @property
    def terminal_turn(self) -> HostedChatTurn:
        return self._terminal_turn


def _mistral_turn_response(turn: HostedChatTurn) -> MistralResponse:
    message = turn.assistant_message
    if message is None:
        raise HostedChatProtocolError("Mistral response message is incomplete.")
    response: dict[str, Any] = {
        "choices": [
            {
                "index": 0,
                "message": dict(message),
                "finish_reason": turn.finish_reason,
            }
        ]
    }
    if turn.usage is not None:
        response["usage"] = dict(turn.usage)
    return MistralResponse(response, terminal_turn=turn)


def _log_usage_metrics(model: str, usage: dict[str, Any]) -> None:
    log_histogram(
        "mistral_api_input_tokens", usage.get("prompt_tokens", 0),
        labels={"model": model},
    )
    log_histogram(
        "mistral_api_output_tokens", usage.get("completion_tokens", 0),
        labels={"model": model},
    )
    log_histogram(
        "mistral_api_total_tokens", usage.get("total_tokens", 0),
        labels={"model": model},
    )


def _log_error_metrics(model: str, duration: float, exc: BaseException) -> None:
    status_code = getattr(exc, "status_code", None)
    error_type = "http_error" if status_code is not None else exc.__class__.__name__
    labels: dict[str, str] = {"model": model, "error_type": error_type}
    if status_code is not None:
        labels["status_code"] = str(status_code)
    log_counter("mistral_api_error", labels=labels)
    log_histogram("mistral_api_error_response_time", duration, labels=labels)


@_provider_recovery.unqualified
def chat_with_mistral(
    input_data: list[dict[str, Any]],
    model: str | None = None,
    api_key: str | None = None,
    system_message: str | None = None,
    temp: float | None = None,
    streaming: bool | None = False,
    topp: float | None = None,
    max_tokens: int | None = None,
    random_seed: int | None = None,
    top_k: int | None = None,
    safe_prompt: bool | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    response_format: dict[str, str] | None = None,
    custom_prompt_arg: str | None = None,
    api_base_url: str | None = None,
):
    del custom_prompt_arg, top_k  # Mistral does not support top_k
    start_time = time.time()
    cli_api_settings = get_runtime_config_snapshot().values.get("api_settings", {})
    mistral_config = cli_api_settings.get("mistral", {})
    final_api_key = api_key or mistral_config.get("api_key")
    if not final_api_key:
        raise ChatConfigurationError(
            provider="mistral", message="Mistral API Key required."
        )

    logger.debug("Mistral: API key provided.")
    current_model = model or mistral_config.get("model", "mistral-large-latest")
    current_temp = (
        temp if temp is not None else float(mistral_config.get("temperature", 0.1))
    )
    current_top_p = topp  # Mistral uses top_p
    current_streaming_cfg = mistral_config.get("streaming", False)
    current_streaming = (
        streaming
        if streaming is not None
        else (
            str(current_streaming_cfg).lower() == "true"
            if isinstance(current_streaming_cfg, str)
            else bool(current_streaming_cfg)
        )
    )

    current_max_tokens = (
        max_tokens
        if max_tokens is not None
        else _coerce_int(mistral_config.get("max_tokens"))
    )
    current_safe_prompt = (
        safe_prompt
        if safe_prompt is not None
        else bool(mistral_config.get("safe_prompt", False))
    )

    log_counter(
        "mistral_api_request",
        labels={"model": current_model, "streaming": str(current_streaming)},
    )

    api_messages = []
    # Mistral expects the system message first; prepend only when the input
    # does not already carry one (pre-migration dedup, preserved).
    has_system_in_input = any(msg.get("role") == "system" for msg in input_data)
    if system_message and not has_system_in_input:
        api_messages.append({"role": "system", "content": system_message})
    api_messages.extend(input_data)

    data: dict[str, Any] = {
        "model": current_model,
        "messages": api_messages,
        "stream": current_streaming,
    }
    if current_temp is not None:
        data["temperature"] = current_temp
    if current_top_p is not None:
        data["top_p"] = current_top_p
    if current_max_tokens is not None:
        data["max_tokens"] = current_max_tokens
    if random_seed is not None:
        data["random_seed"] = random_seed  # Mistral uses random_seed
    if current_safe_prompt is not None:
        data["safe_prompt"] = current_safe_prompt  # Mistral specific
    if tools is not None:
        data["tools"] = tools
    if tool_choice is not None:
        data["tool_choice"] = tool_choice  # "auto", "any", "none"
    if response_format is not None:
        data["response_format"] = response_format  # {"type": "json_object"}
    if current_streaming:
        data["stream_options"] = {"include_usage": True}

    base_url = (
        api_base_url
        or mistral_config.get("api_base_url")
        or builtin_provider_endpoint("mistralai", mistral_config)
    )
    if not is_sensitive_llm_request():
        # task-2116: allowlisted summary only; see the OpenAI branch in
        # LLM_API_Calls for why a denylist isn't safe here.
        logger.debug(
            "Mistral Request Payload (safe fields only): "
            f"{safe_llm_request_payload_summary(data)}"
        )

    config = HostedHTTPTransportConfig(
        provider="mistral",
        base_url=base_url,
        api_key=final_api_key,
        timeout=180.0 if current_streaming else 120.0,
        retries=llm_retry_count(int(mistral_config.get("api_retries", 3))),
        retry_delay=float(mistral_config.get("api_retry_delay", 1)),
        extra_headers={"Accept": "application/json"},
    )
    try:
        result = hosted_chat_request(
            config=config,
            payload=data,
            streaming=current_streaming,
            finish_policy=_FINISH_POLICY,
        )
    except HostedChatProtocolError:
        duration = time.time() - start_time
        _log_error_metrics(current_model, duration, exc=ChatProviderError(
            provider="mistral",
            message="Mistral returned a malformed successful response.",
            status_code=502,
        ))
        raise ChatProviderError(
            provider="mistral",
            message="Mistral returned a malformed successful response.",
            status_code=502,
        ) from None
    except (
        ChatAuthenticationError,
        ChatRateLimitError,
        ChatBadRequestError,
        ChatProviderError,
    ) as exc:
        # The engine converts every transport failure to a typed, redacted
        # Chat error before it reaches here (ADR-062); matching the moonshot
        # exemplar, anything else propagates raw rather than being wrapped.
        _log_error_metrics(current_model, time.time() - start_time, exc=exc)
        raise

    duration = time.time() - start_time
    log_histogram(
        "mistral_api_response_time",
        duration,
        labels={"model": current_model, "streaming": str(current_streaming)},
    )
    log_counter(
        "mistral_api_success",
        labels={"model": current_model, "streaming": str(current_streaming)},
    )

    if isinstance(result, HostedChatStream):
        return LegacyLineStream(result)
    if result.usage is not None:
        _log_usage_metrics(current_model, result.usage)
    return _mistral_turn_response(result)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning(f"Could not cast '{value}' to int. Using default.")
        return None
