"""DeepSeek hosted Chat-Completions provider profile (TASK-32852, ADR-062).

Transport, bounded retries, strict SSE framing, streamed usage capture, and
exactly-once resource closure live in :mod:`tldw_chatbook.LLM_Calls.hosted_chat`;
this module owns DeepSeek's request resolution, payload allowlist, finish
policy, and the legacy consumer surfaces (the shared ``LegacyLineStream`` and
the legacy ``DeepSeekResponse`` choices/usage dict).

Defects the migration fixes, pinned in
``Tests/LLM_Calls/test_groq_openrouter_migration_characterization.py``:
the sentinel-in-``finally`` Stop leak, unrequested/unforwarded streamed
usage, the duplicated ``[DONE]`` on normal completion, the synthetic
error-chunk yield on stream-read failures (a typed, redacted error now
raises instead), and the wrong metric labels (the old non-streaming path
logged ``mistral_api_*`` series for DeepSeek).

TASK-15677 (ADR-064, DeepSeek dual-API) builds on this profile: its
Responses-API wire mode lands as a mode here rather than on a hand-rolled
handler.
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


class DeepSeekFinishPolicy:
    """Validate DeepSeek finish and reasoning fields.

    DeepSeek emits the standard OpenAI finish reasons; ``content_filter`` is
    grouped with the terminal-text reasons so a filtered completion is a
    classified terminal state.
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
            raise HostedChatProtocolError("DeepSeek finish state is malformed.")
        if finish_reason == "tool_calls":
            if not has_calls:
                raise HostedChatProtocolError("DeepSeek finish state is inconsistent.")
        elif has_calls or not has_text:
            raise HostedChatProtocolError("DeepSeek finish state is inconsistent.")
        return finish_reason  # type: ignore[return-value]

    def validate_reasoning_content(self, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise HostedChatProtocolError("DeepSeek reasoning content is malformed.")
        return value


_FINISH_POLICY = DeepSeekFinishPolicy()


class DeepSeekResponse(dict):
    """Public legacy response dict with the normalized terminal turn attached."""

    def __init__(
        self, value: dict[str, Any], *, terminal_turn: HostedChatTurn
    ) -> None:
        super().__init__(value)
        self._terminal_turn = terminal_turn

    @property
    def terminal_turn(self) -> HostedChatTurn:
        return self._terminal_turn


def _deepseek_turn_response(turn: HostedChatTurn) -> DeepSeekResponse:
    message = turn.assistant_message
    if message is None:
        raise HostedChatProtocolError("DeepSeek response message is incomplete.")
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
    return DeepSeekResponse(response, terminal_turn=turn)


def _log_usage_metrics(model: str, usage: dict[str, Any]) -> None:
    log_histogram(
        "deepseek_api_input_tokens", usage.get("prompt_tokens", 0),
        labels={"model": model},
    )
    log_histogram(
        "deepseek_api_output_tokens", usage.get("completion_tokens", 0),
        labels={"model": model},
    )
    log_histogram(
        "deepseek_api_total_tokens", usage.get("total_tokens", 0),
        labels={"model": model},
    )


def _log_error_metrics(model: str, duration: float, exc: BaseException) -> None:
    status_code = getattr(exc, "status_code", None)
    error_type = "http_error" if status_code is not None else exc.__class__.__name__
    labels: dict[str, str] = {"model": model, "error_type": error_type}
    if status_code is not None:
        labels["status_code"] = str(status_code)
    log_counter("deepseek_api_error", labels=labels)
    log_histogram("deepseek_api_error_response_time", duration, labels=labels)


@_provider_recovery.unqualified
def chat_with_deepseek(
    input_data: list[dict[str, Any]],
    model: str | None = None,
    api_key: str | None = None,
    system_message: str | None = None,
    temp: float | None = None,
    streaming: bool | None = False,
    topp: float | None = None,  # top_p
    max_tokens: int | None = None,
    seed: int | None = None,
    stop: str | list[str] | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    response_format: dict[str, str] | None = None,
    n: int | None = None,
    user: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    logit_bias: dict[str, float] | None = None,
    custom_prompt_arg: str | None = None,
    api_base_url: str | None = None,
):
    del custom_prompt_arg
    start_time = time.time()
    cli_api_settings = get_runtime_config_snapshot().values.get("api_settings", {})
    deepseek_config = cli_api_settings.get("deepseek", {})
    final_api_key = api_key or deepseek_config.get("api_key")
    if not final_api_key:
        raise ChatConfigurationError(
            provider="deepseek", message="DeepSeek API Key required."
        )

    logger.debug("DeepSeek: API key provided.")
    current_model = model or deepseek_config.get("model", "deepseek-v4-flash")
    current_temp = (
        temp if temp is not None else float(deepseek_config.get("temperature", 0.1))
    )
    current_top_p = topp  # DeepSeek uses top_p
    current_streaming_cfg = deepseek_config.get("streaming", False)
    current_streaming = (
        streaming
        if streaming is not None
        else (
            str(current_streaming_cfg).lower() == "true"
            if isinstance(current_streaming_cfg, str)
            else bool(current_streaming_cfg)
        )
    )

    log_counter(
        "deepseek_api_request",
        labels={"model": current_model, "streaming": str(current_streaming)},
    )

    def _coerce_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning(f"Could not cast '{value}' to int. Using default.")
            return None

    current_max_tokens = (
        max_tokens
        if max_tokens is not None
        else _coerce_int(deepseek_config.get("max_tokens"))
    )

    api_messages = []
    if system_message:
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
    if seed is not None:
        data["seed"] = seed
    if stop is not None:
        data["stop"] = stop
    if logprobs is not None:
        data["logprobs"] = logprobs  # DeepSeek uses 'logprobs' (boolean)
    if top_logprobs is not None and data.get("logprobs"):
        data["top_logprobs"] = top_logprobs
    if presence_penalty is not None:
        data["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        data["frequency_penalty"] = frequency_penalty
    if response_format is not None:
        data["response_format"] = response_format
    if n is not None:
        data["n"] = n
    if user is not None:
        data["user"] = user
    if tools is not None:
        data["tools"] = tools
    if tool_choice is not None:
        data["tool_choice"] = tool_choice
    if logit_bias is not None:
        data["logit_bias"] = logit_bias
    if current_streaming:
        data["stream_options"] = {"include_usage": True}

    base_url = (
        api_base_url
        or deepseek_config.get("api_base_url")
        or builtin_provider_endpoint("deepseek", deepseek_config)
    )
    if not is_sensitive_llm_request():
        # task-2116: allowlisted summary only; see the OpenAI branch in
        # LLM_API_Calls for why a denylist isn't safe here.
        logger.debug(
            "DeepSeek Request Payload (safe fields only): "
            f"{safe_llm_request_payload_summary(data)}"
        )

    config = HostedHTTPTransportConfig(
        provider="deepseek",
        base_url=base_url,
        api_key=final_api_key,
        timeout=180.0 if current_streaming else 120.0,
        retries=llm_retry_count(int(deepseek_config.get("api_retries", 3))),
        retry_delay=float(deepseek_config.get("api_retry_delay", 1)),
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
            provider="deepseek",
            message="DeepSeek returned a malformed successful response.",
            status_code=502,
        ))
        raise ChatProviderError(
            provider="deepseek",
            message="DeepSeek returned a malformed successful response.",
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
        "deepseek_api_response_time",
        duration,
        labels={"model": current_model, "streaming": str(current_streaming)},
    )
    log_counter(
        "deepseek_api_success",
        labels={"model": current_model, "streaming": str(current_streaming)},
    )

    if isinstance(result, HostedChatStream):
        return LegacyLineStream(result)
    if result.usage is not None:
        _log_usage_metrics(current_model, result.usage)
    return _deepseek_turn_response(result)
