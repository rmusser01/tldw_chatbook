# LLM_API_Calls.py
#########################################
# General LLM API Calling Library
# This library is used to perform API Calls against commercial LLM endpoints.
#
####
####################
# Function List
#
# 1. extract_text_from_segments(segments: List[Dict]) -> str
# 2. chat_with_openai(api_key, file_path, custom_prompt_arg, streaming=None)
# 3. chat_with_anthropic(api_key, file_path, model, custom_prompt_arg, max_retries=3, retry_delay=5, streaming=None)
# 4. chat_with_cohere(api_key, file_path, model, custom_prompt_arg, streaming=None)
# 5. chat_with_groq(api_key, input_data, custom_prompt_arg, system_prompt=None, streaming=None):
# 6. chat_with_openrouter(api_key, input_data, custom_prompt_arg, system_prompt=None, streaming=None)
# 7. chat_with_huggingface(api_key, input_data, custom_prompt_arg, system_prompt=None, streaming=None)
# 8. chat_with_deepseek(api_key, input_data, custom_prompt_arg, system_prompt=None, streaming=None)
# 9. chat_with_moonshot(api_key, input_data, custom_prompt_arg, system_prompt=None, streaming=None)
# 10. chat_with_zai(api_key, input_data, model, custom_prompt_arg, streaming=None)
#
#
####################
#
# Import necessary libraries
import json
import time
from collections.abc import Mapping
from copy import deepcopy
from typing import List, Any, Optional, Tuple, Dict, Union
from urllib.parse import urlparse

#
# Import 3rd-Party Libraries
import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

#
# Import Local libraries
from tldw_chatbook.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatRateLimitError,
    ChatBadRequestError,
    ChatProviderError,
    ChatConfigurationError,
)
from tldw_chatbook.Chat.console_provider_endpoints import builtin_provider_endpoint
from tldw_chatbook.Chat.provider_continuation import ProviderContinuationCheckpoint
from tldw_chatbook.config import (
    get_cli_setting,
    get_runtime_config_snapshot,
    load_settings,
    resolve_provider_api_key,
)
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram
from tldw_chatbook.Utils.egress import create_default_session
from tldw_chatbook.LLM_Calls.moonshot import (
    chat_with_moonshot as _strict_chat_with_moonshot,
)
from tldw_chatbook.LLM_Calls.zai import chat_with_zai as _strict_chat_with_zai
from tldw_chatbook.model_capabilities import (
    anthropic_model_rejects_disabled_thinking,
    anthropic_model_rejects_fixed_thinking_budget,
    anthropic_model_rejects_sampling_params,
    anthropic_model_thinks_by_default,
    openai_model_rejects_sampling_params,
    openai_model_requires_max_completion_tokens,
)
from tldw_chatbook.Utils.input_validation import validate_url
from tldw_chatbook.Utils.sensitive_llm_logging import (
    is_sensitive_llm_request,
    llm_content_byte_count,
    llm_retry_count,
    safe_llm_error_detail,
    safe_llm_exception_message,
    safe_llm_request_payload_summary,
    safe_llm_url_host,
)
#
#######################################################################################################################
# Provider Parameter Support Documentation
#
# IMPORTANT: None of the commercial providers in this file accept a 'provider_name' parameter.
# The provider_name parameter is only used by some local providers in LLM_API_Calls_Local.py
# for dynamic configuration loading.
#
# Commercial providers that DO NOT accept provider_name:
# - chat_with_openai
# - chat_with_anthropic
# - chat_with_cohere
# - chat_with_deepseek
# - chat_with_groq
# - chat_with_google
# - chat_with_huggingface
# - chat_with_mistral
# - chat_with_openrouter
# - chat_with_moonshot
#
#######################################################################################################################
# Function Definitions
#

# FIXME: Update to include full arguments


# --- Helper function for safe type conversion ---
def _safe_cast(value: Any, cast_to: type, default: Any = None) -> Any:
    """Safely casts value to specified type, returning default on failure."""
    if value is None:
        return default
    try:
        return cast_to(value)
    except (ValueError, TypeError):
        logger.warning(
            f"Could not cast '{value}' to {cast_to}. Using default: {default}"
        )
        return default


def _optional_config_string(value: Any, default: str = "") -> str:
    """Return a stripped config string, or a safe default for empty values.

    Args:
        value: Raw configuration value.
        default: Value to use when the raw value is missing or empty.

    Returns:
        A stripped string suitable for URL/path construction.
    """
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or default
    return str(value).strip() or default


def _huggingface_router_chat_url(base_url: Any) -> Optional[str]:
    """Return the OpenAI-compatible HuggingFace router chat URL.

    Args:
        base_url: Raw HuggingFace router base URL from configuration.

    Returns:
        Normalized ``/v1/chat/completions`` URL for HuggingFace router bases,
        or ``None`` when the URL is invalid or points to another host.
    """
    stripped_base_url = _optional_config_string(base_url)
    if not stripped_base_url:
        return None
    candidate = (
        stripped_base_url
        if "://" in stripped_base_url
        else f"https://{stripped_base_url}"
    )
    if not validate_url(candidate):
        return None
    parsed = urlparse(candidate)
    if (parsed.hostname or "").lower() != "router.huggingface.co":
        return None
    return f"{parsed.scheme}://{parsed.netloc}/v1/chat/completions"


def extract_text_from_segments(segments):
    logger.debug(f"Segments received: {segments}")
    logger.debug(f"Type of segments: {type(segments)}")

    text = ""

    if isinstance(segments, list):
        for segment in segments:
            logger.debug(f"Current segment: {segment}")
            logger.debug(f"Type of segment: {type(segment)}")
            if "Text" in segment:
                text += segment["Text"] + " "
            else:
                logger.warning(f"Skipping segment due to missing 'Text' key: {segment}")
    else:
        logger.warning(f"Unexpected type of 'segments': {type(segments)}")

    return text.strip()


def _parse_data_url_for_multimodal(data_url: str) -> Optional[Tuple[str, str]]:
    """Parses a data URL (e.g., data:image/png;base64,xxxx) into (mime_type, base64_data)."""
    if data_url.startswith("data:") and ";base64," in data_url:
        try:
            header, b64_data = data_url.split(";base64,", 1)
            mime_type = header.split("data:", 1)[1]
            return mime_type, b64_data
        except Exception as e:
            logger.warning(f"Could not parse data URL: {data_url[:60]}... Error: {e}")
            return None
    logger.debug(f"Data URL did not match expected format: {data_url[:60]}...")
    return None


_ANTHROPIC_THINKING_BUDGETS_BY_EFFORT = {
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
    "max": 32768,
}
# Models that merely *prefer* adaptive thinking. Models that outright REJECT a
# fixed thinking budget are not listed here -- that is a provider request
# capability and comes from
# `model_capabilities.anthropic_model_rejects_fixed_thinking_budget`, so a new
# release in those families never needs a marker added by hand (TASK-18414).
_ANTHROPIC_ADAPTIVE_THINKING_MODEL_MARKERS = (
    "sonnet-4-6",
    "sonnet-4.6",
)
_ANTHROPIC_SONNET_5_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


def _is_present_setting(value: object) -> bool:
    return value is not None and str(value).strip() != ""


def _is_openai_gpt_5_6_model(model: object) -> bool:
    """Return whether ``model`` is an unprefixed OpenAI GPT-5.6 family ID."""
    if not isinstance(model, str):
        return False
    normalized_model = model.strip().lower()
    return normalized_model == "gpt-5.6" or normalized_model.startswith("gpt-5.6-")


def _normalize_openai_reasoning_effort(value: object) -> Optional[str]:
    """Return a normalized OpenAI reasoning effort, if one was provided."""
    if not _is_present_setting(value):
        return None
    return str(value).strip().lower()


def _openai_use_responses_api(
    normalized_reasoning_effort: Optional[str],
    reasoning_summary: object,
    verbosity: object,
    is_gpt_5_6_model: bool,
) -> bool:
    return (
        normalized_reasoning_effort is not None
        and (normalized_reasoning_effort != "none" or not is_gpt_5_6_model)
    ) or (_is_present_setting(reasoning_summary) or _is_present_setting(verbosity))


def _extract_openai_responses_text(response_data: Dict[str, Any]) -> str:
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
    response_data: Dict[str, Any],
    *,
    model: str,
) -> Dict[str, Any]:
    """Normalize Responses API output to the chat-completions shape used by Console."""
    return {
        "id": response_data.get("id", f"resp-openai-{time.time_ns()}"),
        "object": "chat.completion",
        "created": int(response_data.get("created_at") or time.time()),
        "model": response_data.get("model", model),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": _extract_openai_responses_text(response_data),
                },
                "finish_reason": "stop"
                if response_data.get("status") in {None, "completed"}
                else response_data.get("status"),
            }
        ],
        "usage": response_data.get("usage"),
    }


def _responses_stream_to_chat_sse(response, *, model: str):
    completion_id = f"chatcmpl-openai-responses-{time.time_ns()}"
    created_ts = int(time.time())
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
                event = json.loads(payload_text)
            except json.JSONDecodeError:
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
                yield f"data: {json.dumps(chunk)}\n\n"
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
                yield f"data: {json.dumps(chunk)}\n\n"
            elif event_type == "error":
                yield f"data: {payload_text}\n\n"
    except requests.exceptions.RequestException as exc:
        logger.opt(exception=True).error(
            "OpenAI Responses: stream connection error: %s", exc
        )
        yield (
            "data: "
            + json.dumps({"error": {"message": f"Stream connection error: {exc}"}})
            + "\n\n"
        )
    finally:
        yield "data: [DONE]\n\n"
        if response:
            response.close()


def _anthropic_uses_adaptive_thinking(model: object) -> bool:
    """Return whether ``model`` must be driven with adaptive thinking.

    True either because the provider rejects a fixed thinking budget outright
    (the capability predicate) or because the model merely prefers adaptive
    thinking (the marker list).
    """
    if anthropic_model_rejects_fixed_thinking_budget(model):
        return True
    model_name = str(model or "").lower()
    return any(
        marker in model_name for marker in _ANTHROPIC_ADAPTIVE_THINKING_MODEL_MARKERS
    )


def _anthropic_is_sonnet_5(model: object) -> bool:
    """Return whether model is the documented unprefixed Claude Sonnet 5 family.

    This selects Sonnet 5's *effort shape* (bare ``output_config.effort`` with
    no ``thinking`` key). It is deliberately not the answer to "does this model
    reject sampling parameters / a fixed thinking budget" -- those are
    capability predicates in ``model_capabilities`` (TASK-18414) -- nor to
    "how must thinking OFF be expressed", which is the
    ``anthropic_model_thinks_by_default`` / ``anthropic_model_rejects_disabled_
    thinking`` predicate pair (TASK-18800).
    """
    model_name = str(model or "").lower()
    return model_name == "claude-sonnet-5" or model_name.startswith("claude-sonnet-5-")


def _anthropic_thinking_config(
    *,
    model: object,
    thinking_effort: object,
    thinking_budget_tokens: object,
    max_tokens: int,
) -> tuple[dict[str, object] | None, dict[str, object] | None, int]:
    """Map Anthropic thinking settings to thinking and output configuration."""
    effort = str(thinking_effort or "").strip().lower()
    if effort == "off":
        if thinking_budget_tokens is not None:
            logger.warning(
                "Anthropic: ignoring fixed thinking budget for model %s with thinking off",
                model,
            )
        # How OFF must be expressed is a per-family capability (TASK-18800):
        # families that think by default need an explicit disabled config,
        # EXCEPT the always-on families, which 400-reject it -- there omission
        # is the only valid payload and thinking still runs (surfaced to the
        # user by the Console settings warning in console_session_settings).
        if anthropic_model_thinks_by_default(model):
            if not anthropic_model_rejects_disabled_thinking(model):
                return {"type": "disabled"}, None, max_tokens
            logger.warning(
                "Anthropic: thinking cannot be turned off on always-on model %s; "
                "omitting the thinking parameter (adaptive thinking still runs)",
                model,
            )
        return None, None, max_tokens
    budget = _safe_cast(thinking_budget_tokens, int)
    is_sonnet_5 = _anthropic_is_sonnet_5(model)
    if is_sonnet_5:
        if thinking_budget_tokens is not None:
            logger.warning(
                "Anthropic: ignoring fixed thinking budget for Claude Sonnet 5 model %s",
                model,
            )
        if effort in _ANTHROPIC_SONNET_5_EFFORTS:
            return None, {"effort": effort}, max_tokens
        return None, None, max_tokens
    if _anthropic_uses_adaptive_thinking(model):
        if effort:
            return {"type": "adaptive"}, {"effort": effort}, max_tokens
        if budget is not None:
            logger.warning(
                "Anthropic: ignoring fixed thinking budget for adaptive-thinking model %s",
                model,
            )
        return None, None, max_tokens
    if budget is None and effort:
        budget = _ANTHROPIC_THINKING_BUDGETS_BY_EFFORT.get(effort)
    if budget is None:
        return None, None, max_tokens
    final_budget = max(1024, int(budget))
    final_max_tokens = max_tokens
    if final_budget >= final_max_tokens:
        final_max_tokens = final_budget + 1024
        logger.warning(
            "Anthropic: increased max_tokens to %s so thinking budget %s has output room.",
            final_max_tokens,
            final_budget,
        )
    return {"type": "enabled", "budget_tokens": final_budget}, None, final_max_tokens


def get_openai_embeddings(input_data: str, model: str) -> List[float]:
    """
    Get embeddings for the input text from OpenAI API.
    Args:
        input_data (str): The input text to get embeddings for.
        model (str): The model to use for generating embeddings.
    Returns:
        List[float]: The embeddings generated by the API.
    """
    loaded_config_data = load_settings()
    api_key = loaded_config_data["openai_api"]["api_key"]

    if not api_key:
        logger.error("OpenAI Embeddings: API key not found or is empty")
        raise ValueError(
            "OpenAI Embeddings: API Key Not Provided/Found in Config file or is empty"
        )

    logger.debug("OpenAI Embeddings: API key provided.")
    logger.debug(
        f"OpenAI Embeddings: Raw input data (first 500 chars): {str(input_data)[:500]}..."
    )
    logger.debug(f"OpenAI Embeddings: Using model: {model}")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    request_data = {
        "input": input_data,
        "model": model,
    }
    try:
        logger.debug("OpenAI Embeddings: Posting request to embeddings API")
        with create_default_session() as session:
            response = session.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=request_data,
            )
        logger.debug(f"Full API response data: {response}")
        if response.status_code == 200:
            response_data = response.json()
            if "data" in response_data and len(response_data["data"]) > 0:
                embedding = response_data["data"][0]["embedding"]
                logger.debug("OpenAI Embeddings: Embeddings retrieved successfully")
                return embedding
            else:
                logger.warning(
                    "OpenAI Embeddings: Embedding data not found in the response"
                )
                raise ValueError(
                    "OpenAI Embeddings: Embedding data not available in the response"
                )
        else:
            logger.error(
                f"OpenAI Embeddings: request failed with status code {response.status_code}"
            )
            logger.error(f"OpenAI Embeddings: Error response: {response.text}")
            # Propagate HTTPError to be caught by chat_api_call's handler (if this were called from there)
            # Or raise specific error if called directly
            response.raise_for_status()  # This will raise HTTPError
            # Fallback if raise_for_status doesn't cover it (it should)
            raise ValueError(
                f"OpenAI Embeddings: Failed to retrieve. Status code: {response.status_code}"
            )
    except requests.RequestException as e:
        logger.opt(exception=True).error(
            f"OpenAI Embeddings: Error making API request: {str(e)}"
        )
        raise ValueError(f"OpenAI Embeddings: Error making API request: {str(e)}")
    except Exception as e:
        logger.opt(exception=True).error(
            f"OpenAI Embeddings: Unexpected error: {str(e)}"
        )
        raise ValueError(f"OpenAI Embeddings: Unexpected error occurred: {str(e)}")


def chat_with_openai(
    input_data: List[Dict[str, Any]],  # Mapped from 'messages_payload'
    model: Optional[str] = None,  # Mapped from 'model'
    api_key: Optional[str] = None,  # Mapped from 'api_key'
    system_message: Optional[str] = None,  # Mapped from 'system_message'
    temp: Optional[float] = None,  # Mapped from 'temp' (temperature)
    maxp: Optional[float] = None,  # Mapped from 'maxp' (top_p)
    streaming: Optional[bool] = False,  # Mapped from 'streaming'
    # New OpenAI specific parameters (and some from original ChatCompletionRequest schema)
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[Dict[str, float]] = None,
    logprobs: Optional[bool] = None,  # True/False
    top_logprobs: Optional[int] = None,
    max_tokens: Optional[
        int
    ] = None,  # This was already implicitly handled by config, now explicit
    n: Optional[int] = None,  # Number of completions
    presence_penalty: Optional[float] = None,
    response_format: Optional[Dict[str, str]] = None,  # e.g., {"type": "json_object"}
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    user: Optional[str] = None,  # This is the 'user_identifier' mapped
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    verbosity: Optional[str] = None,
    custom_prompt_arg: Optional[str] = None,  # Legacy
    api_base_url: Optional[str] = None,
):
    """
    Sends a chat completion request to the OpenAI API.

    Args:
        input_data: List of message objects (OpenAI format).
        model: ID of the model to use.
        api_key: OpenAI API key.
        system_message: Optional system message to prepend.
        temp: Sampling temperature.
        maxp: Top-p (nucleus) sampling parameter.
        streaming: Whether to stream the response.
        frequency_penalty: Penalizes new tokens based on their existing frequency.
        logit_bias: Modifies the likelihood of specified tokens appearing.
        logprobs: Whether to return log probabilities of output tokens.
        top_logprobs: An integer between 0 and 5 specifying the number of most likely tokens to return at each token position.
        max_tokens: Maximum number of tokens to generate.
        n: How many chat completion choices to generate for each input message.
        presence_penalty: Penalizes new tokens based on whether they appear in the text so far.
        response_format: An object specifying the format that the model must output. e.g. {"type": "json_object"}.
        seed: This feature is in Beta. If specified, the system will make a best effort to sample deterministically.
        stop: Up to 4 sequences where the API will stop generating further tokens.
        tools: A list of tools the model may call.
        tool_choice: Controls which (if any) function is called by the model.
        user: A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
        reasoning_effort: Uses the Responses API for supported models; GPT-5.6
            keeps an explicit ``none`` effort on Chat Completions for
            non-reasoning compatibility.
        reasoning_summary: Responses API reasoning summary detail for supported models.
        verbosity: Responses API text verbosity for GPT-5-style models.
        custom_prompt_arg: Legacy, largely ignored.
    """
    loaded_config_data = load_settings()
    legacy_openai_config = loaded_config_data.get("openai_api", {})
    api_settings = loaded_config_data.get("api_settings", {})
    canonical_openai_config = (
        api_settings.get("openai", {}) if isinstance(api_settings, Mapping) else {}
    )
    openai_config = (
        dict(legacy_openai_config) if isinstance(legacy_openai_config, Mapping) else {}
    )
    if isinstance(canonical_openai_config, Mapping):
        # Speech Settings owns the canonical credential, while the canonical
        # provider table may also contain defaults for unrelated chat axes.
        # Overlay only connection-owned values so moving the credential does
        # not silently change established model or sampling behavior.
        for key in ("api_key", "api_base_url"):
            if key in canonical_openai_config:
                openai_config[key] = canonical_openai_config[key]
    if not openai_config.get("api_key") and isinstance(legacy_openai_config, Mapping):
        # The legacy projection resolves environment-backed credentials.
        # Keep that resolved value when the canonical table intentionally
        # stores only api_key_env_var or an empty local fallback.
        openai_config["api_key"] = legacy_openai_config.get("api_key")

    final_api_key = api_key or openai_config.get("api_key")
    if not final_api_key:
        logger.error("OpenAI: API key is missing.")
        raise ChatConfigurationError(
            provider="openai", message="OpenAI API Key is required but not found."
        )

    logger.debug("OpenAI: API key provided.")

    # Resolve parameters: User-provided > Function arg default > Config default > Hardcoded default
    final_model = (
        model if model is not None else openai_config.get("model", "gpt-5.6-terra")
    )
    final_temp = (
        temp if temp is not None else float(openai_config.get("temperature", 0.7))
    )
    final_top_p = (
        maxp if maxp is not None else float(openai_config.get("top_p", 0.95))
    )  # 'maxp' from chat_api_call maps to 'top_p'

    final_streaming_cfg = openai_config.get("streaming", False)
    final_streaming = (
        streaming
        if streaming is not None
        else (
            str(final_streaming_cfg).lower() == "true"
            if isinstance(final_streaming_cfg, str)
            else bool(final_streaming_cfg)
        )
    )

    final_max_tokens = (
        max_tokens
        if max_tokens is not None
        else _safe_cast(openai_config.get("max_tokens"), int)
    )

    if custom_prompt_arg:
        logger.warning(
            "OpenAI: 'custom_prompt_arg' was provided but is generally ignored if 'input_data' and 'system_message' are used correctly."
        )

    # Construct messages for OpenAI API
    api_messages = []
    has_system_message_in_input = any(msg.get("role") == "system" for msg in input_data)
    if system_message and not has_system_message_in_input:
        api_messages.append({"role": "system", "content": system_message})
    api_messages.extend(input_data)

    normalized_reasoning_effort = _normalize_openai_reasoning_effort(reasoning_effort)
    is_gpt_5_6_model = _is_openai_gpt_5_6_model(final_model)
    use_responses_api = _openai_use_responses_api(
        normalized_reasoning_effort,
        reasoning_summary,
        verbosity,
        is_gpt_5_6_model,
    )
    payload = {
        "model": final_model,
        "stream": final_streaming,
    }
    if use_responses_api:
        payload["input"] = api_messages
    else:
        payload["messages"] = api_messages
    if final_streaming and not use_responses_api:
        payload["stream_options"] = {"include_usage": True}
    # Add optional parameters if they have a value. Reasoning-family models
    # (and therefore every Responses-API request, which this handler only
    # builds for reasoning params) reject non-default temperature/top_p with
    # HTTP 400 (value-level: the default is accepted, so omitting is always
    # safe), so the config-backed defaults must not be injected there
    # (task-404). The per-model fact is a `model_capabilities` predicate, not
    # a hand-maintained name list, so a new release in a covered family never
    # needs a marker added here (TASK-18803).
    omit_sampling_params = use_responses_api or openai_model_rejects_sampling_params(
        final_model
    )
    if omit_sampling_params:
        if temp is not None or maxp is not None:
            logger.warning(
                "OpenAI: dropping explicit temperature/top_p for reasoning "
                f"model '{final_model}' — the API rejects them."
            )
    else:
        if final_temp is not None:
            payload["temperature"] = final_temp
        if final_top_p is not None:
            payload["top_p"] = final_top_p  # OpenAI uses top_p
    if final_max_tokens is not None and use_responses_api:
        payload["max_output_tokens"] = final_max_tokens
    elif final_max_tokens is not None and openai_model_requires_max_completion_tokens(
        final_model
    ):
        # Modern chat-completions surface: `max_tokens` is HTTP 400
        # `unsupported_parameter` for these families (probe-verified with the
        # exact built gpt-5 payload, TASK-18803) -- gpt-5/o-series with no
        # reasoning effort configured used to fall through to `max_tokens`
        # here because only gpt-5.6 was special-cased.
        payload["max_completion_tokens"] = final_max_tokens
    elif final_max_tokens is not None:
        payload["max_tokens"] = final_max_tokens
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty
    if logit_bias is not None:
        payload["logit_bias"] = logit_bias
    if logprobs is not None:
        payload["logprobs"] = logprobs
    if top_logprobs is not None and payload.get("logprobs") is True:
        payload["top_logprobs"] = top_logprobs
    elif top_logprobs is not None:
        logger.warning(
            "OpenAI: 'top_logprobs' provided but 'logprobs' is not true. 'top_logprobs' will be ignored."
        )
    if n is not None:
        payload["n"] = n
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if response_format is not None:
        if use_responses_api:
            payload.setdefault("text", {})["format"] = response_format
        else:
            payload["response_format"] = response_format
    if seed is not None:
        payload["seed"] = seed
    if stop is not None:
        payload["stop"] = stop
    if tools is not None:
        payload["tools"] = tools
    if use_responses_api:
        reasoning_options = {}
        if normalized_reasoning_effort is not None:
            reasoning_options["effort"] = normalized_reasoning_effort
        if _is_present_setting(reasoning_summary):
            summary_value = str(reasoning_summary).strip().lower()
            if summary_value != "none":
                reasoning_options["summary"] = summary_value
        if reasoning_options:
            payload["reasoning"] = reasoning_options
        if _is_present_setting(verbosity):
            payload.setdefault("text", {})["verbosity"] = str(verbosity).strip().lower()
    elif is_gpt_5_6_model:
        payload["reasoning_effort"] = normalized_reasoning_effort or "none"

    # Then conditionally add tool_choice:
    if payload.get("tools") and tool_choice is not None:
        payload["tool_choice"] = tool_choice
    elif tool_choice == "none":  # Allow "none" even if no tools are present
        payload["tool_choice"] = "none"
    if user is not None:
        payload["user"] = user  # 'user' is OpenAI's user identifier field

    headers = {
        "Authorization": f"Bearer {final_api_key}",
        "Content-Type": "application/json",
    }
    if not is_sensitive_llm_request():
        # task-2116: this now actually interpolates (it used to be a plain
        # string missing its `f` prefix, so it silently logged literal
        # template text). Skipped entirely for sensitive/auxiliary requests
        # -- the payload can carry a caller-supplied system prompt or other
        # request content that must never reach a log in that context (see
        # Tests/Chat/test_sensitive_llm_logging.py).
        # task-2117 Qodo round: an allowlisted summary, not a denylist -- the
        # Responses API puts the WHOLE conversation under "input", which the
        # old "excluding messages" denylist never accounted for.
        logger.debug(
            "OpenAI Request Payload (safe fields only): "
            f"{safe_llm_request_payload_summary(payload, content_keys=('input', 'messages'))}"
        )

    api_path = "/responses" if use_responses_api else "/chat/completions"
    api_url = (
        api_base_url
        or openai_config.get("api_base_url")
        or builtin_provider_endpoint("openai", openai_config)
    ).rstrip("/") + api_path

    start_time = time.time()
    log_counter(
        "openai_api_request",
        labels={"model": final_model, "streaming": str(final_streaming)},
    )

    try:
        if final_streaming:
            logger.debug("OpenAI: Posting request (streaming)")

            def stream_generator():
                session_context = create_default_session()
                session = session_context.__enter__()
                response = None
                try:
                    response = session.post(
                        api_url, headers=headers, json=payload, stream=True, timeout=180
                    )
                    if (
                        response.status_code == 400
                        and "stream_options" in payload
                        and "stream_options" in (response.text or "")
                    ):
                        logger.warning(
                            "OpenAI: endpoint rejected stream_options; retrying without usage reporting."
                        )
                        retry_payload = {
                            k: v for k, v in payload.items() if k != "stream_options"
                        }
                        response = session.post(
                            api_url,
                            headers=headers,
                            json=retry_payload,
                            stream=True,
                            timeout=180,
                        )
                    response.raise_for_status()
                    if use_responses_api:
                        yield from _responses_stream_to_chat_sse(
                            response, model=final_model
                        )
                        return
                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.strip():
                            # Pass through OpenAI's SSE lines directly.
                            # Ensure they end with \n\n if not already.
                            # OpenAI's SSE usually includes double newlines.
                            yield line if line.endswith("\n") else line + "\n"
                except requests.exceptions.RequestException as e_request:
                    logger.opt(exception=True).error(
                        f"OpenAI: RequestException during stream: {e_request}"
                    )
                    error_content = json.dumps(
                        {
                            "error": {
                                "message": f"Stream connection error: {str(e_request)}",
                                "type": "openai_stream_error",
                            }
                        }
                    )
                    yield f"data: {error_content}\n\n"  # Yield as SSE error
                except Exception as e_stream:
                    logger.opt(exception=True).error(
                        f"OpenAI: Error during stream iteration: {e_stream}"
                    )
                    error_content = json.dumps(
                        {
                            "error": {
                                "message": f"Stream iteration error: {str(e_stream)}",
                                "type": "openai_stream_error",
                            }
                        }
                    )
                    yield f"data: {error_content}\n\n"  # Yield as SSE error
                finally:
                    # Ensure DONE is sent for the endpoint wrapper's logic
                    if not use_responses_api:
                        yield "data: [DONE]\n\n"
                    if response:
                        response.close()
                    session_context.__exit__(None, None, None)

            return stream_generator()

        else:  # Non-streaming
            logger.debug("OpenAI: Posting request (non-streaming)")
            retry_count = int(openai_config.get("api_retries", 3))
            retry_delay = float(
                openai_config.get("api_retry_delay", 1.0)
            )  # Ensure float

            retry_strategy = Retry(
                total=llm_retry_count(retry_count),
                backoff_factor=retry_delay,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST"],  # Changed from method_whitelist
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            with create_default_session() as session:
                session.mount("https://", adapter)
                session.mount("http://", adapter)  # Though OpenAI is https
                response = session.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=float(openai_config.get("api_timeout", 90.0)),
                )

            logger.debug(f"OpenAI: Full API response status: {response.status_code}")
            response.raise_for_status()  # Raise HTTPError for 4xx/5xx AFTER retries
            response_data = response.json()

            # Log success metrics
            duration = time.time() - start_time
            log_histogram(
                "openai_api_response_time",
                duration,
                labels={
                    "model": final_model,
                    "streaming": "false",
                    "status_code": str(response.status_code),
                },
            )
            log_counter(
                "openai_api_success",
                labels={"model": final_model, "streaming": "false"},
            )

            # Log token usage if available
            usage = response_data.get("usage", {})
            if usage:
                log_histogram(
                    "openai_api_prompt_tokens",
                    usage.get("prompt_tokens", 0),
                    labels={"model": final_model},
                )
                log_histogram(
                    "openai_api_completion_tokens",
                    usage.get("completion_tokens", 0),
                    labels={"model": final_model},
                )
                log_histogram(
                    "openai_api_total_tokens",
                    usage.get("total_tokens", 0),
                    labels={"model": final_model},
                )
                log_histogram(
                    "openai_api_cached_tokens",
                    (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    or 0,
                    labels={"model": final_model},
                )

            logger.debug("OpenAI: Non-streaming request successful.")
            if use_responses_api:
                return _normalize_openai_responses_payload(
                    response_data, model=final_model
                )
            return response_data

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else 0

        # Log error metrics
        duration = time.time() - start_time
        log_counter(
            "openai_api_error",
            labels={
                "model": final_model,
                "error_type": "http_error",
                "status_code": str(status_code),
            },
        )
        log_histogram(
            "openai_api_error_response_time",
            duration,
            labels={"model": final_model, "status_code": str(status_code)},
        )

        if e.response is not None:
            error_detail = str(safe_llm_error_detail(e.response.text))
            logger.error(
                "OpenAI request failed; "
                f"status={e.response.status_code}; detail={error_detail}"
            )
        else:
            logger.error(
                "OpenAI HTTPError with no response object; "
                f"error_type={safe_llm_exception_message(e)}"
            )
        raise
        # if e.response is not None:
        #     error_content_text = e.response.text
        #     try:
        #         error_content_json = e.response.json()
        #     except json.JSONDecodeError:
        #         pass
        # logger.error(
        #     f"OpenAI HTTPError {e.response.status_code if e.response is not None else 'Unknown'}. Text: {error_content_text}. JSON: {error_content_json}",
        #     exc_info=True)
        # raise
    except requests.exceptions.RequestException as e:
        # Log network error metrics
        duration = time.time() - start_time
        log_counter(
            "openai_api_error",
            labels={"model": final_model, "error_type": "network_error"},
        )
        log_histogram(
            "openai_api_error_response_time",
            duration,
            labels={"model": final_model, "error_type": "network"},
        )
        error_detail = safe_llm_exception_message(e)
        if is_sensitive_llm_request():
            logger.error(f"OpenAI RequestException: {error_detail}")
        else:
            logger.opt(exception=True).error(f"OpenAI RequestException: {error_detail}")
        raise
    except Exception as e:  # Catch any other unexpected error
        # Log unexpected error metrics
        duration = time.time() - start_time
        log_counter(
            "openai_api_error",
            labels={"model": final_model, "error_type": "unexpected"},
        )
        error_detail = safe_llm_exception_message(e)
        error_copy = f"OpenAI: Unexpected error: {error_detail}"
        if is_sensitive_llm_request():
            logger.error(error_copy)
        else:
            logger.opt(exception=True).error(error_copy)
        raise ChatProviderError(
            provider="openai", message=f"Unexpected error: {error_detail}"
        )


def _anthropic_block_index(event: dict) -> int | None:
    """Best-effort parse of an SSE event's content-block ``index``.

    Anthropic always sends an int, but a malformed event must not abort an
    otherwise-valid stream (PR #659 review): non-int-castable values yield
    None and the caller skips the event.

    Args:
        event: A decoded Anthropic SSE event payload.

    Returns:
        The block index, or None when absent/unparseable.
    """
    raw = event.get("index", 0)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    return None


def _anthropic_supports_caching(model: str) -> bool:
    """True for Claude models that support prompt caching (``cache_control``).

    All modern Claude (3 / 3.5 / 3.7 / 4+) support it; legacy ``claude-2*`` and
    ``claude-instant*`` do not.

    Args:
        model: The model identifier.

    Returns:
        True when the model accepts ``cache_control`` breakpoints.
    """
    m = (model or "").lower()
    return (
        m.startswith("claude-") and not m.startswith("claude-2") and "instant" not in m
    )


def _anthropic_caching_enabled() -> bool:
    """[caching].anthropic_enabled kill-switch for ALL Anthropic cache_control.

    Defaults to True when the section or key is absent (prompt caching is the
    shipped task-323 behavior; this gate only adds an opt-out). Any config
    read failure also defaults to True so a broken config file cannot
    silently change request shapes.

    Returns:
        True when cache_control breakpoints should be emitted.
    """
    try:
        return bool(get_cli_setting("caching", "anthropic_enabled", True))
    except Exception as exc:
        logger.warning(
            f"caching config read failed; defaulting anthropic prompt caching ON: {exc!r}"
        )
        return True


def _without_cache_control(obj: Any) -> Any:
    """Deep-copy ``obj`` with every ``cache_control`` key removed.

    Args:
        obj: Any JSON-shaped structure (dicts/lists/scalars).

    Returns:
        The same structure minus all ``cache_control`` entries.
    """
    if isinstance(obj, dict):
        return {
            key: _without_cache_control(value)
            for key, value in obj.items()
            if key != "cache_control"
        }
    if isinstance(obj, list):
        return [_without_cache_control(item) for item in obj]
    return obj


def _contains_cache_control(obj: Any) -> bool:
    """True when any nested dict carries a ``cache_control`` key.

    Args:
        obj: Any JSON-shaped structure (dicts/lists/scalars).

    Returns:
        True when ``cache_control`` appears anywhere within ``obj``.
    """
    if isinstance(obj, dict):
        return "cache_control" in obj or any(
            _contains_cache_control(value) for value in obj.values()
        )
    if isinstance(obj, list):
        return any(_contains_cache_control(item) for item in obj)
    return False


def _anthropic_tools_payload(tools: list) -> list:
    """Convert OpenAI function-format tool entries to Anthropic's format.

    Valid entries already in Anthropic shape are copied through with their
    native fields preserved. Malformed entries are dropped with a bounded
    diagnostic.

    Args:
        tools: The ``tools`` list as received (OpenAI or Anthropic shaped).

    Returns:
        Anthropic-format entries: ``{"name", "description", "input_schema"}``.
    """
    converted = []
    for entry in tools or []:
        if not isinstance(entry, dict):
            logger.warning(
                "Anthropic: dropping invalid tools entry (expected a mapping)."
            )
            continue
        function = entry.get("function")
        if entry.get("type") == "function" and isinstance(function, dict):
            name = str(function.get("name") or "").strip()
            if not name:
                # Anthropic rejects empty tool names — dropping the entry
                # keeps the failure local instead of a provider 400
                # (PR #659 review).
                continue
            parameters = function.get("parameters")
            if not isinstance(parameters, dict) or not parameters:
                parameters = {"type": "object", "properties": {}}
            converted.append(
                {
                    "name": name,
                    "description": str(function.get("description") or ""),
                    "input_schema": parameters,
                }
            )
        elif (
            isinstance(entry.get("name"), str)
            and entry["name"].strip()
            and isinstance(entry.get("input_schema"), dict)
        ):
            converted.append(dict(entry))
        else:
            logger.warning(
                "Anthropic: dropping invalid tools entry "
                "(expected an OpenAI function tool or Anthropic native tool)."
            )
    return converted


def chat_with_anthropic(
    input_data: List[Dict[str, Any]],  # Mapped from 'messages_payload'
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,  # Mapped from 'system_message'
    temp: Optional[float] = None,
    topp: Optional[float] = None,  # Mapped from 'topp' (becomes top_p)
    topk: Optional[int] = None,
    streaming: Optional[bool] = False,
    max_tokens: Optional[int] = None,  # New: Anthropic uses 'max_tokens'
    stop_sequences: Optional[List[str]] = None,  # New: Mapped from 'stop'
    tools: Optional[List[Dict[str, Any]]] = None,  # New: Anthropic tool format
    thinking_effort: Optional[str] = None,
    thinking_budget_tokens: Optional[int] = None,
    prompt_caching: Optional[bool] = None,
    # Anthropic doesn't typically use seed, response_format (for JSON object mode directly), n, user identifier, logit_bias,
    # presence_penalty, frequency_penalty, logprobs, top_logprobs in the same way as OpenAI.
    # tool_choice is usually implicit with tools or controlled differently.
    custom_prompt_arg: Optional[str] = None,  # Legacy
    api_base_url: Optional[str] = None,
):
    """Call the Anthropic Messages API (streaming or non-streaming).

    Args:
        input_data: List of message objects (OpenAI-style ``role``/``content``
            dicts). ``role: "tool"`` entries are converted to Anthropic
            ``tool_result`` blocks; assistant ``tool_calls`` echoes are
            converted to ``tool_use`` blocks.
        model: Anthropic model ID. Falls back to config, then
            ``"claude-sonnet-5"``.
        api_key: Anthropic API key. Falls back to config; raises if still
            missing.
        system_prompt: Optional system prompt sent as the top-level
            ``system`` field.
        temp: Sampling temperature. Falls back to config, then ``0.7``.
        topp: Nucleus sampling parameter (Anthropic ``top_p``).
        topk: Anthropic ``top_k`` sampling parameter.
        streaming: Whether to stream the response via SSE. Falls back to
            config, then ``False``.
        max_tokens: Maximum tokens to generate. Falls back to config
            (``max_tokens_to_sample``/``max_tokens``), then ``4096``.
        stop_sequences: Custom stop sequences (Anthropic ``stop_sequences``).
        tools: Tools in OpenAI function-call or native Anthropic shape;
            normalized to Anthropic's ``{name, description, input_schema}``.
        thinking_effort: Extended-thinking effort level, when the model
            supports it (translated to a thinking token budget).
        thinking_budget_tokens: Explicit extended-thinking token budget;
            takes precedence over ``thinking_effort`` when both are set.
        prompt_caching: Opt-in for the PER-TURN message ``cache_control``
            breakpoint. Only multi-turn callers (the Console gateway) should
            pass True: the breakpoint bills the whole conversation prefix at
            the 1.25x cache-write premium, which a one-shot call (media
            summarization, websearch, evals, document generation) can never
            earn back because it never sends a second turn. The system and
            last-tool breakpoints are NOT gated on this flag — those shipped
            provider-wide in task-323 and are harmless for one-shots, whose
            short system prefix falls below the cacheable minimum and is
            simply not cached. Also subject to the provider-wide
            ``[caching].anthropic_enabled`` kill-switch
            (``_anthropic_caching_enabled()``), which disables ALL
            ``cache_control`` breakpoints regardless of this flag.
        custom_prompt_arg: Legacy/unused prompt override, retained for call
            signature compatibility.

    Returns:
        Non-streaming: a normalized OpenAI-style response dict (``choices``
        with ``message``, ``usage``, etc.) built from the Anthropic Messages
        API response. Streaming: a generator yielding OpenAI-style SSE
        strings (``"data: {...}\\n\\n"``, terminated by ``"data: [DONE]\\n\\n"``).

    Raises:
        ChatConfigurationError: No API key is available (argument or config).
        ChatBadRequestError: No valid user message could be built from
            ``input_data``, or the API returned a 4xx error other than 401/429.
        ChatAuthenticationError: The API returned 401 (invalid/expired key).
        ChatRateLimitError: The API returned 429 (rate limited).
        ChatProviderError: Any other HTTP error status, a network-level
            request failure, or an unexpected exception while calling the
            API.
    """
    # Assuming load_settings is defined elsewhere
    loaded_config_data = load_settings()
    anthropic_config = loaded_config_data.get("anthropic_api", {})
    final_api_key = api_key or anthropic_config.get("api_key")
    if not final_api_key:
        raise ChatConfigurationError(
            provider="anthropic", message="Anthropic API Key is required."
        )

    logger.debug("Anthropic: API key provided.")

    current_model = model or anthropic_config.get("model", "claude-sonnet-5")
    default_temperature = float(anthropic_config.get("temperature", 0.7))
    current_temp = temp if temp is not None else default_temperature
    current_top_p = topp
    current_top_k = topk
    current_streaming_cfg = anthropic_config.get("streaming", False)
    current_streaming = (
        streaming
        if streaming is not None
        else (
            str(current_streaming_cfg).lower() == "true"
            if isinstance(current_streaming_cfg, str)
            else bool(current_streaming_cfg)
        )
    )

    # Use the passed max_tokens if available, else config, else a default
    default_max_tokens = int(
        anthropic_config.get(
            "max_tokens_to_sample", anthropic_config.get("max_tokens", 4096)
        )
    )
    current_max_tokens = max_tokens if max_tokens is not None else default_max_tokens
    thinking_config, output_config, current_max_tokens = _anthropic_thinking_config(
        model=current_model,
        thinking_effort=thinking_effort,
        thinking_budget_tokens=thinking_budget_tokens,
        max_tokens=current_max_tokens,
    )

    anthropic_messages = []
    from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY

    for msg in input_data:
        role = msg.get("role")
        content = msg.get("content")
        if role == "tool":
            # OpenAI tool-result convention -> Anthropic tool_result block.
            # Consecutive tool results coalesce into ONE user turn: they all
            # answer the same assistant tool_use turn, and Anthropic requires
            # alternating roles (task-263 AC#2).
            block = {
                "type": "tool_result",
                "tool_use_id": str(msg.get("tool_call_id") or ""),
                "content": str(content or ""),
            }
            last = anthropic_messages[-1] if anthropic_messages else None
            if (
                last is not None
                and last.get("role") == "user"
                and isinstance(last.get("content"), list)
                and any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in last["content"]
                )
            ):
                last["content"].append(block)
            else:
                anthropic_messages.append({"role": "user", "content": [block]})
            continue
        if role == "user" and msg.get(EPHEMERAL_ORIGIN_KEY) == "project_instructions":
            text_block = {"type": "text", "text": str(content or "")}
            last = anthropic_messages[-1] if anthropic_messages else None
            if (
                last is not None
                and last.get("role") == "user"
                and isinstance(last.get("content"), list)
                and any(
                    isinstance(block, dict) and block.get("type") == "tool_result"
                    for block in last["content"]
                )
            ):
                last["content"].append(text_block)
            else:
                anthropic_messages.append(
                    {"role": "user", "content": [text_block]}
                )
            continue
        if role == "assistant" and msg.get("tool_calls"):
            # OpenAI assistant tool_calls echo -> Anthropic tool_use blocks
            # (text block first when the turn also carried visible content).
            # Guards mirror native_tools.parse_native_tool_calls: the live
            # Anthropic API rejects both an empty "content": [] array and a
            # tool_use block with an empty "name", so a call only converts
            # when it has a dict `function` with a non-empty stripped
            # `name` (task-263 review). Build the candidate blocks first —
            # if every call is junk, fall through to the plain content
            # handling below instead of sending a blocks-only message.
            tool_use_blocks = []
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                name = str(function.get("name") or "").strip()
                if not name:
                    continue
                raw_args = function.get("arguments")
                tool_input = raw_args if isinstance(raw_args, dict) else {}
                if isinstance(raw_args, str) and raw_args.strip():
                    try:
                        parsed = json.loads(raw_args)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict):
                        tool_input = parsed
                tool_use_blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(call.get("id") or ""),
                        "name": name,
                        "input": tool_input,
                    }
                )
            if tool_use_blocks:
                blocks = []
                if isinstance(content, str) and content.strip():
                    blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    # List-form (multimodal) content: keep its text parts —
                    # dropping them would silently lose visible text that
                    # accompanied the tool calls (PR #659 review).
                    for part in content:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "text"
                            and isinstance(part.get("text"), str)
                            and part["text"].strip()
                        ):
                            blocks.append({"type": "text", "text": part["text"]})
                blocks.extend(tool_use_blocks)
                anthropic_messages.append({"role": "assistant", "content": blocks})
                continue
            # else: no valid tool_use blocks survived the guards above —
            # fall through to the plain user/assistant content handling.
        if role not in ["user", "assistant"]:
            logger.warning(f"Anthropic: Skipping message with unsupported role: {role}")
            continue
        # ... (multimodal content processing for Anthropic from your existing function) ...
        anthropic_content_parts = []
        if isinstance(content, str):
            anthropic_content_parts.append({"type": "text", "text": content})
        elif isinstance(content, list):  # OpenAI content part list
            for part in content:
                part_type = part.get("type")
                if part_type == "text":
                    anthropic_content_parts.append(
                        {"type": "text", "text": part.get("text", "")}
                    )
                elif part_type == "image_url":
                    image_url_obj = part.get("image_url", {})
                    url_str = image_url_obj.get("url", "")
                    parsed_image = _parse_data_url_for_multimodal(url_str)
                    if parsed_image:
                        mime_type, b64_data = parsed_image
                        anthropic_content_parts.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": b64_data,
                                },
                            }
                        )
        if anthropic_content_parts:
            anthropic_messages.append(
                {"role": role, "content": anthropic_content_parts}
            )

    if not any(m["role"] == "user" for m in anthropic_messages):
        raise ChatBadRequestError(
            provider="anthropic", message="No valid user messages found for Anthropic."
        )

    headers = {
        "x-api-key": final_api_key,
        "anthropic-version": anthropic_config.get("api_version", "2023-06-01"),
        "Content-Type": "application/json",
    }
    caching_active = (
        _anthropic_supports_caching(current_model) and _anthropic_caching_enabled()
    )
    data = {
        "model": current_model,
        "max_tokens": current_max_tokens,  # Changed from max_tokens_to_sample to the parameter
        "messages": anthropic_messages,
        "stream": current_streaming,
    }
    if system_prompt is not None:
        if caching_active and system_prompt:
            # cache_control on the system prompt (the largest stable prefix)
            # activates Anthropic prompt caching; per the tools->system->messages
            # hierarchy this caches tools+system. Applied for both streaming and
            # non-streaming (the payload is built before the streaming branch).
            data["system"] = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            data["system"] = system_prompt  # unchanged for non-caching models
    # Sampling parameters are suppressed for two independent reasons: the model
    # rejects them outright (a provider capability -- 400 on Fable 5, Mythos 5,
    # Opus 5, Opus 4.8, Opus 4.7 and Sonnet 5), or thinking is enabled for this
    # request. The capability check must not be conditioned on the thinking
    # config: Opus 4.8/4.7 produce no thinking config when no effort is
    # configured, which used to reopen this branch (TASK-18414 AC #5).
    model_rejects_sampling = anthropic_model_rejects_sampling_params(current_model)
    if not model_rejects_sampling and thinking_config is None:
        if temp is not None:
            data["temperature"] = current_temp
            if current_top_p is not None:
                logger.warning(
                    "Anthropic: both temperature and top_p were provided; sending temperature and dropping top_p."
                )
        elif current_top_p is not None:
            data["top_p"] = current_top_p
        else:
            data["temperature"] = current_temp
        if current_top_k is not None:
            data["top_k"] = current_top_k
    elif any(value is not None for value in (temp, current_top_p, current_top_k)):
        if model_rejects_sampling:
            logger.warning(
                "Anthropic: omitting temperature/top_p/top_k because model %s "
                "rejects sampling parameters.",
                current_model,
            )
        else:
            logger.warning(
                "Anthropic: omitting temperature/top_p/top_k because thinking is enabled."
            )
    if stop_sequences is not None:
        data["stop_sequences"] = stop_sequences
    if tools is not None:
        tools_payload = _anthropic_tools_payload(tools)
        if caching_active and tools_payload:
            # Optional second breakpoint on the last converted tool. A fresh dict
            # so the caller's input `tools` are never mutated.
            tools_payload[-1] = {
                **tools_payload[-1],
                "cache_control": {"type": "ephemeral"},
            }
        data["tools"] = tools_payload
    if thinking_config is not None:
        data["thinking"] = thinking_config
    if output_config is not None:
        data["output_config"] = output_config

    if (
        caching_active
        and prompt_caching
        and anthropic_messages
        and isinstance(anthropic_messages[-1].get("content"), list)
        and anthropic_messages[-1]["content"]
    ):
        # Per-turn breakpoint (cost-ticker PR2): mark the last content block
        # of the final message so the WHOLE conversation prefix becomes a
        # reusable cache entry next turn -- the task-323 system/tools
        # breakpoints alone never cache message history. Budget:
        # system(1) + last-tool(1) + this(1) = 3 of the 4 allowed.
        #
        # OPT-IN (`prompt_caching`), unlike the two breakpoints above: this
        # one bills the entire conversation prefix at the 1.25x write
        # premium, so a one-shot caller pays ~25% extra input and can never
        # read the entry back. Only the Console gateway (multi-turn by
        # construction) sets the flag; every other caller of
        # `chat_with_anthropic` leaves it None and is unaffected.
        #
        # Fresh dict so no caller-held block is mutated (same rule as the
        # tools breakpoint above).
        last_content = anthropic_messages[-1]["content"]
        last_content[-1] = {
            **last_content[-1],
            "cache_control": {"type": "ephemeral"},
        }

    api_url = (
        api_base_url
        or anthropic_config.get("api_base_url")
        or builtin_provider_endpoint("anthropic", anthropic_config)
    ).rstrip("/") + "/messages"
    if not is_sensitive_llm_request():
        # task-2116: see the OpenAI branch above for why this is gated.
        # task-2117 Qodo round: allowlisted summary -- "system" carries the
        # actual system-prompt text and must never be logged verbatim.
        logger.debug(
            "Anthropic Request Payload (safe fields only): "
            f"{safe_llm_request_payload_summary(data, system_keys=('system',))}"
        )

    start_time = time.time()
    log_counter(
        "anthropic_api_request",
        labels={"model": current_model, "streaming": str(current_streaming)},
    )

    try:
        retry_count = int(anthropic_config.get("api_retries", 3))
        retry_delay = float(anthropic_config.get("api_retry_delay", 1))
        retry_strategy = Retry(
            total=llm_retry_count(retry_count),
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        with create_default_session() as session:
            session.mount("https://", adapter)
            response = session.post(
                api_url,
                headers=headers,
                json=data,
                stream=current_streaming,
                timeout=180,
                # task-19557: the API key travels in the custom `x-api-key`
                # header. `requests` strips `Authorization` across a
                # redirect host change but NOT custom headers, so a 3xx
                # from this endpoint would re-send the key wherever
                # `Location` points. Refuse to follow rather than silently
                # forward credentials -- see the explicit 3xx check below.
                # Mirrors the `x-goog-api-key` fix in the Google branch
                # (task-686/chat_with_google).
                allow_redirects=False,
            )
            if (
                response.status_code == 400
                and _contains_cache_control(data)
                and "cache_control" in (response.text or "")
            ):
                # Caching must never break sends (cost-ticker PR2): odd
                # proxies/gateways can reject cache_control. Retry exactly
                # once without any breakpoints; every other error path is
                # untouched. Reading .text here is safe -- it is the error
                # body, not a stream.
                logger.warning(
                    "Anthropic: endpoint rejected cache_control; retrying without prompt caching."
                )
                log_counter(
                    "anthropic_cache_control_degrade",
                    labels={"model": current_model},
                )
                response = session.post(
                    api_url,
                    headers=headers,
                    json=_without_cache_control(data),
                    stream=current_streaming,
                    timeout=180,
                    allow_redirects=False,
                )
            if 300 <= response.status_code < 400:
                # No new logging call here, deliberately: this file's
                # diagnostic call sites participate in the pinned
                # cross-file inventory that
                # test_summarization_diagnostic_privacy.py's
                # "manifest_boundary" enforces (task-3796/TASK-492). The
                # raised exception's message is the caller-visible signal;
                # it deliberately omits the redirect target -- a 3xx
                # `Location` is server/attacker-controlled data, same
                # reasoning as the Google branch above never echoing it in
                # the raised message (only its own already-reviewed log
                # line does).
                response.close()
                raise ChatProviderError(
                    provider="anthropic",
                    message=(
                        "Anthropic endpoint redirected unexpectedly -- refusing to "
                        "follow with credentials."
                    ),
                    status_code=response.status_code,
                )
            response.raise_for_status()

        if current_streaming:
            logger.debug(
                "Anthropic: Streaming response received. Normalizing to OpenAI SSE."
            )

            def stream_generator():
                completion_id = f"chatcmpl-anthropic-{time.time_ns()}"
                int(time.time())

                created_ts = int(time.time())
                # model_name = current_model # Defined outside this generator
                # Note: Anthropic event types: message_start, content_block_start, content_block_delta, content_block_stop, message_delta, message_stop
                # We primarily care about content_block_delta for text and message_delta/message_stop for finish_reason.

                # task-263: map Anthropic tool_use content-block indexes to
                # 0-based OpenAI tool_calls positions (Anthropic's index also
                # counts text blocks; OpenAI consumers key fragments by
                # tool-call position — see the gateway's _ToolCallAccumulator).
                tool_call_positions = {}
                next_tool_position = 0

                usage_accumulator: dict = {}
                output_captured = False

                def _usage_sse_chunk(usage: dict | None = None) -> str:
                    sse_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": current_model,
                        "choices": [],
                        "usage": dict(
                            usage if usage is not None else usage_accumulator
                        ),
                    }
                    return f"data: {json.dumps(sse_chunk)}\n\n"

                try:
                    for line_bytes in response.iter_lines():  # iter_lines gives bytes
                        line = line_bytes.decode("utf-8").strip()
                        if not line:
                            continue  # Skip keep-alive newlines
                        # Anthropic SSE has "event:" and "data:" lines
                        # Parse them and reformat
                        # Example (simplified, actual Anthropic events are more complex):
                        if line.startswith("event:"):
                            # event_name = line[len("event:"):].strip() # Store event name if needed
                            pass  # We'll parse the data line
                        elif line.startswith("data:"):
                            event_data_str = line[len("data:") :].strip()
                            try:
                                anthropic_event = json.loads(event_data_str)
                                delta_content = None
                                finish_reason = None
                                tool_calls_delta = None  # For future tool streaming

                                if anthropic_event.get("type") == "message_start":
                                    message_obj = anthropic_event.get("message")
                                    start_usage = (
                                        message_obj.get("usage")
                                        if isinstance(message_obj, dict)
                                        else None
                                    )
                                    if isinstance(start_usage, dict) and start_usage:
                                        usage_accumulator.update(start_usage)
                                        # message_start's usage always carries a
                                        # small placeholder output_tokens value
                                        # (Anthropic bills it before generation
                                        # starts) -- never surface it as if it
                                        # were authoritative output usage.
                                        input_usage = {
                                            k: v
                                            for k, v in start_usage.items()
                                            if k != "output_tokens"
                                        }
                                        if input_usage:
                                            yield _usage_sse_chunk(input_usage)
                                    continue

                                if anthropic_event.get("type") == "content_block_start":
                                    block = anthropic_event.get("content_block") or {}
                                    if block.get("type") == "tool_use":
                                        index = _anthropic_block_index(anthropic_event)
                                        if index is None:
                                            continue
                                        position = next_tool_position
                                        next_tool_position += 1
                                        tool_call_positions[index] = position
                                        tool_calls_delta = [
                                            {
                                                "index": position,
                                                "id": str(block.get("id") or ""),
                                                "type": "function",
                                                "function": {
                                                    "name": str(
                                                        block.get("name") or ""
                                                    ),
                                                    "arguments": "",
                                                },
                                            }
                                        ]
                                elif (
                                    anthropic_event.get("type") == "content_block_delta"
                                ):
                                    delta = anthropic_event.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        delta_content = delta.get("text")
                                    elif delta.get("type") == "input_json_delta":
                                        index = _anthropic_block_index(anthropic_event)
                                        if index in tool_call_positions:
                                            tool_calls_delta = [
                                                {
                                                    "index": tool_call_positions[index],
                                                    "function": {
                                                        "arguments": delta.get(
                                                            "partial_json", ""
                                                        )
                                                    },
                                                }
                                            ]
                                elif anthropic_event.get("type") == "message_delta":
                                    finish_reason_anth = anthropic_event.get(
                                        "delta", {}
                                    ).get("stop_reason")
                                    delta_usage = anthropic_event.get("usage")
                                    if isinstance(delta_usage, dict):
                                        usage_accumulator.update(delta_usage)
                                        if "output_tokens" in delta_usage:
                                            output_captured = True
                                    if finish_reason_anth:
                                        finish_reason_map = {
                                            "end_turn": "stop",
                                            "max_tokens": "length",
                                            "stop_sequence": "stop",
                                            "tool_use": "tool_calls",
                                        }
                                        finish_reason = finish_reason_map.get(
                                            finish_reason_anth, finish_reason_anth
                                        )
                                # message_stop is the final event, might contain final usage metrics.
                                elif anthropic_event.get("type") == "message_stop":
                                    # This event confirms the end. If no explicit finish_reason was in message_delta,
                                    # the previous one (or lack thereof) stands.
                                    # It's a good place to emit the [DONE] signal.
                                    # logger.debug(f"Anthropic stream: message_stop received. Full event: {anthropic_event}")
                                    pass  # The [DONE] is yielded in finally or after loop.

                                sse_choice_payload = {}
                                if delta_content is not None:  # Can be empty string
                                    sse_choice_payload["delta"] = {
                                        "content": delta_content
                                    }
                                if tool_calls_delta:  # Placeholder for tool streaming
                                    if "delta" not in sse_choice_payload:
                                        sse_choice_payload["delta"] = {}
                                    sse_choice_payload["delta"]["tool_calls"] = (
                                        tool_calls_delta
                                    )
                                if finish_reason:
                                    sse_choice_payload["finish_reason"] = finish_reason

                                if (
                                    sse_choice_payload
                                ):  # If there's anything to send in choices
                                    sse_choice_payload["index"] = (
                                        0  # Standard for non-batched choices
                                    )
                                    sse_chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_ts,
                                        "model": current_model,
                                        "choices": [sse_choice_payload],
                                    }
                                    yield f"data: {json.dumps(sse_chunk)}\n\n"
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Anthropic Stream: Could not decode JSON: {event_data_str}"
                                )

                    if output_captured:
                        yield _usage_sse_chunk()
                except (
                    requests.exceptions.ChunkedEncodingError
                ) as e:  # ... error handling ...
                    logger.opt(exception=True).error(
                        f"Anthropic: ChunkedEncodingError during stream: {e}"
                    )
                    yield f"data: {json.dumps({'error': {'message': f'Stream connection error: {str(e)}', 'type': 'anthropic_stream_error'}})}\n\n"
                except Exception as e:  # ... error handling ...
                    logger.opt(exception=True).error(
                        f"Anthropic: Error during stream iteration: {e}"
                    )
                    yield f"data: {json.dumps({'error': {'message': f'Stream iteration error: {str(e)}', 'type': 'anthropic_stream_error'}})}\n\n"
                finally:
                    yield "data: [DONE]\n\n"
                    if response:
                        response.close()

            return stream_generator()
        else:
            # ... (non-streaming logic remains the same) ...
            logger.debug("Anthropic: Non-streaming request successful.")
            response_data = response.json()
            logger.debug(
                "Anthropic: Non-streaming request successful. Normalizing response."
            )
            assistant_content_parts = []
            if response_data.get("content"):
                for part in response_data.get("content", []):
                    if part.get("type") == "text":
                        assistant_content_parts.append(part.get("text", ""))
            full_assistant_content = "\n".join(assistant_content_parts).strip()
            tool_call_entries = []
            for part in response_data.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "tool_use":
                    tool_call_entries.append(
                        {
                            "id": str(part.get("id") or ""),
                            "type": "function",
                            "function": {
                                "name": str(part.get("name") or ""),
                                "arguments": json.dumps(part.get("input") or {}),
                            },
                        }
                    )
            finish_reason_map = {
                "end_turn": "stop",
                "max_tokens": "length",
                "stop_sequence": "stop",
                "tool_use": "tool_calls",
            }  # Added tool_use
            openai_finish_reason = finish_reason_map.get(
                response_data.get("stop_reason"), response_data.get("stop_reason")
            )
            if openai_finish_reason == "tool_calls" and not tool_call_entries:
                # stop_reason claimed tool_use but the body carried no
                # tool_use blocks — never emit the self-contradictory
                # finish_reason="tool_calls" with no message.tool_calls
                # (PR #659 review).
                openai_finish_reason = "stop"
            message_payload = {"role": "assistant", "content": full_assistant_content}
            if tool_call_entries:
                message_payload["tool_calls"] = tool_call_entries
            normalized_response = {
                "id": response_data.get("id", f"anthropic-{time.time_ns()}"),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": response_data.get("model", current_model),
                "choices": [
                    {
                        "index": 0,
                        "message": message_payload,
                        "finish_reason": openai_finish_reason,
                    }
                ],
                "usage": response_data.get("usage"),
            }

            # Log success metrics
            duration = time.time() - start_time
            log_histogram(
                "anthropic_api_response_time",
                duration,
                labels={
                    "model": current_model,
                    "streaming": "false",
                    "status_code": str(response.status_code),
                },
            )
            log_counter(
                "anthropic_api_success",
                labels={"model": current_model, "streaming": "false"},
            )

            # Log token usage if available
            usage = response_data.get("usage", {})
            if usage:
                log_histogram(
                    "anthropic_api_input_tokens",
                    usage.get("input_tokens", 0),
                    labels={"model": current_model},
                )
                log_histogram(
                    "anthropic_api_output_tokens",
                    usage.get("output_tokens", 0),
                    labels={"model": current_model},
                )
                total_tokens = usage.get("input_tokens", 0) + usage.get(
                    "output_tokens", 0
                )
                log_histogram(
                    "anthropic_api_total_tokens",
                    total_tokens,
                    labels={"model": current_model},
                )
                log_histogram(
                    "anthropic_api_cache_read_input_tokens",
                    usage.get("cache_read_input_tokens") or 0,
                    labels={"model": current_model},
                )
                log_histogram(
                    "anthropic_api_cache_creation_input_tokens",
                    usage.get("cache_creation_input_tokens") or 0,
                    labels={"model": current_model},
                )

            return normalized_response

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else 500
        raw_error_text = (
            e.response.text if e.response is not None else safe_llm_exception_message(e)
        )
        error_text = str(safe_llm_error_detail(raw_error_text))

        # Log error metrics
        duration = time.time() - start_time
        log_counter(
            "anthropic_api_error",
            labels={
                "model": current_model,
                "error_type": "http_error",
                "status_code": str(status_code),
            },
        )
        log_histogram(
            "anthropic_api_error_response_time",
            duration,
            labels={"model": current_model, "status_code": str(status_code)},
        )

        if status_code == 401:
            raise ChatAuthenticationError(
                provider="anthropic", message=f"Auth failed. Detail: {error_text[:200]}"
            ) from e
        elif status_code == 429:
            raise ChatRateLimitError(
                provider="anthropic", message=f"Rate limit. Detail: {error_text[:200]}"
            ) from e
        elif 400 <= status_code < 500:
            raise ChatBadRequestError(
                provider="anthropic",
                message=f"Bad request ({status_code}). Detail: {error_text[:200]}",
            ) from e
        else:
            raise ChatProviderError(
                provider="anthropic",
                message=f"API error ({status_code}). Detail: {error_text[:200]}",
                status_code=status_code,
            ) from e
    except requests.exceptions.RequestException as e:
        # Log network error metrics
        duration = time.time() - start_time
        log_counter(
            "anthropic_api_error",
            labels={"model": current_model, "error_type": "network_error"},
        )
        log_histogram(
            "anthropic_api_error_response_time",
            duration,
            labels={"model": current_model, "error_type": "network_error"},
        )
        error_text = safe_llm_exception_message(e)
        raise ChatProviderError(
            provider="anthropic",
            message=f"Network error: {error_text}",
            status_code=504,
        ) from e
    except Exception as e:
        # Log unexpected error metrics
        duration = time.time() - start_time
        log_counter(
            "anthropic_api_error",
            labels={"model": current_model, "error_type": "unexpected_error"},
        )
        log_histogram(
            "anthropic_api_error_response_time",
            duration,
            labels={"model": current_model, "error_type": "unexpected_error"},
        )
        error_text = safe_llm_exception_message(e)
        if is_sensitive_llm_request():
            logger.error(f"Anthropic: Unexpected error: {error_text}")
        else:
            logger.opt(exception=True).error(
                f"Anthropic: Unexpected error: {error_text}"
            )
        raise ChatProviderError(
            provider="anthropic", message=f"Unexpected error: {error_text}"
        ) from e


_COHERE_SUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "description",
        "required",
        "enum",
        "items",
        "anyOf",
        "additionalProperties",
    }
)


def _cohere_schema_projection(schema: dict) -> dict:
    """Copy only Cohere strict-tools' supported schema keywords."""
    raw_type = schema.get("type")
    type_union = raw_type if isinstance(raw_type, list) else None
    existing_any_of = schema.get("anyOf")
    projected: dict = {}
    for key, value in schema.items():
        if key not in _COHERE_SUPPORTED_SCHEMA_KEYWORDS:
            continue
        if type_union is not None and key in {"type", "anyOf"}:
            continue
        if key == "properties" and isinstance(value, dict):
            projected[key] = {
                name: _cohere_schema_projection(property_schema)
                if isinstance(property_schema, dict)
                else deepcopy(property_schema)
                for name, property_schema in value.items()
            }
        elif isinstance(value, dict):
            projected[key] = _cohere_schema_projection(value)
        elif isinstance(value, list) and key in {"anyOf", "items"}:
            projected[key] = [
                _cohere_schema_projection(item)
                if isinstance(item, dict)
                else deepcopy(item)
                for item in value
            ]
        else:
            projected[key] = deepcopy(value)

    if type_union is not None:
        type_branches = [{"type": deepcopy(item)} for item in type_union]
        if isinstance(existing_any_of, list):
            any_of_branches = [
                _cohere_schema_projection(item)
                if isinstance(item, dict)
                else deepcopy(item)
                for item in existing_any_of
            ]
            projected["anyOf"] = [
                {**type_branch, "anyOf": [deepcopy(any_of_branch)]}
                for any_of_branch in any_of_branches
                for type_branch in type_branches
            ]
        else:
            projected["anyOf"] = type_branches
    return projected


def _cohere_tools_payload(tools: list) -> list:
    """Normalize OpenAI-format tools for Cohere v2's schema subset.

    Cohere v2 keeps the OpenAI-shaped outer tool envelope, while strict-tools
    accepts only a subset of JSON Schema. Each parameter schema is therefore
    projected into a fresh transport disclosure; exact raw tool validation
    remains authoritative. Entries missing ``function.name`` are dropped
    locally instead of being forwarded into a 400.

    Args:
        tools: The ``tools`` list as received (OpenAI shaped).

    Returns:
        A Cohere v2 ``tools`` list.
    """
    converted = []
    for entry in tools or []:
        if not isinstance(entry, dict):
            logger.warning("Cohere: dropping non-dict tools entry.")
            continue
        function = entry.get("function")
        if entry.get("type") == "function" and isinstance(function, dict):
            name = str(function.get("name") or "").strip()
            if not name:
                logger.warning(
                    "Cohere: dropping tool entry with a blank function name."
                )
                continue
            parameters = function.get("parameters")
            if not isinstance(parameters, dict) or not parameters:
                parameters = {"type": "object", "properties": {}}
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(function.get("description") or ""),
                        "parameters": _cohere_schema_projection(parameters),
                    },
                }
            )
        else:
            # Cohere v2's outer tools shape is OpenAI-like; anything outside
            # that function envelope is junk and would 400 the request.
            logger.warning(
                "Cohere: dropping tools entry that is not a valid function tool."
            )
    return converted


def _cohere_request_tool_calls(tool_calls: list) -> list:
    """Convert an OpenAI-shape assistant ``tool_calls`` list into Cohere
    v2's echo shape: ``arguments`` normalized to a JSON STRING (dict ->
    ``json.dumps``; an unparseable string passes through as-is, since v2
    takes strings either way). Junk entries (non-dict, no dict
    ``function``, blank ``name``) are skipped rather than raising --
    callers fall back to plain-content handling when NO entry survives
    (task-267 Task 2; mirrors the anthropic/google all-junk precedent).
    """
    converted = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        raw_args = function.get("arguments")
        if isinstance(raw_args, dict):
            arguments = json.dumps(raw_args)
        elif isinstance(raw_args, str):
            # A NO-ARG streamed call accumulates to "" (tool-call-start
            # seeds arguments:"" and no deltas follow); Cohere 400s the
            # echo unless arguments is a stringified JSON OBJECT (live
            # gate case B, 2026-07-17).
            arguments = raw_args.strip() or "{}"
        else:
            arguments = "{}"
        converted.append(
            {
                "id": str(call.get("id") or ""),
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return converted


def _cohere_response_tool_calls(raw_tool_calls: list) -> list:
    """Normalize a v2 ``message.tool_calls`` list into OpenAI-shape
    entries: ``id``/``type``/``function.name`` passthrough, ``arguments``
    GUARANTEED a string (dict -> ``json.dumps``; anything else falsy/wrong
    type -> ``"{}"``, never crashes the parser -- task-267 Task 3).
    """
    converted = []
    for tc in raw_tool_calls or []:
        if not isinstance(tc, dict):
            continue
        function = tc.get("function")
        if not isinstance(function, dict):
            continue
        if not str(function.get("name") or "").strip():
            # Downstream parsing drops nameless entries anyway; skipping here
            # avoids emitting a non-empty tool_calls list that cannot dispatch
            # (Qodo #690-7).
            continue
        raw_args = function.get("arguments")
        if isinstance(raw_args, str):
            arguments = raw_args
        elif isinstance(raw_args, dict):
            arguments = json.dumps(raw_args)
        else:
            arguments = "{}"
        converted.append(
            {
                "id": str(tc.get("id") or ""),
                "type": str(tc.get("type") or "function"),
                "function": {"name": function.get("name"), "arguments": arguments},
            }
        )
    return converted


def _cohere_stream_event_index(event: dict, message_delta: dict, fallback: int) -> int:
    """Best-effort position resolution for a Cohere v2 streaming tool-call
    event: prefer the event's top-level ``index``; fall back to an
    ``index`` nested under ``delta.message.tool_calls``; else the caller's
    own running counter.

    NOTE: the exact placement of Cohere's ``index`` field on tool-call
    stream events is scout knowledge, not independently verified against
    the live API in this offline task (task-267 Task 4) -- this helper is
    deliberately tolerant of either placement, or its total absence, so
    the position is never mis-synced with the gateway's
    `_ToolCallAccumulator` regardless of which shape the real API sends.
    Task 6's live gate is authoritative.
    """
    tool_calls_field = message_delta.get("tool_calls")
    candidates = [event.get("index")]
    if isinstance(tool_calls_field, dict):
        candidates.append(tool_calls_field.get("index"))
    for raw in candidates:
        if isinstance(raw, bool):
            continue
        if isinstance(raw, int):
            return raw
        if isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
            return int(raw.strip())
    return fallback


def chat_with_cohere(
    input_data: List[Dict[str, Any]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temp: Optional[float] = None,
    streaming: Optional[bool] = False,
    topp: Optional[float] = None,
    topk: Optional[int] = None,
    max_tokens: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
    seed: Optional[int] = None,
    num_generations: Optional[int] = None,  # Only for non-streaming
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    custom_prompt_arg: Optional[
        str
    ] = None,  # Kept for legacy, but focus on structured input
    api_base_url: Optional[str] = None,
):
    start_time = time.time()
    logger.debug(
        f"Cohere Chat: Request process starting for model '{model}' (Streaming: {streaming})"
    )
    cli_api_settings = get_runtime_config_snapshot().values.get("api_settings", {})
    cohere_config = cli_api_settings.get(
        "cohere", {}
    )  # Get the [api_settings.cohere] sub-table

    final_api_key = api_key or cohere_config.get("api_key")
    if not final_api_key:
        raise ChatAuthenticationError(
            provider="cohere", message="Cohere API key is missing."
        )
    logger.debug("Cohere: API key provided.")

    # task-267: the config default is 'command-a-03-2025' (config.py); this
    # inline fallback previously stated the stale v1-era 'command-r' (known
    # discrepancy, fixed here) and is only reached when BOTH the caller and
    # the loaded config omit a model.
    final_model = model or cohere_config.get("model", "command-a-03-2025")

    # Log request metrics
    log_counter(
        "cohere_api_request", labels={"model": final_model, "streaming": str(streaming)}
    )
    api_base_url = (
        api_base_url
        or cohere_config.get("api_base_url")
        or builtin_provider_endpoint("cohere", cohere_config)
    ).rstrip("/")
    # task-267: migrated v1 /chat -> v2 /chat. v1's flat parameter_definitions
    # cannot express nested JSON Schema (MCP tools inexpressible), tool_results
    # lived outside the history model, and there were no call ids. v2 is
    # OpenAI-shaped end-to-end.
    COHERE_CHAT_URL = f"{api_base_url}/v2/chat"

    # Timeout for each attempt, retries will extend total possible time
    timeout_seconds = float(
        cohere_config.get("api_timeout", 180.0)
    )  # Increased default
    # For streaming, timeout usually applies to establishing connection and time between chunks.
    # The session timeout below will handle per-try timeout.

    headers = {
        "Authorization": f"Bearer {final_api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if streaming else "application/json",
    }

    # --- task-267: build the v2 `messages` array -------------------------
    # v2 takes the whole conversation as an OpenAI-shaped messages array
    # (incl. role:"tool") instead of v1's separate message/chat_history/
    # preamble split. A leading system message (or the `system_prompt` param)
    # becomes one inline {"role": "system", ...} entry; any OTHER system
    # message in the history is dropped, matching the v1 handler's prior
    # behavior of only ever honoring ONE system/preamble slot.
    temp_messages = list(input_data or [])  # Make a mutable copy
    cohere_messages: List[Dict[str, Any]] = []

    if system_prompt:
        cohere_messages.append({"role": "system", "content": system_prompt})
    elif temp_messages and temp_messages[0].get("role") == "system":
        sys_msg = temp_messages.pop(0)
        sys_content = sys_msg.get("content") or ""
        cohere_messages.append({"role": "system", "content": str(sys_content)})
        logger.debug(
            "Cohere: Using leading system message as v2 system entry; "
            f"content_bytes={llm_content_byte_count(sys_content)}"
        )

    # task-267 Task 2: role="tool" history and assistant tool_calls echoes
    # convert to v2's shapes here; plain user/assistant text turns pass
    # through as before (Task 1).
    last_tool_call_id: Optional[str] = None
    for msg in temp_messages:
        role = str(msg.get("role") or "").lower()
        content = msg.get("content")

        if role == "tool":
            # OpenAI tool-result history -> v2 tool-role message. A result
            # missing tool_call_id falls back to the most recent assistant
            # tool_call id (positional pairing, mirrors google's fallback
            # -- task-266).
            tool_call_id = msg.get("tool_call_id") or last_tool_call_id or ""
            cohere_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tool_call_id),
                    "content": [
                        {"type": "document", "document": {"data": str(content or "")}}
                    ],
                }
            )
            continue

        if role == "assistant" and msg.get("tool_calls"):
            # OpenAI assistant tool_calls echo -> v2 assistant turn carrying
            # tool_calls (+ tool_plan when present). Guards mirror the
            # anthropic/google precedent: if every tool_calls entry is
            # junk, fall through to the plain-content handling below
            # instead of sending a tool_calls-less message with an empty
            # [] array.
            entries = _cohere_request_tool_calls(msg.get("tool_calls"))
            if entries:
                assistant_msg: Dict[str, Any] = {"role": "assistant"}
                tool_plan = msg.get("cohere_tool_plan")
                if not tool_plan:
                    # Streamed turns carry the plan INSIDE the accumulated
                    # tool_calls entry (fragment extra -> accumulator), not at
                    # message level -- read it back so streamed tool_plan
                    # round-trips too (Qodo #690-4). `_cohere_request_tool_calls`
                    # rebuilds entries, so the extra never leaks to the wire.
                    for _tc in msg.get("tool_calls") or []:
                        if isinstance(_tc, dict) and _tc.get("cohere_tool_plan"):
                            tool_plan = _tc["cohere_tool_plan"]
                            break
                if tool_plan:
                    assistant_msg["tool_plan"] = str(tool_plan)
                elif isinstance(content, str) and content.strip():
                    # No preserved tool_plan extra -- fall back to the
                    # turn's own visible content so the model's reasoning
                    # isn't silently dropped from the echoed history.
                    assistant_msg["tool_plan"] = content
                assistant_msg["tool_calls"] = entries
                cohere_messages.append(assistant_msg)
                last_tool_call_id = entries[-1]["id"] or last_tool_call_id
                continue

        if role not in ("user", "assistant"):
            logger.warning(f"Cohere: skipping message with unsupported role: {role!r}")
            continue

        text_content = content
        if isinstance(content, list):  # Extract text if content is a list of parts
            text_content = next(
                (
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ),
                "",
            )
        cohere_messages.append(
            {
                "role": role,
                "content": str(text_content) if text_content is not None else "",
            }
        )

    if custom_prompt_arg:
        # Legacy path: append (or fold into the trailing user turn) an
        # extra user instruction.
        if cohere_messages and cohere_messages[-1]["role"] == "user":
            cohere_messages[-1]["content"] = (
                f"{cohere_messages[-1]['content']}\n{custom_prompt_arg}"
            )
        else:
            cohere_messages.append({"role": "user", "content": custom_prompt_arg})

    if not any(m["role"] in ("user", "assistant", "tool") for m in cohere_messages):
        raise ChatBadRequestError(
            provider="cohere",
            message="No user/assistant/tool messages found for Cohere chat after processing system message.",
        )

    payload: Dict[str, Any] = {
        "model": final_model,
        "messages": cohere_messages,
        "stream": bool(streaming),
    }
    # Add parameters to payload only if they are not None or have meaningful values
    if temp is not None:
        payload["temperature"] = temp
    if topp is not None:
        payload["p"] = topp
    if topk is not None:
        payload["k"] = topk
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if stop_sequences:
        payload["stop_sequences"] = stop_sequences
    if seed is not None:
        payload["seed"] = seed
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if tools:
        payload["tools"] = _cohere_tools_payload(tools)

    if num_generations is not None:
        # task-267: 'num_generations' is v1-only -- v2 has no equivalent
        # (each request produces exactly one assistant turn). Drop it
        # rather than sending an unknown field.
        logger.debug(
            f"Cohere: 'num_generations' ({num_generations}) is v1-only and has no v2 equivalent; dropping."
        )

    logger.debug(
        "Cohere request metadata: "
        f"model={final_model}; streaming={bool(streaming)}; "
        f"message_count={len(cohere_messages)}; "
        f"content_bytes={llm_content_byte_count(cohere_messages)}"
    )
    logger.debug(f"Cohere request host: {safe_llm_url_host(COHERE_CHAT_URL)}")

    # --- Retry Mechanism ---
    session = create_default_session()
    retry_count = int(cohere_config.get("api_retries", 3))
    retry_delay = float(
        cohere_config.get("api_retry_delay", 1.0)
    )  # Ensure float for backoff_factor

    retry_strategy = Retry(
        total=llm_retry_count(retry_count),
        backoff_factor=retry_delay,
        status_forcelist=[429, 500, 502, 503, 504],  # Standard retry statuses
        allowed_methods=["POST"],  # Retry only for POST requests
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # --- End Retry Mechanism ---

    try:
        if streaming:
            # For streaming, the session.post will use the retry for initial connection.
            # The timeout applies to each attempt for connection and then for pauses in stream.
            response = session.post(
                COHERE_CHAT_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=timeout_seconds,
            )
            response.raise_for_status()  # Check for HTTP errors on initial connection
            logger.debug("Cohere: Streaming response connection established.")

            # Log streaming success metrics
            duration = time.time() - start_time
            log_histogram(
                "cohere_api_response_time",
                duration,
                labels={
                    "model": final_model,
                    "streaming": "true",
                    "status_code": str(response.status_code),
                },
            )
            log_counter(
                "cohere_api_success", labels={"model": final_model, "streaming": "true"}
            )

            def stream_generator_cohere_sse(response_iterator):
                completion_id = f"chatcmpl-cohere-{time.time_ns()}"
                created_ts = int(time.time())
                stream_properly_closed = False
                accumulated_text_for_log = []
                # task-267 Task 4: tool-call streaming state. `tool_plan_text`
                # accumulates tool-plan-delta text so it can ride on the
                # FIRST tool-call fragment (mirrors the shipped
                # google_thought_signature mechanism -- task-266); position
                # tracking lets tool-call-delta events resolve their index
                # even when the event itself omits one.
                next_tool_position = 0
                last_tool_position = 0
                tool_plan_text = ""
                # v2 SSE events are discriminated by "type" (v1 used
                # "event_type"): message-start, content-start, content-delta
                # (delta.message.content.text), content-end, tool-plan-delta
                # (delta.message.tool_plan), tool-call-start/-delta/-end
                # (delta.message.tool_calls), message-end (delta.finish_reason,
                # usage).
                fr_map = {
                    "COMPLETE": "stop",
                    "MAX_TOKENS": "length",
                    "STOP_SEQUENCE": "stop",
                    "TOOL_CALL": "tool_calls",
                }
                try:
                    for line_bytes in response_iterator:
                        if not line_bytes:
                            continue
                        decoded_line = (
                            line_bytes.decode("utf-8")
                            if isinstance(line_bytes, bytes)
                            else str(line_bytes)
                        ).strip()
                        if not decoded_line:
                            continue

                        if not decoded_line.startswith("data:"):
                            if not decoded_line.startswith("event:"):
                                logger.warning(
                                    f"Cohere Stream: Unexpected line format: '{decoded_line}'"
                                )
                            continue

                        json_data_str = decoded_line[len("data:") :].strip()
                        if not json_data_str:
                            continue
                        try:
                            cohere_event = json.loads(json_data_str)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Cohere Stream: JSON decode error for data: '{json_data_str}'"
                            )
                            continue

                        event_type = cohere_event.get("type")
                        delta = cohere_event.get("delta") or {}
                        message_delta = delta.get("message") or {}
                        sse_delta: Dict[str, Any] = {}
                        finish_reason = None

                        if event_type == "content-delta":
                            text_chunk = (message_delta.get("content") or {}).get(
                                "text"
                            )
                            if text_chunk:
                                accumulated_text_for_log.append(text_chunk)
                                sse_delta["content"] = text_chunk
                        elif event_type == "tool-plan-delta":
                            plan_chunk = message_delta.get("tool_plan")
                            if plan_chunk:
                                tool_plan_text += plan_chunk
                        elif event_type == "tool-call-start":
                            tool_call = message_delta.get("tool_calls")
                            if not isinstance(tool_call, dict):
                                logger.warning(
                                    "Cohere Stream: malformed tool-call-start event, skipping."
                                )
                            else:
                                function = tool_call.get("function") or {}
                                position = _cohere_stream_event_index(
                                    cohere_event, message_delta, next_tool_position
                                )
                                next_tool_position = position + 1
                                last_tool_position = position
                                fragment: Dict[str, Any] = {
                                    "index": position,
                                    "id": str(tool_call.get("id") or ""),
                                    "type": str(tool_call.get("type") or "function"),
                                    "function": {
                                        "name": str(function.get("name") or ""),
                                        "arguments": function.get("arguments") or "",
                                    },
                                }
                                if tool_plan_text:
                                    # Ride the accumulated tool-plan text on
                                    # the FIRST fragment only -- the gateway
                                    # accumulator's extras allow-list
                                    # preserves whatever key survives the
                                    # merge (task-267 Task 4).
                                    fragment["cohere_tool_plan"] = tool_plan_text
                                sse_delta["tool_calls"] = [fragment]
                        elif event_type == "tool-call-delta":
                            tool_call = message_delta.get("tool_calls")
                            if not isinstance(tool_call, dict):
                                logger.warning(
                                    "Cohere Stream: malformed tool-call-delta event, skipping."
                                )
                            else:
                                function = tool_call.get("function") or {}
                                position = _cohere_stream_event_index(
                                    cohere_event, message_delta, last_tool_position
                                )
                                sse_delta["tool_calls"] = [
                                    {
                                        "index": position,
                                        "function": {
                                            "arguments": function.get("arguments") or ""
                                        },
                                    }
                                ]
                        elif event_type == "message-end":
                            stream_properly_closed = True
                            raw_finish_reason = delta.get(
                                "finish_reason"
                            ) or cohere_event.get("finish_reason")
                            finish_reason = fr_map.get(
                                raw_finish_reason,
                                raw_finish_reason.lower()
                                if raw_finish_reason
                                else "stop",
                            )
                            logger.info(
                                f"Cohere stream: 'message-end' event. Finish: {raw_finish_reason} "
                                f"(Mapped: {finish_reason}). Fragments: {len(accumulated_text_for_log)}"
                            )
                        elif event_type in (
                            "message-start",
                            "content-start",
                            "content-end",
                            "tool-call-end",
                        ):
                            logger.debug(f"Cohere stream: '{event_type}' event.")
                        elif event_type:
                            logger.debug(
                                f"Cohere stream event type: {event_type}, data: {cohere_event}"
                            )

                        if sse_delta or finish_reason:
                            sse_choice_payload: Dict[str, Any] = {
                                "index": 0,
                                "delta": sse_delta,
                            }
                            if finish_reason:
                                sse_choice_payload["finish_reason"] = finish_reason
                            sse_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": final_model,
                                "choices": [sse_choice_payload],
                            }
                            yield f"data: {json.dumps(sse_chunk)}\n\n"
                            if (
                                event_type == "message-end"
                            ):  # After sending final choice, send DONE
                                yield "data: [DONE]\n\n"
                                return  # End generator

                except requests.exceptions.ChunkedEncodingError as e:
                    logger.warning(
                        f"Cohere stream: ChunkedEncodingError: {e}. Stream may have been interrupted."
                    )
                except Exception as e_stream:
                    logger.opt(exception=True).error(
                        f"Cohere stream: Error during streaming: {e_stream}"
                    )
                finally:  # Ensure [DONE] is sent if loop terminates unexpectedly
                    if not stream_properly_closed:
                        logger.warning(
                            "Cohere stream generator loop finished without explicit 'message-end'."
                        )
                        # The 'message-end' branch already emitted [DONE] on the
                        # happy path; emitting here too doubled the terminator
                        # (Qodo #690-3).
                        yield "data: [DONE]\n\n"
                    logger.debug(
                        f"Cohere SSE stream_generator for {final_model} finished. Total text: {''.join(accumulated_text_for_log)[:100]}..."
                    )
                    if response:
                        response.close()

            return stream_generator_cohere_sse(response.iter_lines())
        else:  # Non-streaming
            # The session.post will use the retry strategy and timeout for each attempt.
            response = session.post(
                COHERE_CHAT_URL,
                headers=headers,
                json=payload,
                stream=False,
                timeout=timeout_seconds,
            )
            # No params={"stream": "false"} needed; payload["stream"] = False handles it.
            response.raise_for_status()  # Will raise HTTPError for bad responses (4xx or 5xx) after retries
            response_data = response.json()
            logger.debug(
                "Cohere non-streaming response metadata: "
                f"status={response.status_code}; "
                f"content_bytes={llm_content_byte_count(response_data)}"
            )

            # ---- v2 response shape ----
            # { "id": "...", "message": {"role":"assistant",
            #     "content":[{"type":"text","text":...}], "tool_calls":[...]?,
            #     "tool_plan":...?},
            #   "finish_reason": "COMPLETE"|"TOOL_CALL"|"MAX_TOKENS"|"STOP_SEQUENCE"|...,
            #   "usage": {...} }
            chat_id = response_data.get("id", f"chatcmpl-cohere-{time.time_ns()}")
            created_timestamp = int(time.time())
            message = response_data.get("message") or {}
            content_parts = message.get("content") or []
            text = "".join(
                part.get("text", "")
                for part in content_parts
                if isinstance(part, dict) and part.get("type") == "text"
            )
            if not message:
                logger.warning(
                    "Cohere non-streaming response missing 'message'; "
                    f"response_type={type(response_data).__name__}; "
                    f"content_bytes={llm_content_byte_count(response_data)}"
                )

            raw_finish_reason = response_data.get("finish_reason")
            fr_map = {
                "COMPLETE": "stop",
                "MAX_TOKENS": "length",
                "STOP_SEQUENCE": "stop",
                "TOOL_CALL": "tool_calls",
            }
            finish_reason = fr_map.get(
                raw_finish_reason,
                raw_finish_reason.lower() if raw_finish_reason else "stop",
            )

            # task-267 Task 3: message.tool_calls -> OpenAI tool_calls,
            # attached ONLY when non-empty; message.tool_plan preserved
            # onto the assistant message as `cohere_tool_plan` (mirrors the
            # shipped `google_thought_signature` round-trip mechanism --
            # task-266 -- so the request converter can re-attach it, see
            # Task 2's `cohere_tool_plan` read).
            message_payload: Dict[str, Any] = {"role": "assistant", "content": text}
            converted_tool_calls = _cohere_response_tool_calls(
                message.get("tool_calls")
            )
            if converted_tool_calls:
                message_payload["tool_calls"] = converted_tool_calls
            tool_plan = message.get("tool_plan")
            if tool_plan:
                message_payload["cohere_tool_plan"] = str(tool_plan)
            choices_payload = [
                {"index": 0, "message": message_payload, "finish_reason": finish_reason}
            ]

            usage_data = None
            usage = response_data.get("usage")
            if isinstance(usage, dict):
                billed = usage.get("billed_units") or usage.get("tokens") or {}
                prompt_tokens = billed.get("input_tokens")
                completion_tokens = billed.get("output_tokens")
                if prompt_tokens is not None and completion_tokens is not None:
                    usage_data = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }

            openai_compatible_response = {
                "id": chat_id,
                "object": "chat.completion",
                "created": created_timestamp,
                "model": final_model,
                "choices": choices_payload,
            }
            if usage_data:
                openai_compatible_response["usage"] = usage_data

            # Log non-streaming success metrics
            duration = time.time() - start_time
            log_histogram(
                "cohere_api_response_time",
                duration,
                labels={
                    "model": final_model,
                    "streaming": "false",
                    "status_code": str(response.status_code),
                },
            )
            log_counter(
                "cohere_api_success",
                labels={"model": final_model, "streaming": "false"},
            )

            # Log token usage if available
            if usage_data:
                log_histogram(
                    "cohere_api_input_tokens",
                    usage_data.get("prompt_tokens", 0),
                    labels={"model": final_model},
                )
                log_histogram(
                    "cohere_api_output_tokens",
                    usage_data.get("completion_tokens", 0),
                    labels={"model": final_model},
                )
                log_histogram(
                    "cohere_api_total_tokens",
                    usage_data.get("total_tokens", 0),
                    labels={"model": final_model},
                )

            return openai_compatible_response

    except requests.exceptions.HTTPError as e:
        status_code = getattr(e.response, "status_code", 500)
        raw_error_text = getattr(e.response, "text", None)
        if raw_error_text is None:
            raw_error_text = safe_llm_exception_message(e)
        error_text = str(safe_llm_error_detail(raw_error_text))
        logger.error(
            "Cohere API call failed; "
            f"host={safe_llm_url_host(COHERE_CHAT_URL)}; "
            f"status={status_code}; detail={error_text[:500]}"
        )

        # Log HTTP error metrics
        duration = time.time() - start_time
        log_counter(
            "cohere_api_error",
            labels={
                "model": final_model,
                "error_type": "http_error",
                "status_code": str(status_code),
            },
        )
        log_histogram(
            "cohere_api_error_response_time",
            duration,
            labels={"model": final_model, "status_code": str(status_code)},
        )
        if status_code == 401:
            raise ChatAuthenticationError(
                provider="cohere",
                message=f"Authentication failed. Detail: {error_text[:200]}",
            )
        elif status_code == 429:
            raise ChatRateLimitError(
                provider="cohere",
                message=f"Rate limit exceeded. Detail: {error_text[:200]}",
            )
        elif 400 <= status_code < 500:
            raise ChatBadRequestError(
                provider="cohere",
                message=f"Bad request (Status {status_code}). Detail: {error_text[:200]}",
            )
        else:  # 5xx
            raise ChatProviderError(
                provider="cohere",
                message=f"Server error (Status {status_code}). Detail: {error_text[:200]}",
                status_code=status_code,
            )
    except (
        requests.exceptions.RequestException
    ) as e:  # Includes ReadTimeout, ConnectionError etc.
        error_detail = safe_llm_exception_message(e)
        error_copy = (
            "Cohere API request failed; reason=network_error; "
            f"host={safe_llm_url_host(COHERE_CHAT_URL)}; "
            f"error_type={error_detail}"
        )
        if is_sensitive_llm_request():
            logger.error(error_copy)
        else:
            logger.opt(exception=True).error(error_copy)

        # Log network error metrics
        duration = time.time() - start_time
        log_counter(
            "cohere_api_error",
            labels={"model": final_model, "error_type": "network_error"},
        )
        log_histogram(
            "cohere_api_error_response_time",
            duration,
            labels={"model": final_model, "error_type": "network_error"},
        )
        # This will catch the ReadTimeout after retries are exhausted
        raise ChatProviderError(
            provider="cohere",
            message=f"Network error after retries: {error_detail}",
            status_code=504,
        )  # 504 for gateway timeout like
    except Exception as e:
        error_detail = safe_llm_exception_message(e)
        error_copy = (
            f"Cohere API call failed; reason=unexpected; error_type={error_detail}"
        )
        if is_sensitive_llm_request():
            logger.error(error_copy)
        else:
            logger.opt(exception=True).error(error_copy)

        # Log unexpected error metrics
        duration = time.time() - start_time
        log_counter(
            "cohere_api_error",
            labels={"model": final_model, "error_type": "unexpected_error"},
        )
        log_histogram(
            "cohere_api_error_response_time",
            duration,
            labels={"model": final_model, "error_type": "unexpected_error"},
        )
        if not isinstance(e, ChatAPIError):
            raise ChatAPIError(
                provider="cohere",
                message=f"Unexpected error in Cohere API call: {error_detail}",
            )
        else:
            raise
    finally:
        if session:  # Ensure session is closed
            session.close()


def chat_with_deepseek(
    input_data: List[Dict[str, Any]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    system_message: Optional[str] = None,
    temp: Optional[float] = None,
    streaming: Optional[bool] = False,
    topp: Optional[float] = None,  # top_p
    # New OpenAI-compatible params for DeepSeek
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    response_format: Optional[Dict[str, str]] = None,  # If supported
    n: Optional[int] = None,  # If supported
    user: Optional[str] = None,  # If supported
    tools: Optional[List[Dict[str, Any]]] = None,  # If supported
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,  # If supported
    logit_bias: Optional[Dict[str, float]] = None,  # If supported
    custom_prompt_arg: Optional[str] = None,  # Legacy
    api_base_url: Optional[str] = None,
):
    start_time = time.time()
    cli_api_settings = get_runtime_config_snapshot().values.get("api_settings", {})
    deepseek_config = cli_api_settings.get(
        "deepseek", {}
    )  # Get the [api_settings.deepseek] sub-table
    final_api_key = api_key or deepseek_config.get("api_key")
    if not final_api_key:
        raise ChatConfigurationError(
            provider="deepseek", message="DeepSeek API Key required."
        )

    logger.debug("DeepSeek: API key provided.")
    current_model = model or deepseek_config.get(
        "model", "deepseek-v4-flash"
    )  # Or deepseek-coder
    current_temp = (
        temp if temp is not None else float(deepseek_config.get("temperature", 0.1))
    )
    current_top_p = topp  # Deepseek uses top_p
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

    # Log request metrics
    log_counter(
        "deepseek_api_request",
        labels={"model": current_model, "streaming": str(current_streaming)},
    )

    current_max_tokens = (
        max_tokens
        if max_tokens is not None
        else _safe_cast(deepseek_config.get("max_tokens"), int)
    )

    api_messages = []
    if system_message:
        api_messages.append({"role": "system", "content": system_message})
    api_messages.extend(input_data)

    headers = {
        "Authorization": f"Bearer {final_api_key}",
        "Content-Type": "application/json",
    }
    data = {
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

    api_url = (
        api_base_url
        or deepseek_config.get("api_base_url")
        or builtin_provider_endpoint("deepseek", deepseek_config)
    ).rstrip("/") + "/chat/completions"
    if not is_sensitive_llm_request():
        # task-2116: see the OpenAI branch above for why this is gated.
        # task-2117 Qodo round: allowlisted summary, see the Anthropic
        # branch above for why a denylist isn't safe here.
        logger.debug(
            "DeepSeek Request Payload (safe fields only): "
            f"{safe_llm_request_payload_summary(data)}"
        )

    try:
        if current_streaming:
            # ... (OpenAI-like streaming logic, use "DeepSeek" in logs) ...
            with create_default_session() as session:
                response = session.post(
                    api_url, headers=headers, json=data, stream=True, timeout=180
                )
                response.raise_for_status()  # Check for HTTP errors on initial connection

                # Log streaming success metrics
                duration = time.time() - start_time
                log_histogram(
                    "deepseek_api_response_time",
                    duration,
                    labels={
                        "model": current_model,
                        "streaming": "true",
                        "status_code": str(response.status_code),
                    },
                )
                log_counter(
                    "deepseek_api_success",
                    labels={"model": current_model, "streaming": "true"},
                )

                def stream_generator():
                    try:
                        for line in response.iter_lines(decode_unicode=True):
                            if (
                                line and line.strip()
                            ):  # DeepSeek provides OpenAI-compatible SSE
                                yield line if line.endswith("\n") else line + "\n"
                    except Exception as e_stream:
                        logger.opt(exception=True).error(
                            f"DeepSeek: Error during stream iteration: {e_stream}"
                        )
                        yield f"data: {json.dumps({'error': {'message': f'Stream iteration error: {str(e_stream)}', 'type': 'deepseek_stream_error'}})}\n\n"
                    finally:
                        yield "data: [DONE]\n\n"
                        if response:
                            response.close()

                return stream_generator()
        else:
            # ... (non-streaming, retry) ...
            adapter = HTTPAdapter(
                max_retries=Retry(
                    total=llm_retry_count(int(deepseek_config.get("api_retries", 3))),
                    backoff_factor=float(deepseek_config.get("api_retry_delay", 1)),
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["POST"],
                )
            )
            with create_default_session() as session:
                session.mount("https://", adapter)
                response = session.post(
                    api_url, headers=headers, json=data, timeout=120
                )
            response.raise_for_status()
            result = response.json()

            # Log non-streaming success metrics
            duration = time.time() - start_time
            log_histogram(
                "mistral_api_response_time",
                duration,
                labels={
                    "model": current_model,
                    "streaming": "false",
                    "status_code": str(response.status_code),
                },
            )
            log_counter(
                "mistral_api_success",
                labels={"model": current_model, "streaming": "false"},
            )

            # Log token usage if available
            usage = result.get("usage", {})
            if usage:
                log_histogram(
                    "mistral_api_input_tokens",
                    usage.get("prompt_tokens", 0),
                    labels={"model": current_model},
                )
                log_histogram(
                    "mistral_api_output_tokens",
                    usage.get("completion_tokens", 0),
                    labels={"model": current_model},
                )
                log_histogram(
                    "mistral_api_total_tokens",
                    usage.get("total_tokens", 0),
                    labels={"model": current_model},
                )

            return result
    except requests.exceptions.HTTPError as e:  # ... error handling ...
        # Log HTTP error metrics
        duration = time.time() - start_time
        status_code = e.response.status_code if e.response is not None else 500
        log_counter(
            "deepseek_api_error",
            labels={
                "model": current_model,
                "error_type": "http_error",
                "status_code": str(status_code),
            },
        )
        log_histogram(
            "deepseek_api_error_response_time",
            duration,
            labels={"model": current_model, "status_code": str(status_code)},
        )
        raise
    except Exception as e:  # ... error handling ...
        # Log unexpected error metrics
        duration = time.time() - start_time
        log_counter(
            "deepseek_api_error",
            labels={"model": current_model, "error_type": "unexpected_error"},
        )
        log_histogram(
            "deepseek_api_error_response_time",
            duration,
            labels={"model": current_model, "error_type": "unexpected_error"},
        )
        raise ChatProviderError(provider="deepseek", message=f"Unexpected error: {e}")


def _google_tools_payload(tools: list) -> list:
    """Wrap OpenAI function-format tool entries as Gemini functionDeclarations.

    Entries already Gemini-shaped (carrying ``functionDeclarations`` /
    ``function_declarations`` or other non-OpenAI keys) pass through
    untouched. OpenAI entries with a blank name are dropped locally —
    Gemini rejects empty tool names (task-263 review precedent).

    Args:
        tools: The ``tools`` list as received (OpenAI or Gemini shaped).

    Returns:
        A Gemini ``tools`` list; OpenAI entries collapse into ONE
        ``{"functionDeclarations": [...]}`` entry, passthrough entries keep
        their positions.
    """
    declarations = []
    passthrough = []
    for entry in tools or []:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        if entry.get("type") == "function" and isinstance(function, dict):
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            parameters = function.get("parameters")
            if not isinstance(parameters, dict) or not parameters:
                parameters = {"type": "object", "properties": {}}
            declarations.append(
                {
                    "name": name,
                    "description": str(function.get("description") or ""),
                    "parametersJsonSchema": deepcopy(parameters),
                }
            )
        else:
            passthrough.append(entry)
    result = list(passthrough)
    if declarations:
        result.append({"functionDeclarations": declarations})
    return result


def _google_function_response(name: str, content) -> dict:
    """Build a Gemini functionResponse part from an OpenAI tool result.

    Gemini requires ``response`` to be a JSON OBJECT: dict-parseable string
    content is used directly; anything else wraps as ``{"result": <str>}``.

    Args:
        name: The function name this result answers (Gemini pairs by name
            plus position — it has no call ids).
        content: The tool result content (string, typically).

    Returns:
        ``{"functionResponse": {"name": ..., "response": {...}}}``.
    """
    response = None
    if isinstance(content, str) and content.strip():
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            response = parsed
    if response is None:
        response = {"result": str(content or "")}
    return {"functionResponse": {"name": name, "response": response}}


def chat_with_google(
    input_data: List[Dict[str, Any]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    system_message: Optional[str] = None,  # -> system_instruction
    temp: Optional[float] = None,  # -> temperature
    streaming: Optional[bool] = False,
    topp: Optional[float] = None,  # -> topP
    topk: Optional[int] = None,  # -> topK
    max_output_tokens: Optional[int] = None,  # from max_tokens
    stop_sequences: Optional[List[str]] = None,  # from stop
    candidate_count: Optional[int] = None,  # from n
    response_format: Optional[Dict[str, str]] = None,  # for response_mime_type
    tools: Optional[List[Dict[str, Any]]] = None,  # Gemini 'tools' config
    custom_prompt_arg: Optional[str] = None,
    api_base_url: Optional[str] = None,
):
    start_time = time.time()
    loaded_config_data = get_runtime_config_snapshot().values.get("api_settings", {})
    # `api_settings.google` -- NOT `api_settings.google_api` (PR-T2 review
    # round 3, finding I4). Nothing in this app has ever produced a
    # `google_api` table: the shipped default config writes `[api_settings.
    # google]`, `config.py`'s credential bridge (`_normalize_legacy_
    # provider_api_key`) writes `api_settings["google"]["api_key"]`, and
    # the legacy dict is the top-level `google_generative_api` (not under
    # `api_settings` at all). So this lookup always returned `{}` and every
    # Google call fell back to the hardcoded defaults below -- meaning a
    # credential set the modern OR legacy way could not reach the spend
    # path even while readiness reported ready and the Library RAG gate
    # opened. `chat_with_mistral` reads its own table the same way.
    google_config = loaded_config_data.get("google", {})
    # `resolve_provider_api_key`, not a bare read (PR-T2 review round 4,
    # N1): `[api_settings.google]` is the ONLY chat-provider table that
    # ships an `api_key` VALUE, and it ships the placeholder
    # `"<API_KEY_HERE>"`. Pointing this lookup at the real table (finding
    # I4, one commit earlier) therefore turned a default config's clear
    # `ChatConfigurationError` into `x-goog-api-key: <API_KEY_HERE>` on
    # the wire and an upstream 400 with an invisible cause. Gated surfaces
    # never reached it -- readiness validates the placeholder away -- so
    # this landed exactly on the ungated `chat_api_call` callers (Evals,
    # briefings, agent runs, WebSearch). This is the same one definition
    # of "valid provider API key" `config.py`'s credential bridge and
    # `Chat/provider_readiness` both use.
    final_api_key = api_key or resolve_provider_api_key(google_config.get("api_key"))
    if not final_api_key:
        raise ChatConfigurationError(
            provider="google", message="Google API Key required."
        )

    current_model = model or google_config.get("model", "gemini-1.5-flash-latest")
    current_streaming_cfg = google_config.get("streaming", False)
    current_streaming = (
        streaming
        if streaming is not None
        else (
            str(current_streaming_cfg).lower() == "true"
            if isinstance(current_streaming_cfg, str)
            else bool(current_streaming_cfg)
        )
    )

    # Log request metrics
    log_counter(
        "google_api_request",
        labels={"model": current_model, "streaming": str(current_streaming)},
    )

    gemini_contents = []
    from tldw_chatbook.Chat.console_project_instructions import EPHEMERAL_ORIGIN_KEY

    tool_call_names: Dict[str, str] = {}
    last_function_call_names: List[str] = []
    consecutive_tool_results = 0
    for msg in input_data:
        role = msg.get("role")
        content = msg.get("content")

        if role == "tool":
            name = tool_call_names.get(str(msg.get("tool_call_id") or ""))
            if name is None:
                # Positional fallback: pair the nth consecutive result with
                # the nth functionCall of the preceding model turn (Gemini
                # pairs by name + order; it has no call ids).
                name = (
                    last_function_call_names[consecutive_tool_results]
                    if consecutive_tool_results < len(last_function_call_names)
                    else ""
                )
            if not name:
                # Unpairable result (id miss + positional fallback
                # exhausted): Gemini rejects empty tool names, so emitting
                # it would 400 the whole request — skip just this part
                # (PR #662 review).
                logger.warning(
                    "Google Gemini: dropping unpairable tool result "
                    f"(tool_call_id={str(msg.get('tool_call_id') or '')!r})"
                )
                consecutive_tool_results += 1
                continue
            part = _google_function_response(name, content)
            consecutive_tool_results += 1
            last = gemini_contents[-1] if gemini_contents else None
            if (
                last is not None
                and last.get("role") == "user"
                and isinstance(last.get("parts"), list)
                and any(
                    "functionResponse" in p
                    for p in last["parts"]
                    if isinstance(p, dict)
                )
            ):
                last["parts"].append(part)
            else:
                gemini_contents.append({"role": "user", "parts": [part]})
            continue
        consecutive_tool_results = 0
        if role == "user" and msg.get(EPHEMERAL_ORIGIN_KEY) == "project_instructions":
            # Consume the internal marker here and keep repository context in
            # its own user turn after any preceding function-response turn.
            gemini_contents.append(
                {"role": "user", "parts": [{"text": str(content or "")}]}
            )
            continue
        if role == "assistant" and msg.get("tool_calls"):
            parts = []
            if isinstance(content, str) and content.strip():
                parts.append({"text": content})
            elif isinstance(content, list):
                # List-form (multimodal) content: keep its text parts —
                # dropping them would silently lose visible text that
                # accompanied the tool calls (same bug class as the
                # anthropic sibling, PR #659 review).
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                        and part["text"].strip()
                    ):
                        parts.append({"text": part["text"]})
            call_names = []
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                if not isinstance(function, dict):
                    continue
                name = str(function.get("name") or "").strip()
                if not name:
                    continue
                raw_args = function.get("arguments")
                args = raw_args if isinstance(raw_args, dict) else {}
                if isinstance(raw_args, str) and raw_args.strip():
                    try:
                        parsed = json.loads(raw_args)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict):
                        args = parsed
                tool_call_names[str(call.get("id") or "")] = name
                call_names.append(name)
                part = {"functionCall": {"name": name, "args": args}}
                signature = call.get("google_thought_signature") or call.get(
                    "thoughtSignature"
                )
                if signature:
                    # Echo Gemini 3 thought signatures back verbatim —
                    # required for tools on current models (live-gate 400).
                    part["thoughtSignature"] = str(signature)
                parts.append(part)
            if call_names:
                last_function_call_names = call_names
                gemini_contents.append({"role": "model", "parts": parts})
                continue
            # All-junk tool_calls: fall through to plain content handling.

        gemini_role = (
            "user" if role == "user" else "model" if role == "assistant" else None
        )
        if not gemini_role:
            continue
        gemini_parts = []
        if isinstance(content, str):
            gemini_parts.append({"text": content})
        elif isinstance(content, list):
            for part_obj in content:
                if part_obj.get("type") == "text":
                    gemini_parts.append({"text": part_obj.get("text", "")})
                elif part_obj.get("type") == "image_url":
                    parsed_image = _parse_data_url_for_multimodal(
                        part_obj.get("image_url", {}).get("url", "")
                    )
                    if parsed_image:
                        gemini_parts.append(
                            {
                                "inline_data": {
                                    "mime_type": parsed_image[0],
                                    "data": parsed_image[1],
                                }
                            }
                        )
        if gemini_parts:
            gemini_contents.append({"role": gemini_role, "parts": gemini_parts})

    generation_config = {}
    if temp is not None:
        generation_config["temperature"] = temp
    if topp is not None:
        generation_config["topP"] = topp
    if topk is not None:
        generation_config["topK"] = topk
    if max_output_tokens is not None:
        generation_config["maxOutputTokens"] = max_output_tokens
    if stop_sequences is not None:
        generation_config["stopSequences"] = stop_sequences
    if candidate_count is not None:
        generation_config["candidateCount"] = candidate_count
    if response_format and response_format.get("type") == "json_object":
        generation_config["responseMimeType"] = "application/json"

    payload = {"contents": gemini_contents}
    if generation_config:
        payload["generationConfig"] = generation_config
    if system_message:
        payload["system_instruction"] = {"parts": [{"text": system_message}]}
    if tools:
        payload["tools"] = _google_tools_payload(tools)

    stream_suffix = (
        ":streamGenerateContent?alt=sse" if current_streaming else ":generateContent"
    )
    google_api_base = (
        api_base_url
        or google_config.get("api_base_url")
        or builtin_provider_endpoint("google", google_config)
    ).rstrip("/")
    api_url = f"{google_api_base}/models/{current_model}{stream_suffix}"
    headers = {"x-goog-api-key": final_api_key, "Content-Type": "application/json"}
    if not is_sensitive_llm_request():
        # task-2116: see the OpenAI branch above for why this is gated.
        # task-2117 Qodo round: allowlisted summary -- "system_instruction"
        # carries the actual system-prompt text and must never be logged
        # verbatim. generationConfig is flattened to the top level first so
        # its (camelCase) sampling params can be picked up by the allowlist.
        google_log_payload = {
            **payload,
            **payload.get("generationConfig", {}),
            "streaming": current_streaming,
        }
        logger.debug(
            "Google Gemini Request Payload (safe fields only): "
            f"{safe_llm_request_payload_summary(google_log_payload, content_keys=('contents',), system_keys=('system_instruction',))}"
        )
    logger.debug(
        "Google Gemini request content metadata: "
        f"message_count={len(gemini_contents)}; "
        f"content_bytes={llm_content_byte_count(gemini_contents)}"
    )

    response = None  # Initialize response to None for the finally block
    try:
        adapter = HTTPAdapter(
            max_retries=Retry(
                total=llm_retry_count(int(google_config.get("api_retries", 3))),
                backoff_factor=float(google_config.get("api_retry_delay", 1)),
                status_forcelist=[429, 500, 503],
                allowed_methods=["POST"],
            )
        )
        with create_default_session() as session:
            session.mount("https://", adapter)
            response = session.post(
                api_url,
                headers=headers,
                json=payload,
                stream=current_streaming,
                timeout=180,
                # task-686 AC #3: the API key travels in the custom
                # `x-goog-api-key` header. `requests` strips `Authorization`
                # across a redirect host change but NOT custom headers, so a
                # 3xx from this endpoint would re-send the key wherever
                # `Location` points. Refuse to follow rather than silently
                # forward credentials -- see the explicit 3xx check below.
                allow_redirects=False,
            )
        if 300 <= response.status_code < 400:
            location = response.headers.get("Location", "<no Location header>")
            logger.error(
                f"Google Gemini: API endpoint returned a redirect "
                f"({response.status_code} -> {location}); refusing to follow "
                f"with the x-goog-api-key credential."
            )
            response.close()
            raise ChatProviderError(
                provider="google",
                message=(
                    "Google endpoint redirected unexpectedly -- refusing to "
                    "follow with credentials."
                ),
                status_code=response.status_code,
            )
        response.raise_for_status()

        if current_streaming:
            logger.debug("Google Gemini: Streaming response received.")

            # Log streaming success metrics
            duration = time.time() - start_time
            log_histogram(
                "google_api_response_time",
                duration,
                labels={
                    "model": current_model,
                    "streaming": "true",
                    "status_code": str(response.status_code),
                },
            )
            log_counter(
                "google_api_success",
                labels={"model": current_model, "streaming": "true"},
            )

            def stream_generator():
                # response object is from the outer scope
                nonlocal response
                completion_id = f"chatcmpl-gemini-{time.time_ns()}"
                created_ts = int(time.time())
                # task-266: 0-based running position across the whole stream
                # for synthesizing OpenAI tool_calls[].index (Gemini streams
                # functionCall parts WHOLE, one complete fragment per call).
                next_tool_position = 0
                try:
                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.strip().startswith("data:"):
                            json_str = line.strip()[len("data:") :].strip()
                            try:
                                data_chunk_outer = json.loads(json_str)
                                openai_sse_choice = None
                                candidates = data_chunk_outer.get("candidates", [])
                                if candidates:
                                    candidate = candidates[0]
                                    chunk_text = ""
                                    chunk_tool_calls = []
                                    if candidate.get("content", {}).get("parts", []):
                                        for part in candidate["content"]["parts"]:
                                            if "text" in part:
                                                chunk_text += part.get("text", "")
                                            if (
                                                isinstance(part, dict)
                                                and "functionCall" in part
                                            ):
                                                fc = part.get("functionCall")
                                                if not isinstance(fc, dict):
                                                    # Malformed part: skip it —
                                                    # never abort an otherwise-
                                                    # valid stream (task-263
                                                    # sibling bug class).
                                                    continue
                                                name = str(fc.get("name") or "").strip()
                                                if not name:
                                                    continue
                                                fragment = {
                                                    "index": next_tool_position,
                                                    "id": f"call_gemini_{time.time_ns()}_{next_tool_position}",
                                                    "type": "function",
                                                    "function": {
                                                        "name": name,
                                                        "arguments": json.dumps(
                                                            fc["args"]
                                                            if isinstance(
                                                                fc.get("args"), dict
                                                            )
                                                            else {}
                                                        ),
                                                    },
                                                }
                                                # Gemini 3 thought signature:
                                                # must round-trip (see the
                                                # non-streaming parser note).
                                                signature = part.get(
                                                    "thoughtSignature"
                                                ) or part.get("thought_signature")
                                                if signature:
                                                    fragment[
                                                        "google_thought_signature"
                                                    ] = str(signature)
                                                chunk_tool_calls.append(fragment)
                                                next_tool_position += 1
                                    raw_finish_reason = candidate.get("finishReason")
                                    openai_finish_reason = None
                                    if raw_finish_reason:
                                        fr_map = {
                                            "MAX_TOKENS": "length",
                                            "STOP": "stop",
                                            "SAFETY": "content_filter",
                                            "RECITATION": "content_filter",
                                            "OTHER": "error",
                                            "TOOL_CODE_NOT_FOUND": "error",
                                            "FUNCTION_CALL": "tool_calls",
                                        }
                                        openai_finish_reason = fr_map.get(
                                            raw_finish_reason, raw_finish_reason.lower()
                                        )
                                    delta_payload_for_choice = {}
                                    if chunk_text:
                                        delta_payload_for_choice["content"] = chunk_text
                                    if chunk_tool_calls:
                                        delta_payload_for_choice["tool_calls"] = (
                                            chunk_tool_calls
                                        )
                                    if delta_payload_for_choice or openai_finish_reason:
                                        openai_sse_choice = {
                                            "index": 0,
                                            "delta": delta_payload_for_choice,
                                        }
                                        if openai_finish_reason:
                                            openai_sse_choice["finish_reason"] = (
                                                openai_finish_reason
                                            )
                                if openai_sse_choice:
                                    sse_chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_ts,
                                        "model": current_model,
                                        "choices": [openai_sse_choice],
                                    }
                                    yield f"data: {json.dumps(sse_chunk)}\n\n"
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Google Gemini: Could not decode JSON line: {json_str}"
                                )
                except requests.exceptions.ChunkedEncodingError as e:
                    logger.opt(exception=True).error(
                        f"Google Gemini: ChunkedEncodingError during stream: {e}"
                    )
                    yield f"data: {json.dumps({'error': {'message': f'Stream connection error: {str(e)}', 'type': 'gemini_stream_error'}})}\n\n"
                except Exception as e_stream:
                    logger.opt(exception=True).error(
                        f"Google Gemini: Error during stream iteration: {e_stream}"
                    )
                    yield f"data: {json.dumps({'error': {'message': f'Stream iteration error: {str(e_stream)}', 'type': 'gemini_stream_error'}})}\n\n"
                finally:
                    yield "data: [DONE]\n\n"
                    if response:  # Close the response from the outer scope
                        response.close()

            return stream_generator()
        else:  # Non-streaming
            response_data = response.json()
            logger.debug("Google Gemini: Non-streaming request successful.")
            assistant_content = ""
            finish_reason = "unknown"
            tool_calls = None

            if response_data.get("candidates"):
                candidate = response_data["candidates"][0]
                if candidate.get("content", {}).get("parts"):
                    parts = candidate["content"]["parts"]
                    for part in parts:
                        if "text" in part:
                            assistant_content += part.get("text", "")
                        if "functionCall" in part:
                            fc = part.get("functionCall")
                            if not isinstance(fc, dict):
                                # Malformed part: skip it — never crash the
                                # parser (PR #662 review; mirrors the
                                # streaming guard).
                                continue
                            if tool_calls is None:
                                tool_calls = []
                            entry = {
                                "id": f"call_gemini_{time.time_ns()}_{len(tool_calls)}",
                                "type": "function",
                                "function": {
                                    "name": fc.get("name"),
                                    "arguments": json.dumps(
                                        fc.get("args")
                                        if isinstance(fc.get("args"), dict)
                                        else {}
                                    ),
                                },
                            }
                            # Gemini 3-family models REQUIRE the part's
                            # thoughtSignature to be echoed back verbatim on
                            # the follow-up request (live-gate 400 without
                            # it). Carry it opaquely on the OpenAI-shape
                            # entry; the request converter re-attaches it.
                            signature = part.get("thoughtSignature") or part.get(
                                "thought_signature"
                            )
                            if signature:
                                entry["google_thought_signature"] = str(signature)
                            tool_calls.append(entry)
                raw_finish_reason = candidate.get("finishReason")
                if raw_finish_reason:
                    fr_map = {
                        "MAX_TOKENS": "length",
                        "STOP": "stop",
                        "SAFETY": "content_filter",
                        "RECITATION": "content_filter",
                        "OTHER": "error",
                        "TOOL_CODE_NOT_FOUND": "error",
                        "FUNCTION_CALL": "tool_calls",
                    }
                    finish_reason = fr_map.get(
                        raw_finish_reason, raw_finish_reason.lower()
                    )

            message_content = {
                "role": "assistant",
                "content": assistant_content.strip(),
            }
            if tool_calls:
                message_content["tool_calls"] = tool_calls
                if not assistant_content.strip():
                    message_content["content"] = None

            prompt_feedback = response_data.get("promptFeedback")
            if prompt_feedback and prompt_feedback.get("blockReason"):
                logger.warning(
                    f"Google Gemini: Prompt blocked. Reason: {prompt_feedback.get('blockReason')}, Safety Ratings: {prompt_feedback.get('safetyRatings')}"
                )
                if not response_data.get("candidates"):
                    message_content["content"] = (
                        "[Blocked by API due to safety settings]"
                    )
                    finish_reason = "content_filter"

            normalized_response = {
                "id": f"gemini-{time.time_ns()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": current_model,
                "choices": [
                    {
                        "index": 0,
                        "message": message_content,
                        "finish_reason": finish_reason,
                    }
                ],
            }
            usage_meta = response_data.get("usageMetadata")
            if usage_meta and all(
                k in usage_meta
                for k in ["promptTokenCount", "candidatesTokenCount", "totalTokenCount"]
            ):
                normalized_response["usage"] = {
                    "prompt_tokens": usage_meta.get("promptTokenCount"),
                    "completion_tokens": usage_meta.get("candidatesTokenCount"),
                    "total_tokens": usage_meta.get("totalTokenCount"),
                }

            # Log non-streaming success metrics
            duration = time.time() - start_time
            log_histogram(
                "google_api_response_time",
                duration,
                labels={
                    "model": current_model,
                    "streaming": "false",
                    "status_code": str(response.status_code),
                },
            )
            log_counter(
                "google_api_success",
                labels={"model": current_model, "streaming": "false"},
            )

            # Log token usage if available
            if usage_meta:
                log_histogram(
                    "google_api_input_tokens",
                    usage_meta.get("promptTokenCount", 0),
                    labels={"model": current_model},
                )
                log_histogram(
                    "google_api_output_tokens",
                    usage_meta.get("candidatesTokenCount", 0),
                    labels={"model": current_model},
                )
                log_histogram(
                    "google_api_total_tokens",
                    usage_meta.get("totalTokenCount", 0),
                    labels={"model": current_model},
                )

            return normalized_response

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else 500
        raw_error_text = (
            e.response.text if e.response is not None else safe_llm_exception_message(e)
        )
        error_text = str(safe_llm_error_detail(raw_error_text))
        logger.error(
            "Google Gemini API call failed; "
            f"status={status_code}; detail={error_text[:500]}"
        )

        # Log HTTP error metrics
        duration = time.time() - start_time
        log_counter(
            "google_api_error",
            labels={
                "model": current_model,
                "error_type": "http_error",
                "status_code": str(status_code),
            },
        )
        log_histogram(
            "google_api_error_response_time",
            duration,
            labels={"model": current_model, "status_code": str(status_code)},
        )
        if status_code == 400:
            if is_sensitive_llm_request():
                raise ChatBadRequestError(
                    provider="google",
                    message=f"Bad request ({status_code}). Detail: {error_text[:200]}",
                ) from e
            try:
                error_json = e.response.json()
                detail = error_json.get("error", {}).get("message", error_text)
                if (
                    "The response was blocked" in detail
                    or "The prompt was blocked" in detail
                    or "SAFETY" in detail.upper()
                ):
                    raise ChatProviderError(
                        provider="google",
                        message=f"Content blocked by API: {detail[:200]}",
                        status_code=status_code,
                        is_content_filter=True,
                    ) from e
                raise ChatBadRequestError(
                    provider="google",
                    message=f"Bad request ({status_code}). Detail: {detail[:200]}",
                ) from e
            except json.JSONDecodeError:
                raise ChatBadRequestError(
                    provider="google",
                    message=f"Bad request ({status_code}). Detail: {error_text[:200]}",
                ) from e
        elif status_code == 401:
            raise ChatAuthenticationError(
                provider="google", message=f"Auth failed. Detail: {error_text[:200]}"
            ) from e
        elif status_code == 429:
            raise ChatRateLimitError(
                provider="google", message=f"Rate limit. Detail: {error_text[:200]}"
            ) from e
        else:
            raise ChatProviderError(
                provider="google",
                message=f"API error ({status_code}). Detail: {error_text[:200]}",
                status_code=status_code,
            ) from e
    except requests.exceptions.RequestException as e:
        # Log network error metrics
        duration = time.time() - start_time
        log_counter(
            "google_api_error",
            labels={"model": current_model, "error_type": "network_error"},
        )
        log_histogram(
            "google_api_error_response_time",
            duration,
            labels={"model": current_model, "error_type": "network_error"},
        )
        error_detail = safe_llm_exception_message(e)
        raise ChatProviderError(
            provider="google", message=f"Network error: {error_detail}", status_code=504
        ) from e
    except Exception as e:
        error_detail = safe_llm_exception_message(e)
        error_copy = f"Google Gemini: Unexpected error: {error_detail}"
        if is_sensitive_llm_request():
            logger.error(error_copy)
        else:
            logger.opt(exception=True).error(error_copy)

        # Log unexpected error metrics
        duration = time.time() - start_time
        log_counter(
            "google_api_error",
            labels={"model": current_model, "error_type": "unexpected_error"},
        )
        log_histogram(
            "google_api_error_response_time",
            duration,
            labels={"model": current_model, "error_type": "unexpected_error"},
        )
        # Ensure it's a ChatAPIError subtype before raising, or wrap it
        if isinstance(e, ChatAPIError):
            raise
        else:
            raise ChatProviderError(
                provider="google", message=f"Unexpected error: {error_detail}"
            ) from e
    finally:
        # If streaming, the response object is closed inside stream_generator's finally.
        # If not streaming, the response object is implicitly closed by the `with create_default_session() as session:` block ending.
        # However, if `session.post` was called outside `with` or `response` was from `session.post(stream=True)`
        # and an error occurred *before* entering `stream_generator`, it might need closing here.
        # The `nonlocal response` and assignment `response = session.post(...)` helps manage this.
        if (
            not current_streaming and response and response.connection is not None
        ):  # Check if response exists and not already closed
            # For non-streaming, response.close() is usually handled by requests Session context manager.
            # For streaming, it's handled in the generator's finally.
            # This is a fallback, generally not needed if using `with session:` properly.
            try:
                if (
                    response.raw and not response.raw.closed
                ):  # For non-streaming with `requests.post`
                    response.raw.close()
            except (
                AttributeError
            ):  # `response.raw` might not exist if connection failed early
                pass
            except Exception as e_close:
                logger.warning(
                    f"Google Gemini: Error during explicit response.close in outer finally: {e_close}"
                )


# https://console.groq.com/docs/quickstart
def chat_with_groq(
    input_data: List[Dict[str, Any]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    system_message: Optional[str] = None,
    temp: Optional[float] = None,
    maxp: Optional[float] = None,  # top_p
    streaming: Optional[bool] = False,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    response_format: Optional[Dict[str, str]] = None,
    n: Optional[int] = None,
    user: Optional[str] = None,  # user_identifier
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    logit_bias: Optional[Dict[str, float]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    custom_prompt_arg: Optional[str] = None,  # Legacy
    api_base_url: Optional[str] = None,
):
    start_time = time.time()
    cli_api_settings = get_runtime_config_snapshot().values.get("api_settings", {})
    groq_config = cli_api_settings.get(
        "groq", {}
    )  # Get the [api_settings.cohere] sub-table
    final_api_key = api_key or groq_config.get("api_key")
    if not final_api_key:
        raise ChatConfigurationError(provider="groq", message="Groq API Key required.")

    logger.debug("Groq: API key provided.")

    current_model = model or groq_config.get("model", "llama3-8b-8192")
    current_temp = (
        temp if temp is not None else float(groq_config.get("temperature", 0.2))
    )
    current_top_p = maxp  # Groq uses top_p
    current_streaming_cfg = groq_config.get("streaming", False)
    current_streaming = (
        streaming
        if streaming is not None
        else (
            str(current_streaming_cfg).lower() == "true"
            if isinstance(current_streaming_cfg, str)
            else bool(current_streaming_cfg)
        )
    )

    # Log request metrics
    log_counter(
        "groq_api_request",
        labels={"model": current_model, "streaming": str(current_streaming)},
    )

    current_max_tokens = (
        max_tokens
        if max_tokens is not None
        else _safe_cast(groq_config.get("max_tokens"), int)
    )

    api_messages = []
    if system_message:
        api_messages.append({"role": "system", "content": system_message})
    api_messages.extend(input_data)

    headers = {
        "Authorization": f"Bearer {final_api_key}",
        "Content-Type": "application/json",
    }
    data = {
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
    if presence_penalty is not None:
        data["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        data["frequency_penalty"] = frequency_penalty
    if logprobs is not None:
        data["logprobs"] = logprobs
    if top_logprobs is not None and data.get("logprobs") is True:
        data["top_logprobs"] = top_logprobs

    api_url = (
        api_base_url
        or groq_config.get("api_base_url")
        or builtin_provider_endpoint("groq", groq_config)
    ).rstrip("/") + "/chat/completions"
    if not is_sensitive_llm_request():
        # task-2116: see the OpenAI branch above for why this is gated.
        # task-2117 Qodo round: allowlisted summary, see the Anthropic
        # branch above for why a denylist isn't safe here.
        logger.debug(
            "Groq Request Payload (safe fields only): "
            f"{safe_llm_request_payload_summary(data)}"
        )
    try:
        if current_streaming:
            # ... (OpenAI-like streaming logic, ensure "Groq" in logs) ...
            with create_default_session() as session:
                response = session.post(
                    api_url, headers=headers, json=data, stream=True, timeout=180
                )
                response.raise_for_status()

                # Log streaming success metrics
                duration = time.time() - start_time
                log_histogram(
                    "openrouter_api_response_time",
                    duration,
                    labels={
                        "model": current_model,
                        "streaming": "true",
                        "status_code": str(response.status_code),
                    },
                )
                log_counter(
                    "openrouter_api_success",
                    labels={"model": current_model, "streaming": "true"},
                )

                def stream_generator():
                    try:
                        for line in response.iter_lines(decode_unicode=True):
                            if (
                                line and line.strip()
                            ):  # Groq provides OpenAI-compatible SSE
                                yield line if line.endswith("\n") else line + "\n"
                    except (
                        requests.exceptions.ChunkedEncodingError
                    ) as e:  # ... error handling ...
                        logger.opt(exception=True).error(
                            f"Groq: ChunkedEncodingError: {e}"
                        )
                        yield f"data: {json.dumps({'error': {'message': f'Stream error: {str(e)}', 'type': 'groq_stream_error'}})}\n\n"
                    except Exception as e:  # ... error handling ...
                        logger.opt(exception=True).error(
                            f"Groq: Stream iteration error: {e}"
                        )
                        yield f"data: {json.dumps({'error': {'message': f'Stream iteration error: {str(e)}', 'type': 'groq_stream_error'}})}\n\n"
                    finally:
                        yield "data: [DONE]\n\n"
                        if response:
                            response.close()

                return stream_generator()
        else:
            # ... (non-streaming logic, retry) ...
            retry_count = int(groq_config.get("api_retries", 3))  # ... retry setup ...
            adapter = HTTPAdapter(
                max_retries=Retry(
                    total=llm_retry_count(retry_count),
                    backoff_factor=float(groq_config.get("api_retry_delay", 1)),
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["POST"],
                )
            )
            with create_default_session() as session:
                session.mount("https://", adapter)
                response = session.post(
                    api_url, headers=headers, json=data, timeout=120
                )
            response.raise_for_status()
            result = response.json()

            # Log non-streaming success metrics
            duration = time.time() - start_time
            log_histogram(
                "mistral_api_response_time",
                duration,
                labels={
                    "model": current_model,
                    "streaming": "false",
                    "status_code": str(response.status_code),
                },
            )
            log_counter(
                "mistral_api_success",
                labels={"model": current_model, "streaming": "false"},
            )

            # Log token usage if available
            usage = result.get("usage", {})
            if usage:
                log_histogram(
                    "mistral_api_input_tokens",
                    usage.get("prompt_tokens", 0),
                    labels={"model": current_model},
                )
                log_histogram(
                    "mistral_api_output_tokens",
                    usage.get("completion_tokens", 0),
                    labels={"model": current_model},
                )
                log_histogram(
                    "mistral_api_total_tokens",
                    usage.get("total_tokens", 0),
                    labels={"model": current_model},
                )

            return result
    except requests.exceptions.HTTPError as e:  # ... error handling ...
        # Log HTTP error metrics
        duration = time.time() - start_time
        status_code = e.response.status_code if e.response is not None else 500
        log_counter(
            "groq_api_error",
            labels={
                "model": current_model,
                "error_type": "http_error",
                "status_code": str(status_code),
            },
        )
        log_histogram(
            "groq_api_error_response_time",
            duration,
            labels={"model": current_model, "status_code": str(status_code)},
        )
        raise
    except Exception as e:  # ... error handling ...
        # Log unexpected error metrics
        duration = time.time() - start_time
        log_counter(
            "groq_api_error",
            labels={"model": current_model, "error_type": "unexpected_error"},
        )
        log_histogram(
            "groq_api_error_response_time",
            duration,
            labels={"model": current_model, "error_type": "unexpected_error"},
        )
        raise ChatProviderError(provider="groq", message=f"Unexpected error: {e}")


def chat_with_huggingface(
    input_data: List[Dict[str, Any]],
    model: Optional[str] = None,  # This is the model_id like "Org/ModelName"
    api_key: Optional[str] = None,
    system_message: Optional[
        str
    ] = None,  # Renamed from system_prompt for clarity if it maps to HF system
    temp: Optional[float] = None,
    streaming: Optional[bool] = False,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[
        int
    ] = None,  # Maps to max_new_tokens for some TGI, or max_tokens for OpenAI compatible
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    response_format: Optional[Dict[str, str]] = None,
    num_return_sequences: Optional[int] = None,  # Mapped from 'n'
    user: Optional[str] = None,  # OpenAI compatible user field
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    logit_bias: Optional[Dict[str, float]] = None,  # OpenAI compatible
    presence_penalty: Optional[float] = None,  # OpenAI compatible name
    frequency_penalty: Optional[float] = None,  # OpenAI compatible name
    logprobs: Optional[bool] = None,  # OpenAI compatible name
    top_logprobs: Optional[int] = None,  # OpenAI compatible name
    custom_prompt_arg: Optional[str] = None,  # Legacy
    api_base_url: Optional[str] = None,
):
    start_time = time.time()
    logger.debug(
        f"HuggingFace Chat: Request process starting for model '{model}' (Streaming: {streaming})"
    )
    loaded_config_data = load_settings()
    hf_config = loaded_config_data.get(
        "huggingface_api", loaded_config_data.get("API", {}).get("huggingface", {})
    )

    final_api_key = api_key or hf_config.get("api_key")
    if final_api_key:
        logger.debug("HuggingFace: API key provided.")
    else:
        logger.warning(
            "HuggingFace: API key is missing. Public Inference API or unsecured TGI assumed."
        )

    headers = {"Content-Type": "application/json"}
    if final_api_key:
        headers["Authorization"] = f"Bearer {final_api_key}"

    final_model_for_payload = (
        model or hf_config.get("model_id") or hf_config.get("model")
    )
    if not final_model_for_payload:
        raise ChatConfigurationError(
            provider="huggingface",
            message="HuggingFace model ID is required (must be passed as 'model' or configured).",
        )
    final_model_for_payload = str(final_model_for_payload).strip().strip("/")
    logger.info(f"HuggingFace: Using model_id for payload: {final_model_for_payload}")

    # --- URL Construction ---
    api_url: str
    use_router_url_format_str = str(
        hf_config.get(
            "use_router_url_format",
            hf_config.get("huggingface_use_router_url_format", "False"),
        )
    ).lower()
    model_path_part = final_model_for_payload.strip("/")

    if use_router_url_format_str == "true":
        # This format explicitly puts the model in the URL path.
        # User must ensure router_base_url and model_id result in a valid endpoint.
        router_base = _optional_config_string(
            api_base_url
            or hf_config.get(
                "router_base_url", hf_config.get("huggingface_router_base_url")
            ),
            default=builtin_provider_endpoint("huggingface", hf_config) or "",
        ).rstrip("/")
        chat_path = _optional_config_string(
            hf_config.get("api_chat_path"), "v1/chat/completions"
        ).lstrip("/")
        # Constructs URL like: {router_base}/models/{model_path_part}/{chat_path}
        router_chat_url = _huggingface_router_chat_url(router_base)
        if router_chat_url:
            api_url = router_chat_url
        elif router_base.endswith("/models"):
            api_url = f"{router_base}/{model_path_part}/{chat_path}"
        else:
            api_url = f"{router_base}/models/{model_path_part}/{chat_path}"
        logger.info(
            "HuggingFace: Using explicit 'use_router_url_format=true'. "
            f"Target host: {safe_llm_url_host(api_url)}"
        )
    else:  # use_router_url_format is false, standard URL construction
        configured_api_base_url = _optional_config_string(
            api_base_url or hf_config.get("api_base_url")
        )
        # Default chat path can be just "chat/completions" if base_url includes /v1, or "v1/chat/completions" if not.
        # Let's make the default api_chat_path more flexible.
        # If using the public HF API, base is /v1 and path is chat/completions.
        configured_api_base = configured_api_base_url.rstrip("/")
        default_chat_path = (
            "chat/completions"
            if configured_api_base.endswith("/v1")
            else "v1/chat/completions"
        )
        chat_completions_path = _optional_config_string(
            hf_config.get("api_chat_path"), default_chat_path
        ).lstrip("/")

        if configured_api_base_url:
            # If api_base_url is configured, use it directly and append the chat_completions_path.
            # The model is expected to be in the payload.
            # If the endpoint needs the model_id in the path, configured_api_base_url should include it fully.
            router_chat_url = _huggingface_router_chat_url(configured_api_base)
            if router_chat_url:
                api_url = router_chat_url
            elif configured_api_base.endswith("/models"):
                api_url = (
                    f"{configured_api_base}/{model_path_part}/{chat_completions_path}"
                )
            else:
                api_url = f"{configured_api_base}/{chat_completions_path}"
            logger.info(
                "HuggingFace: Using configured endpoint; "
                f"host={safe_llm_url_host(api_url)}; model_in_payload=true."
            )
        else:
            # Fallback if no api_base_url is configured.
            # Use the public Hugging Face Inference API endpoint for OpenAI-like chat completions.
            default_hf_api_base = builtin_provider_endpoint("huggingface", hf_config)
            default_chat_path_for_api_inference = (
                "chat/completions"  # Path relative to /v1 base
            )
            api_url = f"{default_hf_api_base.rstrip('/')}/{default_chat_path_for_api_inference}"
            logger.warning(
                "HuggingFace: 'api_base_url' not configured. "
                f"Defaulting to host={safe_llm_url_host(api_url)}; "
                "model_in_payload=true."
            )
    # --- End URL Construction ---

    final_temp = (
        temp
        if temp is not None
        else _safe_cast(hf_config.get("temperature"), float, 0.7)
    )
    # Ensure final_streaming is a boolean for the payload
    hf_config_streaming = hf_config.get("streaming", False)
    final_streaming_payload_val = (
        streaming
        if streaming is not None
        else (
            str(hf_config_streaming).lower() == "true"
            if isinstance(hf_config_streaming, str)
            else bool(hf_config_streaming)
        )
    )

    # TGI uses max_new_tokens. OpenAI compatible layers might expect max_tokens.
    # If max_tokens is provided, prefer it. Otherwise, check hf_config for max_new_tokens or max_tokens
    final_max_val = max_tokens
    if final_max_val is None:
        final_max_val = _safe_cast(
            hf_config.get("max_tokens", hf_config.get("max_new_tokens")), int
        )

    # Log request metrics
    log_counter(
        "huggingface_api_request",
        labels={
            "model": final_model_for_payload,
            "streaming": str(final_streaming_payload_val),
        },
    )

    api_messages = []
    # Handle system message: TGI usually wants it as the first message if no dedicated 'system' field in payload root
    # For OpenAI compatible /v1/chat/completions, system message is standard.
    if system_message:
        api_messages.append({"role": "system", "content": system_message})
    api_messages.extend(
        input_data
    )  # input_data should be correctly formatted by caller

    payload: Dict[str, Any] = {
        "model": final_model_for_payload,  # Model ID is crucial for endpoints that multiplex
        "messages": api_messages,
        "stream": final_streaming_payload_val,  # Use the boolean value
    }

    if final_temp is not None:
        payload["temperature"] = final_temp
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if final_max_val is not None:
        # Use "max_tokens" for OpenAI compatibility, TGI might map this or use "max_new_tokens"
        # Sticking to "max_tokens" if the endpoint is /v1/chat/completions
        payload["max_tokens"] = final_max_val
    if seed is not None:
        payload["seed"] = seed
    if stop is not None:
        payload["stop_sequences"] = (
            stop if isinstance(stop, list) else [stop]
        )  # TGI often uses stop_sequences
    if response_format is not None:
        payload["response_format"] = response_format  # For OpenAI compatible JSON mode

    if num_return_sequences is not None and not final_streaming_payload_val:
        payload["n"] = num_return_sequences
    if user is not None:
        payload["user"] = user
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if logit_bias is not None:
        payload["logit_bias"] = logit_bias
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty
    if logprobs is not None:
        payload["logprobs"] = logprobs
    if top_logprobs is not None and payload.get("logprobs"):
        payload["top_logprobs"] = top_logprobs

    # Remove None values from payload before sending, common practice
    payload = {k: v for k, v in payload.items() if v is not None}

    if is_sensitive_llm_request():
        logger.debug(
            "HuggingFace request metadata: "
            f"model={final_model_for_payload}; "
            f"streaming={final_streaming_payload_val}; "
            f"message_count={len(api_messages)}; "
            f"content_bytes={llm_content_byte_count(api_messages)}"
        )
    else:
        logger.debug(
            "HuggingFace Final Payload (safe fields only): "
            f"{safe_llm_request_payload_summary(payload)}"
        )
    if "tools" in payload and not is_sensitive_llm_request():
        tools_summary = safe_llm_request_payload_summary(
            {"tools": payload["tools"]}, content_keys=()
        )
        logger.debug(f"HuggingFace Tools: {tools_summary}")
    redacted_headers = {
        key: "<redacted>" if key.lower() == "authorization" else value
        for key, value in headers.items()
    }
    logger.debug(f"HuggingFace Headers: {redacted_headers}")

    timeout_seconds = float(hf_config.get("api_timeout", 120.0))
    # For streaming, timeout applies to initial connection and pauses between data.
    # Consider a tuple timeout (connect_timeout, read_timeout) for more control if needed.

    try:
        if final_streaming_payload_val:  # Check the boolean intended for payload
            logger.debug(
                "HuggingFace: Posting streaming request to "
                f"host={safe_llm_url_host(api_url)}"
            )
            # Session might not be strictly necessary for a single streaming POST, but good for potential keep-alive
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=timeout_seconds,
            )
            response.raise_for_status()

            # Log streaming success metrics
            duration = time.time() - start_time
            log_histogram(
                "huggingface_api_response_time",
                duration,
                labels={
                    "model": final_model_for_payload,
                    "streaming": "true",
                    "status_code": str(response.status_code),
                },
            )
            log_counter(
                "huggingface_api_success",
                labels={"model": final_model_for_payload, "streaming": "true"},
            )

            def stream_generator_huggingface():
                try:
                    for line_bytes in response.iter_lines():
                        if line_bytes:
                            decoded_line = line_bytes.decode("utf-8").strip()
                            if not decoded_line:
                                continue  # Skip empty keep-alive lines

                            # logger.debug(f"HF Stream raw line: {decoded_line}")
                            if decoded_line.startswith("data:"):
                                data_content = decoded_line[len("data:") :].strip()
                                if data_content == "[DONE]":
                                    logger.debug(
                                        "HuggingFace stream received [DONE] marker."
                                    )
                                    break
                                try:
                                    chunk_json = json.loads(data_content)
                                    delta_content = (
                                        chunk_json.get("choices", [{}])[0]
                                        .get("delta", {})
                                        .get("content")
                                    )
                                    if delta_content:
                                        yield delta_content
                                    # Consider if other parts of the chunk are needed, e.g., finish_reason in delta
                                    # For now, just yielding content as per OpenAI's typical text stream delta.
                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"HuggingFace stream: JSON decode error for data: '{data_content}'"
                                    )
                except requests.exceptions.ChunkedEncodingError as e_chunked:
                    logger.error(
                        f"HuggingFace stream: ChunkedEncodingError during streaming: {e_chunked}"
                    )
                except Exception as e_stream:
                    logger.opt(exception=True).error(
                        f"HuggingFace stream: Unexpected error during streaming: {e_stream}"
                    )
                finally:
                    if response:
                        response.close()  # Ensure response is closed
                    logger.debug("HuggingFace stream generator finished.")

            return stream_generator_huggingface()
        else:  # Non-streaming
            logger.debug(
                "HuggingFace: Posting non-streaming request to "
                f"host={safe_llm_url_host(api_url)}"
            )
            adapter = HTTPAdapter(
                max_retries=Retry(
                    total=llm_retry_count(int(hf_config.get("api_retries", 3))),
                    backoff_factor=float(hf_config.get("api_retry_delay", 1)),
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=[
                        "POST"
                    ],  # Should be allowed_methods for Retry v0.9.2+ (urllib3)
                    # or method_whitelist for older versions.
                )
            )
            session = create_default_session()
            session.mount("https://", adapter)
            session.mount("http://", adapter)

            response = session.post(
                api_url, headers=headers, json=payload, timeout=timeout_seconds
            )
            response.raise_for_status()
            result = (
                response.json()
            )  # This should be an OpenAI compatible JSON response

            # Log non-streaming success metrics
            duration = time.time() - start_time
            log_histogram(
                "huggingface_api_response_time",
                duration,
                labels={
                    "model": final_model_for_payload,
                    "streaming": "false",
                    "status_code": str(response.status_code),
                },
            )
            log_counter(
                "huggingface_api_success",
                labels={"model": final_model_for_payload, "streaming": "false"},
            )

            # Log token usage if available
            usage = result.get("usage", {})
            if usage:
                log_histogram(
                    "huggingface_api_input_tokens",
                    usage.get("prompt_tokens", 0),
                    labels={"model": final_model_for_payload},
                )
                log_histogram(
                    "huggingface_api_output_tokens",
                    usage.get("completion_tokens", 0),
                    labels={"model": final_model_for_payload},
                )
                log_histogram(
                    "huggingface_api_total_tokens",
                    usage.get("total_tokens", 0),
                    labels={"model": final_model_for_payload},
                )

            return result

    except requests.exceptions.HTTPError as e:
        status_code = getattr(e.response, "status_code", 500)
        raw_error_text = getattr(e.response, "text", None)
        if raw_error_text is None:
            raw_error_text = safe_llm_exception_message(e)
        error_text = str(safe_llm_error_detail(raw_error_text))
        endpoint_copy = safe_llm_url_host(api_url)
        logger.error(
            "HuggingFace API call failed; "
            f"host={endpoint_copy}; status={status_code}; detail={error_text[:500]}"
        )

        # Log HTTP error metrics
        duration = time.time() - start_time
        log_counter(
            "huggingface_api_error",
            labels={
                "model": final_model_for_payload,
                "error_type": "http_error",
                "status_code": str(status_code),
            },
        )
        log_histogram(
            "huggingface_api_error_response_time",
            duration,
            labels={"model": final_model_for_payload, "status_code": str(status_code)},
        )
        if status_code == 401:
            raise ChatAuthenticationError(
                provider="huggingface",
                message=f"Authentication failed. Detail: {error_text[:200]}",
            )
        elif status_code == 404:  # Specifically handle 404 for URL/model issues
            raise ChatBadRequestError(
                provider="huggingface",
                message=f"Endpoint or Model not found (404) at {endpoint_copy}. Detail: {error_text[:200]}",
            )
        elif status_code == 429:
            raise ChatRateLimitError(
                provider="huggingface",
                message=f"Rate limit exceeded. Detail: {error_text[:200]}",
            )
        elif 400 <= status_code < 500:  # Other 4xx
            raise ChatBadRequestError(
                provider="huggingface",
                message=f"Bad request (Status {status_code}) to {endpoint_copy}. Detail: {error_text[:200]}",
            )
        else:  # 5xx
            raise ChatProviderError(
                provider="huggingface",
                message=f"Server error (Status {status_code}) from {endpoint_copy}. Detail: {error_text[:200]}",
                status_code=status_code,
            )
    except (
        requests.exceptions.RequestException
    ) as e:  # Covers DNS, Connection, Timeout errors
        endpoint_copy = safe_llm_url_host(api_url)
        error_detail = safe_llm_exception_message(e)
        error_copy = (
            "HuggingFace API request failed; reason=network_error; "
            f"host={endpoint_copy}; error_type={error_detail}"
        )
        if is_sensitive_llm_request():
            logger.error(error_copy)
        else:
            logger.opt(exception=True).error(error_copy)

        # Log network error metrics
        duration = time.time() - start_time
        log_counter(
            "huggingface_api_error",
            labels={"model": final_model_for_payload, "error_type": "network_error"},
        )
        log_histogram(
            "huggingface_api_error_response_time",
            duration,
            labels={"model": final_model_for_payload, "error_type": "network_error"},
        )
        raise ChatProviderError(
            provider="huggingface",
            message=f"Network error connecting to {endpoint_copy}: {error_detail}",
            status_code=504,
        )  # 504 for timeout/gateway like
    except Exception as e:
        endpoint_copy = safe_llm_url_host(api_url)
        error_detail = safe_llm_exception_message(e)
        error_copy = (
            "HuggingFace API call failed; reason=unexpected; "
            f"host={endpoint_copy}; error_type={error_detail}"
        )
        if is_sensitive_llm_request():
            logger.error(error_copy)
        else:
            logger.opt(exception=True).error(error_copy)

        # Log unexpected error metrics
        duration = time.time() - start_time
        log_counter(
            "huggingface_api_error",
            labels={"model": final_model_for_payload, "error_type": "unexpected_error"},
        )
        log_histogram(
            "huggingface_api_error_response_time",
            duration,
            labels={"model": final_model_for_payload, "error_type": "unexpected_error"},
        )
        if not isinstance(e, ChatAPIError):  # Avoid re-wrapping known chat errors
            raise ChatAPIError(
                provider="huggingface",
                message=f"Unexpected error in HuggingFace API call: {error_detail}",
            )
        else:
            raise  # Re-raise if it's already a ChatAPIError subtype


def chat_with_mistral(
    input_data: List[Dict[str, Any]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    system_message: Optional[str] = None,
    temp: Optional[float] = None,
    streaming: Optional[bool] = False,
    topp: Optional[float] = None,
    max_tokens: Optional[int] = None,
    random_seed: Optional[int] = None,
    top_k: Optional[int] = None,
    safe_prompt: Optional[bool] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    response_format: Optional[Dict[str, str]] = None,
    custom_prompt_arg: Optional[str] = None,
    api_base_url: Optional[str] = None,
):
    start_time = time.time()
    cli_api_settings = get_runtime_config_snapshot().values.get("api_settings", {})
    mistral_config = cli_api_settings.get(
        "mistral", {}
    )  # Get the [api_settings.mistral] sub-table
    final_api_key = api_key or mistral_config.get("api_key")
    if not final_api_key:
        raise ChatConfigurationError(
            provider="mistral", message="Mistral API Key required."
        )

    logger.debug("Mistral: API key provided.")
    current_model = model or mistral_config.get(
        "model", "mistral-large-latest"
    )  # or mistral-small, mistral-medium
    current_temp = (
        temp if temp is not None else float(mistral_config.get("temperature", 0.1))
    )  # Mistral defaults to 0.7
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
        else _safe_cast(mistral_config.get("max_tokens"), int)
    )
    current_safe_prompt = (
        safe_prompt
        if safe_prompt is not None
        else bool(mistral_config.get("safe_prompt", False))
    )

    # Log request metrics
    log_counter(
        "mistral_api_request",
        labels={"model": current_model, "streaming": str(current_streaming)},
    )

    api_messages = []
    # Mistral expects system message as the first message with role: system if provided
    # However, their latest guidance often shows it as part of the first user message or specific instructions.
    # For OpenAI compatibility, if system_message is given, and not already in input_data, prepend it.
    has_system_in_input = any(msg.get("role") == "system" for msg in input_data)
    if system_message and not has_system_in_input:
        api_messages.append({"role": "system", "content": system_message})
    api_messages.extend(input_data)

    headers = {
        "Authorization": f"Bearer {final_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    data = {
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
    # Note: Mistral API does not support top_k parameter
    if current_safe_prompt is not None:
        data["safe_prompt"] = current_safe_prompt  # Mistral specific
    if tools is not None:
        data["tools"] = tools
    if tool_choice is not None:
        data["tool_choice"] = tool_choice  # "auto", "any", "none"
    if response_format is not None:
        data["response_format"] = response_format  # {"type": "json_object"}

    api_url = (
        api_base_url
        or mistral_config.get("api_base_url")
        or builtin_provider_endpoint("mistralai", mistral_config)
    ).rstrip("/") + "/chat/completions"
    if not is_sensitive_llm_request():
        # task-2116: see the OpenAI branch above for why this is gated.
        # task-2117 Qodo round: allowlisted summary, see the Anthropic
        # branch above for why a denylist isn't safe here.
        logger.debug(
            "Mistral Request Payload (safe fields only): "
            f"{safe_llm_request_payload_summary(data)}"
        )

    try:
        if current_streaming:
            # ... (OpenAI-like streaming logic, use "Mistral" in logs) ...
            with create_default_session() as session:
                response = session.post(
                    api_url, headers=headers, json=data, stream=True, timeout=180
                )
                response.raise_for_status()

                # Log streaming success metrics
                duration = time.time() - start_time
                log_histogram(
                    "openrouter_api_response_time",
                    duration,
                    labels={
                        "model": current_model,
                        "streaming": "true",
                        "status_code": str(response.status_code),
                    },
                )
                log_counter(
                    "openrouter_api_success",
                    labels={"model": current_model, "streaming": "true"},
                )

                def stream_generator():
                    try:
                        for line in response.iter_lines(decode_unicode=True):
                            if line and line.strip():
                                yield line + "\n\n"
                    # ... (error handling for stream) ...
                    finally:
                        yield "data: [DONE]\n\n"
                        if response:
                            response.close()

                return stream_generator()
        else:
            # ... (non-streaming, retry) ...
            adapter = HTTPAdapter(
                max_retries=Retry(
                    total=llm_retry_count(int(mistral_config.get("api_retries", 3))),
                    backoff_factor=float(mistral_config.get("api_retry_delay", 1)),
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["POST"],
                )
            )
            with create_default_session() as session:
                session.mount("https://", adapter)
                response = session.post(
                    api_url, headers=headers, json=data, timeout=120
                )
            response.raise_for_status()
            result = response.json()

            # Log non-streaming success metrics
            duration = time.time() - start_time
            log_histogram(
                "mistral_api_response_time",
                duration,
                labels={
                    "model": current_model,
                    "streaming": "false",
                    "status_code": str(response.status_code),
                },
            )
            log_counter(
                "mistral_api_success",
                labels={"model": current_model, "streaming": "false"},
            )

            # Log token usage if available
            usage = result.get("usage", {})
            if usage:
                log_histogram(
                    "mistral_api_input_tokens",
                    usage.get("prompt_tokens", 0),
                    labels={"model": current_model},
                )
                log_histogram(
                    "mistral_api_output_tokens",
                    usage.get("completion_tokens", 0),
                    labels={"model": current_model},
                )
                log_histogram(
                    "mistral_api_total_tokens",
                    usage.get("total_tokens", 0),
                    labels={"model": current_model},
                )

            return result
    except requests.exceptions.HTTPError as e:  # ... error handling ...
        # Log HTTP error metrics
        duration = time.time() - start_time
        status_code = e.response.status_code if e.response is not None else 500
        log_counter(
            "mistral_api_error",
            labels={
                "model": current_model,
                "error_type": "http_error",
                "status_code": str(status_code),
            },
        )
        log_histogram(
            "mistral_api_error_response_time",
            duration,
            labels={"model": current_model, "status_code": str(status_code)},
        )
        raise
    except Exception as e:  # ... error handling ...
        # Log unexpected error metrics
        duration = time.time() - start_time
        log_counter(
            "mistral_api_error",
            labels={"model": current_model, "error_type": "unexpected_error"},
        )
        log_histogram(
            "mistral_api_error_response_time",
            duration,
            labels={"model": current_model, "error_type": "unexpected_error"},
        )
        raise ChatProviderError(provider="mistral", message=f"Unexpected error: {e}")


def chat_with_openrouter(
    input_data: List[Dict[str, Any]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    system_message: Optional[str] = None,
    temp: Optional[float] = None,
    streaming: Optional[bool] = False,
    # OpenRouter specific names from your map
    top_p: Optional[float] = None,  # from generic topp
    top_k: Optional[int] = None,  # from generic topk
    min_p: Optional[float] = None,  # from generic minp (OpenRouter uses min_p not minp)
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    response_format: Optional[Dict[str, str]] = None,
    n: Optional[int] = None,
    user: Optional[str] = None,  # from user_identifier
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    logit_bias: Optional[Dict[str, float]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    custom_prompt_arg: Optional[str] = None,
    api_base_url: Optional[str] = None,
):
    start_time = time.time()
    cli_api_settings = get_runtime_config_snapshot().values.get("api_settings", {})
    openrouter_config = cli_api_settings.get(
        "openrouter", {}
    )  # Get the [api_settings.cohere] sub-table
    # ... (api key, model, temp, streaming setup) ...
    final_api_key = api_key or openrouter_config.get("api_key")
    if not final_api_key:
        raise ChatConfigurationError(
            provider="openrouter", message="OpenRouter API Key required."
        )
    current_model = model or openrouter_config.get(
        "model", "mistralai/mistral-7b-instruct:free"
    )
    # ... other param resolutions ...
    current_streaming_cfg = openrouter_config.get("streaming", False)
    current_streaming = (
        streaming
        if streaming is not None
        else (
            str(current_streaming_cfg).lower() == "true"
            if isinstance(current_streaming_cfg, str)
            else bool(current_streaming_cfg)
        )
    )

    # Log request metrics
    log_counter(
        "openrouter_api_request",
        labels={"model": current_model, "streaming": str(current_streaming)},
    )

    api_messages = []
    if system_message:
        api_messages.append({"role": "system", "content": system_message})
    api_messages.extend(input_data)

    headers = {
        "Authorization": f"Bearer {final_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": openrouter_config.get(
            "site_url", "http://localhost"
        ),  # OpenRouter specific
        "X-Title": openrouter_config.get(
            "site_name", "TLDW-API"
        ),  # OpenRouter specific
    }
    data = {
        "model": current_model,
        "messages": api_messages,
        "stream": current_streaming,
    }
    # Add all other accepted parameters to data if they are not None
    if temp is not None:
        data["temperature"] = temp
    if top_p is not None:
        data["top_p"] = top_p
    if top_k is not None:
        data["top_k"] = top_k
    if min_p is not None:
        data["min_p"] = min_p  # OpenRouter uses min_p
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    if seed is not None:
        data["seed"] = seed
    if stop is not None:
        data["stop"] = stop
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
    if presence_penalty is not None:
        data["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        data["frequency_penalty"] = frequency_penalty
    if logprobs is not None:
        data["logprobs"] = logprobs
    if top_logprobs is not None and data.get("logprobs"):
        data["top_logprobs"] = top_logprobs

    api_url = (
        api_base_url
        or openrouter_config.get("api_base_url")
        or builtin_provider_endpoint("openrouter", openrouter_config)
    ).rstrip("/") + "/chat/completions"
    if not is_sensitive_llm_request():
        # task-2116: see the OpenAI branch above for why this is gated.
        # task-2117 Qodo round: allowlisted summary, see the Anthropic
        # branch above for why a denylist isn't safe here.
        logger.debug(
            "OpenRouter Request Payload (safe fields only): "
            f"{safe_llm_request_payload_summary(data)}"
        )

    try:
        if current_streaming:
            # ... (OpenAI-like streaming logic, ensure "OpenRouter" in logs) ...
            with create_default_session() as session:
                response = session.post(
                    api_url, headers=headers, json=data, stream=True, timeout=180
                )
                response.raise_for_status()

                # Log streaming success metrics
                duration = time.time() - start_time
                log_histogram(
                    "openrouter_api_response_time",
                    duration,
                    labels={
                        "model": current_model,
                        "streaming": "true",
                        "status_code": str(response.status_code),
                    },
                )
                log_counter(
                    "openrouter_api_success",
                    labels={"model": current_model, "streaming": "true"},
                )

                def stream_generator():
                    try:
                        for line in response.iter_lines(decode_unicode=True):
                            if line and line.strip():
                                yield line + "\n\n"
                    # ... (error handling for stream) ...
                    finally:
                        yield "data: [DONE]\n\n"
                        if response:
                            response.close()

                return stream_generator()
        else:
            # ... (non-streaming logic) ...
            # ... (retry setup) ...
            adapter = HTTPAdapter(
                max_retries=Retry(
                    total=llm_retry_count(int(openrouter_config.get("api_retries", 3))),
                    backoff_factor=float(openrouter_config.get("api_retry_delay", 1)),
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["POST"],
                )
            )
            with create_default_session() as session:
                session.mount("https://", adapter)
                response = session.post(
                    api_url, headers=headers, json=data, timeout=120
                )
            response.raise_for_status()
            result = (
                response.json()
            )  # OpenRouter usually returns OpenAI compatible JSON

            # Log non-streaming success metrics
            duration = time.time() - start_time
            log_histogram(
                "openrouter_api_response_time",
                duration,
                labels={
                    "model": current_model,
                    "streaming": "false",
                    "status_code": str(response.status_code),
                },
            )
            log_counter(
                "openrouter_api_success",
                labels={"model": current_model, "streaming": "false"},
            )

            # Log token usage if available
            usage = result.get("usage", {})
            if usage:
                log_histogram(
                    "openrouter_api_input_tokens",
                    usage.get("prompt_tokens", 0),
                    labels={"model": current_model},
                )
                log_histogram(
                    "openrouter_api_output_tokens",
                    usage.get("completion_tokens", 0),
                    labels={"model": current_model},
                )
                log_histogram(
                    "openrouter_api_total_tokens",
                    usage.get("total_tokens", 0),
                    labels={"model": current_model},
                )

            return result
    except requests.exceptions.HTTPError as e:  # ... error handling ...
        # Log HTTP error metrics
        duration = time.time() - start_time
        status_code = e.response.status_code if e.response is not None else 500
        log_counter(
            "openrouter_api_error",
            labels={
                "model": current_model,
                "error_type": "http_error",
                "status_code": str(status_code),
            },
        )
        log_histogram(
            "openrouter_api_error_response_time",
            duration,
            labels={"model": current_model, "status_code": str(status_code)},
        )
        raise
    except Exception as e:  # ... error handling ...
        # Log unexpected error metrics
        duration = time.time() - start_time
        log_counter(
            "openrouter_api_error",
            labels={"model": current_model, "error_type": "unexpected_error"},
        )
        log_histogram(
            "openrouter_api_error_response_time",
            duration,
            labels={"model": current_model, "error_type": "unexpected_error"},
        )
        raise ChatProviderError(provider="openrouter", message=f"Unexpected error: {e}")


def chat_with_moonshot(
    input_data: List[Dict[str, Any]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    system_message: Optional[str] = None,
    temp: Optional[float] = None,
    maxp: Optional[float] = None,
    streaming: Optional[bool] = False,
    frequency_penalty: Optional[float] = None,
    max_tokens: Optional[int] = None,
    n: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    response_format: Optional[Dict[str, str]] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    user: Optional[str] = None,
    custom_prompt_arg: Optional[str] = None,
    api_base_url: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    provider_continuations: Tuple[ProviderContinuationCheckpoint, ...] = (),
    request_timeout: Optional[float] = None,
    request_retries: Optional[int] = None,
    request_retry_delay: Optional[float] = None,
):
    """Compatibility wrapper for the strict first-class Moonshot adapter."""
    started_at = time.time()
    labels = {"model": model or "configured", "streaming": str(bool(streaming))}
    log_counter("moonshot_api_request", labels=labels)
    try:
        result = _strict_chat_with_moonshot(
            input_data=input_data,
            model=model,
            api_key=api_key,
            system_message=system_message,
            temp=temp,
            maxp=maxp,
            streaming=streaming,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            user=user,
            custom_prompt_arg=custom_prompt_arg,
            api_base_url=api_base_url,
            reasoning_effort=reasoning_effort,
            provider_continuations=provider_continuations,
            request_timeout=request_timeout,
            request_retries=request_retries,
            request_retry_delay=request_retry_delay,
        )
    except Exception as exc:
        log_counter(
            "moonshot_api_error",
            labels={**labels, "error_type": type(exc).__name__},
        )
        log_histogram(
            "moonshot_api_error_response_time",
            time.time() - started_at,
            labels=labels,
        )
        raise
    log_counter("moonshot_api_success", labels=labels)
    log_histogram(
        "moonshot_api_response_time",
        time.time() - started_at,
        labels=labels,
    )
    return result


def chat_with_zai(
    input_data: List[Dict[str, Any]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    system_message: Optional[str] = None,
    temp: Optional[float] = None,
    maxp: Optional[float] = None,
    streaming: Optional[bool] = False,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    do_sample: Optional[bool] = None,
    request_id: Optional[str] = None,
    custom_prompt_arg: Optional[str] = None,
    api_base_url: Optional[str] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    stop: Optional[Union[str, List[str]]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    user: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    provider_continuations: Tuple[ProviderContinuationCheckpoint, ...] = (),
    request_timeout: Optional[float] = None,
    request_retries: Optional[int] = None,
    request_retry_delay: Optional[float] = None,
):
    """Compatibility wrapper for the strict first-class Z.ai adapter."""
    started_at = time.time()
    labels = {"model": model or "configured", "streaming": str(bool(streaming))}
    log_counter("zai_api_request", labels=labels)
    try:
        result = _strict_chat_with_zai(
            input_data=input_data,
            model=model,
            api_key=api_key,
            system_message=system_message,
            temp=temp,
            maxp=maxp,
            streaming=streaming,
            max_tokens=max_tokens,
            tools=tools,
            do_sample=do_sample,
            request_id=request_id,
            custom_prompt_arg=custom_prompt_arg,
            api_base_url=api_base_url,
            tool_choice=tool_choice,
            stop=stop,
            response_format=response_format,
            user=user,
            reasoning_effort=reasoning_effort,
            provider_continuations=provider_continuations,
            request_timeout=request_timeout,
            request_retries=request_retries,
            request_retry_delay=request_retry_delay,
        )
    except Exception as exc:
        log_counter(
            "zai_api_error",
            labels={**labels, "error_type": type(exc).__name__},
        )
        log_histogram(
            "zai_api_error_response_time",
            time.time() - started_at,
            labels=labels,
        )
        raise
    log_counter("zai_api_success", labels=labels)
    log_histogram(
        "zai_api_response_time",
        time.time() - started_at,
        labels=labels,
    )
    return result


#
#
#######################################################################################################################
