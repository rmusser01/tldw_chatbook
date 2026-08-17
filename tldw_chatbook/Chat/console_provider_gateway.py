"""Console-native provider resolution and streaming gateway."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import threading
import weakref
from collections.abc import Iterator, Mapping
from contextvars import copy_context
from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import GeneratorType, MappingProxyType
from typing import Any, AsyncIterator, Callable, Literal, cast
from urllib.parse import urlparse, urlunparse

import httpx
from loguru import logger
from rich.markup import escape as escape_markup

from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_provider_endpoints import (
    effective_provider_endpoint,
    generic_endpoint_differs,
    normalize_generic_endpoint_for_compare,
    provider_uses_endpoint,
    unsaved_endpoint_copy,
)
from tldw_chatbook.Chat.console_prepared_request import (
    CONTINUATION_OWNER_KEY,
    PreparedConsoleRequest,
    PreparedProviderRequest,
    WireStyle,
    build_console_request,
    prepare_provider_request,
    resolve_request_capacity,
    thaw_json,
)
from tldw_chatbook.Chat.console_history_budget import (
    DEFAULT_PER_IMAGE_TOKENS,
    ProviderContinuationSidecar,
    is_deleted_history_value,
    provider_continuation_owner_groups,
)
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationConflictError,
    ContinuationRestoreTarget,
    ProviderContinuationCheckpoint,
    validate_continuation_restore,
)
from tldw_chatbook.Chat.console_provider_support import (
    build_local_thinking_payload_fields,
    resolve_console_provider_identity,
)
from tldw_chatbook.Chat.llamacpp_think_filter import StartAnchoredThinkFilter
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.LLM_Calls.qwencloud import (
    normalize_qwencloud_api_mode,
    normalize_qwencloud_base_url,
)
from tldw_chatbook.LLM_Calls.hosted_chat import HostedChatTurn
from tldw_chatbook.config import ProviderSettingsError, provider_settings_for_key
from tldw_chatbook.Utils.input_validation import validate_url
from tldw_chatbook.Utils.sensitive_llm_logging import (
    is_sensitive_llm_request,
    sensitive_llm_request,
)


DEFAULT_LLAMACPP_BASE_URL = "http://127.0.0.1:9099"
PROBE_TIMEOUT_SECONDS = 5.0
"""Per-request timeout for readiness probes (``/health``, ``/v1/models``)."""
GENERATION_CONNECT_TIMEOUT_SECONDS = 10.0
"""Connect timeout for the owned HTTP client used for generation calls."""
GENERATION_READ_TIMEOUT_SECONDS = 300.0
"""Read/write/pool timeout for generation calls.

Large local models routinely need 60-180s for a non-streamed completion, so
the owned client must not cap reads at the old 30s ceiling.
"""
INVALID_LLAMACPP_BASE_URL_COPY = (
    "Provider blocked: invalid llama.cpp base URL. "
    "Use an http(s) URL such as http://127.0.0.1:9099."
)
UNSUPPORTED_PROVIDER_RESPONSE_COPY = "Provider returned an unsupported response shape."
NO_PROVIDER_CONTENT_COPY = "Provider returned no assistant content."
_UNSUPPORTED_RESPONSE = object()
_EMPTY_RESPONSE = object()
_CUSTOM_CREDENTIAL_DECISION_PROVIDERS = frozenset(
    {"custom-openai-api", "custom-openai-api-2"}
)
MAX_AUXILIARY_OUTPUT_TOKENS = 16_384
"""Application hard ceiling for one auxiliary completion's output allowance."""
PROVIDER_ERROR_MODEL_ID_MAX_CHARS = 256
"""Maximum model-ID context included in user-visible provider error copy."""
_CONTINUATION_PROTOCOLS = frozenset({"chat_completions", "responses"})


def _normalize_deepseek_api_mode(provider_settings: Mapping[str, Any]) -> str:
    """Resolve ADR-064's pinned DeepSeek mode without changing legacy default."""
    candidate = provider_settings.get("api_mode", "chat_completions")
    if not isinstance(candidate, str):
        raise ChatConfigurationError("DeepSeek API mode must be a string.")
    normalized = candidate.strip().lower()
    if normalized not in _CONTINUATION_PROTOCOLS:
        raise ChatConfigurationError(
            "DeepSeek API mode must be 'responses' or 'chat_completions'."
        )
    return normalized


@dataclass(slots=True)
class ConsoleProviderStreamSignals:
    """Expose thread-safe provenance signals for one provider stream.

    The gateway marks these signals while normalizing provider output so the
    controller can distinguish provider content from locally synthesized
    fallback copy.
    """

    _synthetic_fallback: threading.Event = field(
        default_factory=threading.Event,
        init=False,
        repr=False,
    )
    # Usage for the provider call currently in flight. Key-merged, because a
    # single Anthropic call splits its usage across two SSE chunks
    # (message_start carries the input/cache buckets, message_delta the
    # output tokens). NEVER merged across CALLS -- see close_usage_call.
    usage_payload: dict[str, Any] | None = field(default=None, repr=False)
    # One entry per provider call that has already finished. An agent turn
    # makes N calls through the SAME signals object; key-merging those
    # together silently corrupts the bill (call 2's prompt_tokens landing
    # next to call 1's stale prompt_tokens_details.cached_tokens yields
    # uncached_input=0 plus a phantom cache read). Consumers normalize each
    # entry on its own and SUM the disjoint buckets instead.
    completed_usage_payloads: list[dict[str, Any]] = field(
        default_factory=list,
        repr=False,
    )
    _active_usage_payloads: dict[object, dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _usage_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    @property
    def synthetic_fallback_emitted(self) -> bool:
        """Return whether the stream emitted locally synthesized fallback copy."""
        return self._synthetic_fallback.is_set()

    def mark_synthetic_fallback(self) -> None:
        """Record that locally synthesized fallback copy was emitted."""
        self._synthetic_fallback.set()

    def record_usage_payload(self, payload: Mapping[str, Any]) -> None:
        """Merge a usage payload into the IN-FLIGHT provider call's payload."""
        with self._usage_lock:
            merged = dict(self.usage_payload or {})
            merged.update(payload)
            self.usage_payload = merged

    def close_usage_call(self) -> None:
        """Close the in-flight provider call out at its own call boundary.

        Called by the gateway when a ``stream_chat`` invocation ends -- the
        one place that knows where one provider call stops and the next
        begins. MOVES (never copies) the in-flight payload into
        ``completed_usage_payloads``, so a consumer that already billed the
        in-flight payload of an aborted stream can never bill it twice.
        """
        with self._usage_lock:
            if self.usage_payload is None:
                return
            self.completed_usage_payloads.append(self.usage_payload)
            self.usage_payload = None

    def usage_payloads(self) -> list[dict[str, Any]]:
        """Return every payload to bill: completed calls + any in flight.

        The in-flight tail is included for aborted streams, whose generator
        may never reach its own close-out before the controller persists.
        """
        with self._usage_lock:
            payloads = [dict(payload) for payload in self.completed_usage_payloads]
            if self.usage_payload is not None:
                payloads.append(dict(self.usage_payload))
            payloads.extend(
                dict(payload) for payload in self._active_usage_payloads.values()
            )
            return payloads

    def new_usage_call(self) -> "ConsoleProviderCallSignals":
        """Create an isolated usage recorder for one provider call.

        Returns:
            A call-scoped signal view publishing into this aggregate.
        """
        return ConsoleProviderCallSignals(self)

    def _record_scoped_usage_call(
        self,
        token: object,
        payload: Mapping[str, Any],
    ) -> None:
        with self._usage_lock:
            self._active_usage_payloads[token] = dict(payload)

    def _complete_scoped_usage_call(
        self,
        token: object,
        payload: Mapping[str, Any],
    ) -> None:
        with self._usage_lock:
            self._active_usage_payloads.pop(token, None)
            self.completed_usage_payloads.append(dict(payload))


@dataclass(slots=True)
class ConsoleProviderCallSignals:
    """Call-scoped signal view that publishes usage to one aggregate."""

    _aggregate: ConsoleProviderStreamSignals = field(repr=False)
    _token: object = field(default_factory=object, init=False, repr=False)
    _usage_payload: dict[str, Any] | None = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _usage_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    @property
    def synthetic_fallback_emitted(self) -> bool:
        """Return whether the aggregate emitted synthetic fallback usage."""
        return self._aggregate.synthetic_fallback_emitted

    def mark_synthetic_fallback(self) -> None:
        """Mark synthetic fallback usage on the aggregate signal."""
        self._aggregate.mark_synthetic_fallback()

    def record_usage_payload(self, payload: Mapping[str, Any]) -> None:
        """Merge a provider usage payload into this call's snapshot.

        Args:
            payload: Provider usage fields observed for this call.
        """
        with self._usage_lock:
            if self._closed:
                return
            merged = dict(self._usage_payload or {})
            merged.update(payload)
            self._usage_payload = merged
            self._aggregate._record_scoped_usage_call(self._token, merged)

    def close_usage_call(self) -> None:
        """Publish this call's final usage snapshot exactly once."""
        with self._usage_lock:
            if self._closed:
                return
            self._closed = True
            payload = (
                dict(self._usage_payload) if self._usage_payload is not None else None
            )
        if payload is not None:
            self._aggregate._complete_scoped_usage_call(self._token, payload)

    def usage_snapshot(self) -> dict[str, Any] | None:
        """Return a defensive copy of this call's current usage.

        Returns:
            The merged usage payload, or ``None`` before usage is observed.
        """
        with self._usage_lock:
            return (
                dict(self._usage_payload) if self._usage_payload is not None else None
            )


_ProviderStreamSignals = ConsoleProviderStreamSignals | ConsoleProviderCallSignals


def safe_provider_error_copy(provider: str, exc: BaseException) -> str:
    """Return safe user-visible provider failure copy.

    Args:
        provider: Provider name associated with the failed request.
        exc: Exception raised by the provider adapter.

    Returns:
        Redacted user-facing error text that categorizes the failure without
        including raw exception content.
    """
    category = "unexpected provider error"
    if isinstance(exc, ChatAuthenticationError):
        category = "authentication failed"
    elif isinstance(exc, ChatRateLimitError):
        category = "rate limit exceeded"
    elif isinstance(exc, ChatBadRequestError):
        category = "bad request"
    elif isinstance(exc, ChatConfigurationError):
        category = "configuration error"
    elif isinstance(exc, ChatProviderError):
        category = "provider unavailable"
    status_code = getattr(exc, "status_code", None)
    status_copy = f" Status: {status_code}." if isinstance(status_code, int) else ""
    return f"Provider error from {provider or 'unknown'}: {category}.{status_copy}"


def _provider_error_copy_with_model_recovery(
    copy: str,
    *,
    model: str | None,
    status_code: int | None,
) -> str:
    """Add safe model-specific recovery to provider bad-request copy."""
    if status_code != 400:
        return copy
    model_id = "".join(
        character for character in str(model or "").strip() if character.isprintable()
    )[:PROVIDER_ERROR_MODEL_ID_MAX_CHARS]
    if not model_id:
        return copy
    return (
        f"{copy} Selected model: {escape_markup(model_id)}. "
        "The provider rejected this request. Confirm the model is still "
        "available, or choose another model from the model picker."
    )


def normalize_llamacpp_base_url(api_url: str | None) -> str:
    """Return the llama.cpp origin root used before appending OpenAI paths.

    Args:
        api_url: User or config-provided llama.cpp endpoint.

    Returns:
        Normalized origin/base path for llama.cpp HTTP calls.
    """
    raw_url = str(api_url or "").strip()
    if not raw_url:
        return DEFAULT_LLAMACPP_BASE_URL

    candidate = raw_url if "://" in raw_url else f"http://{raw_url}"
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return raw_url.rstrip("/")

    path = parsed.path.rstrip("/")
    normalized_endpoint_paths = {
        "/v1",
        "/v1/models",
        "/models",
        "/v1/chat/completions",
        "/chat/completions",
        "/completion",
        "/completions",
    }
    if path.lower() in normalized_endpoint_paths:
        path = ""
    normalized = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            path,
            "",
            "",
            "",
        )
    ).rstrip("/")
    return normalized or DEFAULT_LLAMACPP_BASE_URL


@dataclass(frozen=True)
class LlamaCppProviderConfig:
    """Configuration needed to resolve a llama.cpp-compatible provider.

    Attributes:
        base_url: llama.cpp server base URL.
        explicit_model: Session-selected model, when present.
        configured_model: Provider-configured fallback model.
        temperature: Optional sampling temperature.
        top_p: Optional nucleus sampling value.
        min_p: Optional min-p sampling value.
        top_k: Optional top-k sampling value.
        max_tokens: Optional response token limit.
        seed: Optional deterministic generation seed.
        presence_penalty: Optional presence penalty value.
        frequency_penalty: Optional frequency penalty value.
        reasoning_effort: Optional OpenAI-style reasoning effort.
        reasoning_summary: Optional OpenAI-style reasoning summary detail.
        verbosity: Optional OpenAI-style verbosity hint.
        thinking_effort: Optional Anthropic-style thinking effort.
        thinking_budget_tokens: Optional Anthropic-style thinking token budget.
        streaming: Whether streaming responses are requested.
    """

    base_url: str = DEFAULT_LLAMACPP_BASE_URL
    explicit_model: str | None = None
    configured_model: str | None = None
    api_key: str | None = field(default=None, repr=False)
    api_key_source: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    min_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    verbosity: str | None = None
    thinking_effort: str | None = None
    thinking_budget_tokens: int | None = None
    streaming: bool = True


@dataclass(frozen=True)
class ConsoleProviderResolution:
    """Provider readiness result used by Console send and recovery UI.

    Attributes:
        provider: Display provider selected for the session.
        base_url: Session endpoint value, when applicable.
        model: Model selected for the request.
        ready: Whether the provider has enough configuration to send.
        visible_copy: User-visible blocker or recovery copy.
        readiness_key: Normalized key used for readiness checks.
        execution_key: Provider key passed to ``chat_api_call``.
        api_key: Resolved API key, omitted from repr output.
        api_key_source: Human-readable source of the resolved API key.
        temperature: Optional sampling temperature.
        top_p: Optional nucleus sampling value.
        min_p: Optional min-p sampling value.
        top_k: Optional top-k sampling value.
        max_tokens: Optional response token limit.
        streaming: Whether streaming responses are requested.
        prompt_caching: Opt-in for the Anthropic per-turn ``cache_control``
            breakpoint. Set only for Anthropic resolutions (and only when
            ``[caching] anthropic_enabled`` is on); ``None`` everywhere else,
            which drops the kwarg entirely in ``_chat_api_kwargs``.
        api_mode: Pinned QwenCloud or DeepSeek wire mode; ``None`` elsewhere.
    """

    provider: str
    base_url: str
    model: str | None
    ready: bool
    visible_copy: str = ""
    readiness_key: str = ""
    execution_key: str = ""
    api_key: str | None = field(default=None, repr=False)
    api_key_source: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    min_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    verbosity: str | None = None
    thinking_effort: str | None = None
    thinking_budget_tokens: int | None = None
    streaming: bool = True
    prompt_caching: bool | None = None
    api_mode: str | None = None
    continuation_protocol: str | None = None
    request_timeout: float | None = None
    request_retries: int | None = None
    request_retry_delay: float | None = None


def _freeze_auxiliary_value(value: Any) -> Any:
    """Copy a JSON-safe request value into an immutable representation."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Auxiliary mapping keys must be strings.")
        return MappingProxyType(
            {key: _freeze_auxiliary_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_auxiliary_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Auxiliary numeric values must be finite.")
        return value
    raise TypeError(
        "Auxiliary values must be JSON-safe scalars, mappings, or sequences."
    )


def _thaw_auxiliary_value(value: Any) -> Any:
    """Return provider-compatible mutable containers from frozen request data."""

    if isinstance(value, Mapping):
        return {str(key): _thaw_auxiliary_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_auxiliary_value(item) for item in value]
    return value


@dataclass(frozen=True)
class AuxiliaryCompletionRequest:
    """Immutable, content-sensitive input to one auxiliary provider call."""

    resolution: ConsoleProviderResolution = field(repr=False)
    messages: tuple[Mapping[str, Any], ...] = field(repr=False)
    response_format: Mapping[str, Any] | None = field(repr=False)
    max_output_tokens: int
    sensitive: Literal[True] = True

    def __post_init__(self) -> None:
        if not isinstance(self.resolution, ConsoleProviderResolution):
            raise TypeError("resolution must be a ConsoleProviderResolution")
        if not self.resolution.ready:
            raise ValueError("Pinned provider resolution is not ready.")
        if (
            not isinstance(self.resolution.model, str)
            or not self.resolution.model.strip()
        ):
            raise ValueError("Pinned provider model is required.")
        if not isinstance(self.messages, tuple) or not self.messages:
            raise TypeError("messages must be a non-empty tuple")

        frozen_messages: list[Mapping[str, Any]] = []
        for message in self.messages:
            if not isinstance(message, Mapping):
                raise TypeError("Each auxiliary message must be a mapping.")
            if set(message) != {"role", "content"}:
                raise ValueError("Auxiliary messages contain only role and content.")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role.strip():
                raise ValueError("Auxiliary message role is required.")
            if not isinstance(content, str):
                raise TypeError("Auxiliary message content must be text.")
            frozen_messages.append(
                cast(Mapping[str, Any], _freeze_auxiliary_value(message))
            )

        if self.response_format is not None and not isinstance(
            self.response_format, Mapping
        ):
            raise TypeError("response_format must be a mapping or None")
        if isinstance(self.max_output_tokens, bool) or not isinstance(
            self.max_output_tokens, int
        ):
            raise TypeError("max_output_tokens must be an integer")
        if not 0 < self.max_output_tokens <= MAX_AUXILIARY_OUTPUT_TOKENS:
            raise ValueError(
                f"max_output_tokens must be between 1 and {MAX_AUXILIARY_OUTPUT_TOKENS}"
            )
        if self.sensitive is not True:
            raise ValueError("Auxiliary completions must be sensitive.")

        object.__setattr__(self, "messages", tuple(frozen_messages))
        if self.response_format is not None:
            object.__setattr__(
                self,
                "response_format",
                cast(Mapping[str, Any], _freeze_auxiliary_value(self.response_format)),
            )


@dataclass(frozen=True)
class AuxiliaryCompletionResult:
    """Exact text and pinned provider identity returned by an auxiliary call."""

    provider: str
    model: str
    text: str = field(repr=False)
    usage: ProviderUsage | None = None


@dataclass(frozen=True)
class _QueueItem:
    kind: str
    text: str = ""
    payload: Any = None
    # F5: the real HTTP status, carried alongside the (already-redacted)
    # text -- never re-derived by parsing that text back out. `None` means
    # "no real status available" (a bare RuntimeError, say), which the
    # consumer maps to ChatProviderError's own upstream-error default.
    status_code: int | None = None

    @classmethod
    def content(cls, text: str) -> "_QueueItem":
        return cls("content", text)

    @classmethod
    def error(cls, text: str, status_code: int | None = None) -> "_QueueItem":
        return cls("error", text, status_code=status_code)

    @classmethod
    def done(cls) -> "_QueueItem":
        return cls("done")

    @classmethod
    def native_tool_calls(
        cls,
        calls: tuple[dict, ...],
        metadata: ProviderTurnMetadata | None = None,
    ) -> "_QueueItem":
        return cls("tool_calls", payload=ProviderToolCalls(calls, metadata=metadata))


@dataclass(frozen=True)
class ProviderTurnMetadata:
    """Typed terminal state for one completed provider call."""

    finish_reason: str
    provider_continuation: ProviderContinuationCheckpoint | None = field(
        default=None, repr=False
    )
    usage: Mapping[str, Any] | None = field(default=None, repr=False)


@dataclass(frozen=True)
class ProviderToolCalls:
    """Accumulated native tool-calls, yielded as ``stream_chat``'s FINAL
    item -- and only when the caller passed ``tools=``. Plain Console sends
    never receive one. ``tool_calls`` entries are OpenAI-shape dicts with
    streaming fragments already merged."""

    tool_calls: tuple[dict, ...]
    metadata: ProviderTurnMetadata | None = field(default=None, repr=False)


def _provider_turn_metadata(response: Any) -> ProviderTurnMetadata | None:
    """Read one provider-local terminal turn after clean exhaustion."""

    try:
        turn = response.terminal_turn
    except AttributeError:
        return None
    if not isinstance(turn, HostedChatTurn):
        raise ChatProviderError("Provider terminal metadata is malformed.")
    candidate = getattr(response, "provider_continuation", None)
    if candidate is not None and not isinstance(
        candidate, ProviderContinuationCheckpoint
    ):
        raise ChatProviderError("Provider continuation metadata is malformed.")
    usage = deepcopy(turn.usage) if isinstance(turn.usage, Mapping) else None
    return ProviderTurnMetadata(
        finish_reason=turn.finish_reason,
        provider_continuation=candidate,
        usage=usage,
    )


_PRESERVED_FRAGMENT_EXTRAS = frozenset(
    {
        # Gemini 3 thought signatures — must round-trip verbatim (task-266).
        "google_thought_signature",
        # Cohere v2 tool_plan text, echoed back on the request's assistant
        # tool_calls turn (task-267).
        "cohere_tool_plan",
    }
)


class _ToolCallAccumulator:
    """Merges OpenAI streaming ``delta.tool_calls`` fragments (and
    non-streaming ``message.tool_calls`` entries) into complete calls."""

    def __init__(self) -> None:
        self._by_index: dict[int, dict] = {}

    def feed_payload(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            return
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        first = choices[0]
        if not isinstance(first, Mapping):
            return
        message = first.get("message")
        if isinstance(message, Mapping):
            for i, raw in enumerate(message.get("tool_calls") or []):
                if isinstance(raw, Mapping):
                    self._merge(i, raw)
        delta = first.get("delta")
        if isinstance(delta, Mapping):
            for raw in delta.get("tool_calls") or []:
                if isinstance(raw, Mapping):
                    try:
                        index = int(raw.get("index", 0))
                    except (TypeError, ValueError):
                        index = 0
                    self._merge(index, raw)

    def _merge(self, index: int, fragment: Mapping[str, Any]) -> None:
        if index not in self._by_index:
            self._by_index[index] = {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        entry = self._by_index[index]
        if fragment.get("id"):
            entry["id"] = str(fragment["id"])
        if fragment.get("type"):
            entry["type"] = str(fragment["type"])
        function = fragment.get("function")
        if isinstance(function, Mapping):
            if function.get("name"):
                entry["function"]["name"] = str(function["name"])
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                entry["function"]["arguments"] += arguments
            elif isinstance(arguments, Mapping):
                entry["function"]["arguments"] = json.dumps(arguments)
        # Preserve KNOWN provider-specific extra keys verbatim (last-wins;
        # falsy-but-present survives, None drops) — e.g. Gemini 3 thought
        # signatures, which the request converter must echo back (task-266
        # live gate). Allow-listed rather than open-ended so a quirky
        # provider can't inject arbitrary keys that get echoed into the
        # next request (PR #662 final-review minor).
        for key in _PRESERVED_FRAGMENT_EXTRAS:
            if key in fragment and fragment[key] is not None:
                entry[key] = fragment[key]

    def calls(self) -> tuple[dict, ...]:
        # Numeric index order, not first-seen order: the provider's index
        # field defines the batch's array position, and fragments may arrive
        # interleaved/out of order (PR #648 review).
        return tuple(
            self._by_index[i]
            for i in sorted(self._by_index)
            if self._by_index[i]["function"]["name"]
        )


def _decode_stream_item(item: Any) -> Any:
    """Best-effort payload decode for accumulator teeing: mappings pass
    through; SSE ``data: {...}`` strings/bytes are JSON-decoded; anything
    else (comments, [DONE], junk) yields None."""
    if isinstance(item, Mapping):
        return item
    if isinstance(item, bytes):
        try:
            item = item.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(item, str):
        return None
    data = item.strip()
    if data.startswith("data:"):
        data = data.removeprefix("data:").strip()
    if not data or data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _tee_tool_calls(response: Any, accumulator: _ToolCallAccumulator) -> Any:
    """Feed every provider item through ``accumulator``, unchanged, for the
    three shapes ``chat_api_call`` returns: a full mapping (non-streaming),
    an iterator of mappings, or an iterator of SSE strings."""
    if isinstance(response, Mapping):
        try:
            accumulator.feed_payload(response)
        except BaseException:
            close = getattr(response, "close", None)
            if callable(close):
                with contextlib.suppress(BaseException):
                    close()
            raise
        return response
    if not _is_iterable_response(response):
        return response

    class _ToolCallTee(Iterator[Any]):
        def __init__(self) -> None:
            self._close_lock = threading.Lock()
            self._closed = False

        def __next__(self) -> Any:
            if self._is_closed():
                raise StopIteration
            try:
                item = next(response)
            except BaseException:
                self.close()
                raise
            if self._is_closed():
                raise StopIteration
            try:
                payload = _decode_stream_item(item)
            except BaseException:
                self.close()
                raise
            if self._is_closed():
                raise StopIteration
            try:
                accumulator.feed_payload(payload)
            except BaseException:
                self.close()
                raise
            if self._is_closed():
                raise StopIteration
            return item

        def _is_closed(self) -> bool:
            with self._close_lock:
                return self._closed

        def close(self) -> None:
            with self._close_lock:
                if self._closed:
                    return
                self._closed = True
            close = getattr(response, "close", None)
            if callable(close):
                with contextlib.suppress(BaseException):
                    close()

    return _ToolCallTee()


def build_llamacpp_chat_payload(
    *,
    model: str,
    messages: list[Mapping[str, Any]],
    stream: bool,
    temperature: float | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    reasoning_effort: str | None = None,
    thinking_budget_tokens: int | None = None,
) -> dict[str, Any]:
    """Build the OpenAI-compatible llama.cpp chat completion payload.

    Args:
        model: Model identifier to send to llama.cpp.
        messages: OpenAI-compatible chat messages.
        stream: Whether llama.cpp should stream chunks.
        temperature: Optional sampling temperature.
        top_p: Optional nucleus sampling value.
        min_p: Optional min-p sampling value.
        top_k: Optional top-k sampling value.
        max_tokens: Optional response token limit.
        seed: Optional deterministic generation seed.
        presence_penalty: Optional presence penalty value.
        frequency_penalty: Optional frequency penalty value.
        reasoning_effort: Optional thinking level forwarded as llama.cpp
            ``chat_template_kwargs.reasoning_effort`` (``none`` additionally
            sets ``enable_thinking`` false).
        thinking_budget_tokens: Optional thinking token budget sent as the
            top-level ``reasoning_budget_tokens`` field.

    Returns:
        Request payload for the llama.cpp chat completions endpoint.

    A trailing ``assistant`` message in ``messages`` is a response prefill:
    the Console's response-prefill send path (a one-shot ``/prefill`` or a
    pinned prefill applied to submit/retry/regenerate) appends the pending
    prefill text as the last message so llama.cpp continues generating from
    it. This is distinct from "continue from here", which never sends a
    trailing-assistant message -- it instead appends a synthetic user
    instruction asking the model to continue its prior reply. llama.cpp
    rejects prefilled requests when the chat template's thinking mode is
    enabled (``Assistant response prefill is incompatible with
    enable_thinking``), and forcing the response's opening text is
    incoherent with a thinking-first template regardless. Prefilled
    requests therefore disable thinking mode via ``chat_template_kwargs``,
    which templates that lack the kwarg -- and older servers that drop
    unknown fields -- simply ignore. When both a prefill and explicit
    thinking controls are present the precedence is
    ``prefill > none > effort``: the prefill's ``enable_thinking: False``
    always wins over the requested effort level, and an effort of ``none``
    itself disables thinking.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "stream": stream,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if min_p is not None:
        payload["min_p"] = min_p
    if top_k is not None:
        payload["top_k"] = top_k
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if seed is not None:
        payload["seed"] = seed
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty
    payload.update(
        build_local_thinking_payload_fields(
            "llama_cpp", reasoning_effort, thinking_budget_tokens
        )
    )
    if messages and messages[-1].get("role") == "assistant":
        template_kwargs = dict(payload.get("chat_template_kwargs") or {})
        template_kwargs["enable_thinking"] = False
        payload["chat_template_kwargs"] = template_kwargs
    return payload


class ConsoleProviderGateway:
    """Resolve Console providers and stream chat responses.

    Args:
        http_client: Optional HTTP client for llama.cpp probes and calls. When
            omitted, an owned client is created with a generous read timeout
            (``GENERATION_READ_TIMEOUT_SECONDS``) so slow local generations do
            not fail mid-request; probes always pass a short per-request
            timeout (``PROBE_TIMEOUT_SECONDS``).
        config_provider: Callable returning the current app configuration.
        environ: Optional environment mapping for provider readiness checks.
        chat_api_call_fn: Optional replacement for ``chat_api_call`` in tests.
        safe_error_copy: Optional error redaction callback.
    """

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        config_provider: Callable[[], Mapping[str, object]] | None = None,
        environ: Mapping[str, str] | None = None,
        chat_api_call_fn: Callable[..., Any] | None = None,
        safe_error_copy: Callable[[str, BaseException], str] | None = None,
    ) -> None:
        self._owns_http_client = http_client is None
        self.http_client = http_client or self._new_owned_http_client()
        # Mirrors whichever entry in `_loop_clients` was most recently
        # resolved by `_active_http_client` -- kept for the "unclaimed
        # client" escape hatch below and for callers/tests that read
        # `http_client`/`_client_loop` directly before any loop has touched
        # the gateway.
        self._client_loop: asyncio.AbstractEventLoop | None = None
        # Whether the client built above (or injected) has ever been
        # resolved through `_active_http_client`/`aclose`'s "unclaimed
        # client" escape hatches. `_client_loop is None` is NOT a safe proxy
        # for "never claimed": `aclose()` resets it to `None` on every
        # teardown, including after a real loop already claimed and
        # released a client, so re-deriving "unclaimed" from it would let a
        # SECOND loop adopt a client a FIRST loop already bound internally
        # (httpx/httpcore lazily bind connection-pool locks to whichever
        # loop first touches them) -- silently reintroducing the very
        # cross-loop binding failure this per-loop cache exists to
        # eliminate. This flag is set exactly once, on the first-ever
        # resolution, and `aclose()` never restores it.
        self._client_ever_claimed = False
        # Guards every read-check-create of `_loop_clients` (and the mirror
        # fields above) as a single atomic critical section (PR #629 Fix
        # 1(a), preserved across the move to a per-loop cache below):
        # concurrent callers on different loops/threads -- e.g. the app
        # loop's readiness probe racing the agent worker thread's per-turn
        # loop -- must never interleave a read with another caller's write,
        # which could otherwise desync `http_client`/`_client_loop` from
        # `_loop_clients`, or race two callers into each building their own
        # client for what should be the same cache slot.
        self._client_lock = threading.Lock()
        # Per-loop client cache (TASK-1064 item 1, same shape as the fix
        # applied to `GitHubAPIClient` under TASK-981 / PR #1009): every
        # event loop that is *currently alive* and has touched
        # `_active_http_client()` gets its own owned `httpx.AsyncClient`, so
        # two loops alive at the same time -- the app's own event loop
        # awaiting a readiness probe while an agent-runtime generation call
        # is bridged in from a worker thread's fresh per-turn
        # ``asyncio.run()`` -- never fight over, or close, each other's
        # client. A single-slot cache (the previous design) discarded and
        # scheduled `aclose()` of whatever the *other*, still-live loop was
        # using on every cross-loop touch -- closing a client mid-flight.
        # Keyed by the loop object itself via a `WeakKeyDictionary` so an
        # entry can be reclaimed as soon as nothing else references that
        # loop; pruned proactively in `_prune_closed_loops` so a long-running
        # process that bridges many short-lived per-turn loops over time
        # doesn't accumulate dead entries waiting on GC alone.
        self._loop_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = weakref.WeakKeyDictionary()
        self._config_provider = config_provider or (lambda: {})
        self._environ = environ
        self._chat_api_call_fn = chat_api_call_fn
        self._safe_error_copy = safe_error_copy or safe_provider_error_copy

    async def aclose(self) -> None:
        """Close the HTTP client(s) owned by this instance.

        The client bound to the caller's current running loop (if any) is
        closed directly -- safe, since we are already running on that loop.
        Every other cached per-loop client whose loop is IDLE -- e.g. one
        built earlier by the app's long-lived loop, still alive while a
        shorter-lived per-turn loop is the one calling ``aclose()`` -- is
        closed best-effort via ``_schedule_stale_client_close`` on its own
        loop; this never awaits, and never closes, a client bound to a loop
        it is not currently running on.

        PR3a-1 Task 6b (audit F5): a cached loop that is still RUNNING is
        skipped and its entry RETAINED. Such a loop is somebody's live
        transport -- since PR3a-1 Task 1, typically a fleet child's
        ``_ModelCallLifeline``, which outlives the turn that spawned it --
        and scheduling ``client.aclose()`` onto it closes the connection
        pool that child is actively issuing requests through. Each is
        closed by its own owner's teardown instead; see the inline comment
        at the sweep for the full argument.

        Returns:
            ``None``. Injected HTTP clients are left open for their owner,
            and so are clients belonging to loops still running.
        """
        if not self._owns_http_client:
            return
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        with self._client_lock:
            self._prune_closed_loops()
            current_client: httpx.AsyncClient | None = None
            if loop is not None:
                current_client = self._loop_clients.pop(loop, None)
                if current_client is None and not self._client_ever_claimed:
                    # Never claimed by any loop yet -- e.g. `aclose()` is
                    # called right after construction, before
                    # `_active_http_client()` was ever touched. Treat the
                    # client built in `__init__` as belonging to the loop
                    # calling `aclose()` now (mirrors the escape hatch in
                    # `_active_http_client`). `_client_ever_claimed` -- not
                    # `_client_loop is None` -- is the correct test here: it
                    # is set exactly once, on first-ever resolution, and is
                    # never restored by a prior teardown, so a client a
                    # loop already bound and released can never be
                    # re-treated as "unclaimed" by this branch.
                    current_client = self.http_client
                    self._client_ever_claimed = True
            others: list[
                tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]
            ] = []
            still_live: list[
                tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]
            ] = []
            for other_loop, other_client in self._loop_clients.items():
                if other_client is current_client:
                    continue
                # PR3a-1 Task 6b (audit F5): a RUNNING loop is somebody's
                # live transport, not a leftover. `_prune_closed_loops`
                # above already dropped the finished per-turn loops, and
                # the remaining ones that are still spinning `run_forever`
                # are fleet children's `_ModelCallLifeline`s -- which now
                # outlive the turn that spawned them (PR3a-1 Task 1), so
                # scheduling `client.aclose()` onto such a loop closes the
                # pool a child is actively issuing requests through.
                # Reproduced by execution in `Tests/Chat/test_console_
                # provider_gateway.py::test_aclose_does_not_close_a_still_
                # running_childs_client`.
                if other_loop.is_running():
                    still_live.append((other_loop, other_client))
                    continue
                others.append((other_loop, other_client))
            self._loop_clients.clear()
            # Retained, not merely spared: the child's next
            # `_active_http_client()` must find the SAME pool rather than
            # build a fresh one per call for the rest of its life.
            # Each is closed by its own owner's teardown -- a
            # `_ModelCallLifeline` closes its loop when the child ends, at
            # which point `_prune_closed_loops` drops the entry and the
            # client's own finalizer releases the sockets (the same
            # reasoning that method's docstring already relies on).
            for live_loop, live_client in still_live:
                self._loop_clients[live_loop] = live_client
            self._client_loop = None

        for other_loop, other_client in others:
            self._schedule_stale_client_close(other_client, other_loop)

        if current_client is not None:
            await current_client.aclose()

    def prepare_chat_request(
        self,
        resolution: ConsoleProviderResolution,
        messages: list[Mapping[str, Any]] | PreparedConsoleRequest,
        *,
        tools: list[Mapping[str, Any]] | None = None,
        context_window_override_tokens: int | None = None,
        apply_safety_window: bool = True,
        response_format: Mapping[str, Any] | None = None,
        continuation_target: ContinuationRestoreTarget | None = None,
        continuation_sidecar: tuple[ProviderContinuationSidecar, ...] = (),
        continuation_owner_key: str | None = None,
    ) -> PreparedProviderRequest:
        """Prepare the one immutable payload later consumed by dispatch.

        Model capability facts are read once here.  Unknown models remain
        explicitly unverified; an optional user override is enforced as a
        bound but never labeled as provider-verified.
        """

        if isinstance(messages, PreparedConsoleRequest) and tools is not None:
            raise ValueError("tools are already owned by PreparedConsoleRequest")
        sidecar = tuple(continuation_sidecar)
        if sidecar and (continuation_target is None or not continuation_owner_key):
            raise ValueError(
                "continuation target and owner key are required for private history"
            )
        if continuation_target is not None and (
            continuation_target.provider,
            continuation_target.model,
            normalize_generic_endpoint_for_compare(
                continuation_target.api_base_url
            ),
        ) != (
            provider_config_key(resolution.provider),
            resolution.model or "",
            normalize_generic_endpoint_for_compare(resolution.base_url),
        ):
            raise ContinuationConflictError(
                "Continuation restore target mismatch."
            ) from None
        if (
            continuation_target is not None
            and resolution.continuation_protocol is not None
            and continuation_target.protocol != resolution.continuation_protocol
        ):
            raise ContinuationConflictError(
                "Continuation restore target mismatch."
            ) from None
        if isinstance(messages, PreparedConsoleRequest):
            continuation_groups = (
                tuple(
                    group
                    for unit in messages.compactable
                    for group in unit.continuation_groups
                )
                + messages.active_continuation_groups
            )
            if continuation_groups and continuation_target is None:
                raise ValueError(
                    "continuation_target is required for provider continuation history"
                )
            if continuation_target is not None:
                for group in continuation_groups:
                    validate_continuation_restore(group.checkpoint, continuation_target)
            semantic = messages
        elif not sidecar:
            if any("provider_continuation" in message for message in messages):
                raise ValueError(
                    "continuation_target is required for provider continuation history"
                )
            semantic = build_console_request(messages, tools=tools or ())
        else:
            assert continuation_target is not None
            assert continuation_owner_key is not None
            selected_owner_ids = {
                message.get(continuation_owner_key)
                for message in messages
                if not is_deleted_history_value(message.get("deleted"))
                and type(message.get(continuation_owner_key)) is str
            }
            selected_sidecar = tuple(
                item for item in sidecar if item.owner_message_id in selected_owner_ids
            )
            continuation_groups = provider_continuation_owner_groups(
                selected_sidecar, target=continuation_target
            )
            owner_ids = {group.owner_message_id for group in continuation_groups}
            visible_messages: list[dict[str, Any]] = []
            for message in messages:
                if is_deleted_history_value(message.get("deleted")):
                    continue
                row = dict(message)
                owner_id = row.pop(continuation_owner_key, None)
                row.pop("provider_continuation", None)
                row.pop("deleted", None)
                if type(owner_id) is str and owner_id in owner_ids:
                    row[CONTINUATION_OWNER_KEY] = owner_id
                visible_messages.append(row)
            semantic = build_console_request(
                visible_messages,
                tools=tools or (),
                continuation_groups=continuation_groups,
            )

        capabilities: Mapping[str, Any] = {}
        try:
            from tldw_chatbook.model_capabilities import get_model_capabilities

            capabilities = get_model_capabilities().get_model_capabilities(
                resolution.provider,
                resolution.model or "",
            )
        except Exception:
            logger.debug("console_request_capability_lookup_failed")

        def positive_cap(*names: str) -> int | None:
            for name in names:
                value = capabilities.get(name)
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    return value
            return None

        capacity = resolve_request_capacity(
            context_window_tokens=positive_cap("context_window"),
            provider_input_cap_tokens=positive_cap(
                "max_input_tokens", "input_token_limit", "provider_input_cap"
            ),
            provider_output_cap_tokens=positive_cap(
                "max_output_tokens", "output_token_limit", "provider_output_cap"
            ),
            requested_response_tokens=resolution.max_tokens,
            context_window_override_tokens=context_window_override_tokens,
        )
        wire_style: WireStyle = (
            "distinct_roles"
            if resolution.provider in {"llama_cpp", "local_llamacpp"}
            else "single_preamble"
        )
        return prepare_provider_request(
            semantic,
            wire_style=wire_style,
            model=resolution.model or "",
            provider=resolution.provider,
            capacity=capacity,
            per_image_tokens=(
                positive_cap("image_input_tokens", "image_tokens", "per_image_tokens")
                or DEFAULT_PER_IMAGE_TOKENS
            ),
            apply_safety_window=apply_safety_window,
            response_format=response_format,
        )

    @staticmethod
    def _new_owned_http_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=GENERATION_CONNECT_TIMEOUT_SECONDS,
                read=GENERATION_READ_TIMEOUT_SECONDS,
                write=GENERATION_READ_TIMEOUT_SECONDS,
                pool=GENERATION_READ_TIMEOUT_SECONDS,
            )
        )

    def _prune_closed_loops(self) -> None:
        """Drop ``_loop_clients`` entries whose owning loop has closed.

        Must be called with ``_client_lock`` held. A ``WeakKeyDictionary``
        alone would eventually reclaim these once the loop object itself is
        garbage collected, but asyncio loops can take a while to be
        collected (reference cycles via internal callbacks), and there is
        nothing left to gracefully close on an already-closed loop -- its
        client's own finalizer releases the underlying sockets. Pruning
        proactively on every cache access bounds ``_loop_clients``'s size
        even under many short-lived per-turn loops (see
        ``console_agent_bridge``, which builds a fresh ``asyncio.run()``
        loop per agent turn).
        """
        for stale_loop in [lp for lp in self._loop_clients if lp.is_closed()]:
            del self._loop_clients[stale_loop]

    def _active_http_client(self) -> httpx.AsyncClient:
        """Return an HTTP client bound to the CURRENTLY running event loop.

        The Console reuses one gateway instance for both readiness probes
        (awaited on the app's own event loop) and agent-runtime generation
        calls (bridged from a worker thread via a fresh ``asyncio.run()``
        per turn -- see ``console_agent_bridge._StreamingModelAdapter``).
        httpx/httpcore lazily bind their internal connection-pool
        ``asyncio.Lock``/``Event`` objects to whichever loop first touches
        them; reusing that same client from a second, different loop raises
        ``RuntimeError: ... is bound to a different event loop`` (or, once
        the first loop has since closed, ``RuntimeError: Event loop is
        closed``) on every request -- observed live as every agent send
        failing with "Agent run failed: ... is bound to a different event
        loop." Give each running loop its own owned client via a per-loop
        cache (``_loop_clients``) so no live loop ever has its client
        replaced -- let alone closed -- by another loop's touch; injected
        clients (tests) are trusted to manage their own loop lifecycle and
        are left untouched.

        The whole check-and-maybe-create below is guarded by
        ``_client_lock`` (PR #629 Fix 1(a), preserved from the single-slot
        design): without it, two concurrent callers on different
        loops/threads (the app loop's readiness probe racing the agent
        worker thread's per-turn loop) could race each other while building
        a client for the SAME not-yet-cached loop, or interleave with
        ``aclose()``'s cache teardown. Holding the lock across the prune,
        lookup, and any creation makes the whole operation a single atomic
        step; unlike the old single-slot swap, nothing here ever discards or
        schedules a close of another loop's still-cached client.
        """
        if not self._owns_http_client:
            return self.http_client
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self.http_client
        with self._client_lock:
            self._prune_closed_loops()
            cached = self._loop_clients.get(loop)
            if cached is not None:
                self.http_client, self._client_loop = cached, loop
                return cached
            if (
                not self._client_ever_claimed
                and self.http_client is not None
                and not self.http_client.is_closed
            ):
                # Unclaimed-client escape hatch: the client built in
                # `__init__` hasn't been adopted by ANY loop yet -- claim it
                # for the loop touching it now instead of discarding +
                # scheduling a close of a client nothing has even used,
                # exactly mirroring `GitHubAPIClient.client`'s unknown-loop
                # escape hatch. Gated on `_client_ever_claimed`, not
                # `self._client_loop is None`: `aclose()` resets
                # `_client_loop` to `None` on every teardown, including
                # after a real loop already claimed and released a client,
                # so re-deriving "unclaimed" from it would let a later loop
                # adopt a client a prior loop already bound -- httpx/httpcore
                # lazily bind connection-pool locks to whichever loop first
                # touches them, so reusing that client from a second loop
                # reintroduces the cross-loop binding failure this per-loop
                # cache exists to eliminate. `_client_ever_claimed` is set
                # exactly once, on the first-ever resolution, and is never
                # restored by a subsequent `aclose()`.
                self._loop_clients[loop] = self.http_client
                self._client_loop = loop
                self._client_ever_claimed = True
                return self.http_client
            new_client = self._new_owned_http_client()
            self._loop_clients[loop] = new_client
            self.http_client, self._client_loop = new_client, loop
            self._client_ever_claimed = True
            return new_client

    @staticmethod
    def _schedule_stale_client_close(
        client: httpx.AsyncClient, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Best-effort close of a client left behind on another loop.

        Used by ``aclose()`` to hand off a still-cached OTHER loop's client
        for closing on its own loop, never on the caller's. That loop may
        already be closed (a completed per-turn ``asyncio.run()``) or may
        still be running elsewhere (the app's main loop) -- either way this
        must never raise into the caller. The returned ``Future`` is
        retained with a done-callback so a failed close on another loop is
        logged rather than silently swallowed.
        """
        if loop.is_closed():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(client.aclose(), loop)
        except RuntimeError:
            return

        def _log_close_failure(fut: "asyncio.Future") -> None:
            try:
                exc = fut.exception()
            except Exception:
                # Cancelled, or otherwise unable to retrieve the exception --
                # nothing more we can do here.
                return
            if exc is not None:
                logger.opt(exception=exc).warning(
                    "Failed to close a stale Console provider HTTP client on "
                    "its owning loop: {}",
                    exc,
                )

        future.add_done_callback(_log_close_failure)

    async def resolve_llamacpp(
        self, config: LlamaCppProviderConfig
    ) -> ConsoleProviderResolution:
        """Resolve llama.cpp readiness and the effective model.

        Args:
            config: llama.cpp provider configuration and sampling settings.

        Returns:
            Provider resolution indicating whether llama.cpp can be used.
        """
        model = config.explicit_model or config.configured_model
        base_url = normalize_llamacpp_base_url(config.base_url)
        if not validate_url(base_url):
            return ConsoleProviderResolution(
                provider="llama_cpp",
                base_url=config.base_url,
                model=model,
                ready=False,
                visible_copy=INVALID_LLAMACPP_BASE_URL_COPY,
                readiness_key="llama_cpp",
                execution_key="llama_cpp",
                **self._resolution_settings(config),
            )

        if model is not None:
            if await self._is_reachable(base_url, api_key=config.api_key):
                return ConsoleProviderResolution(
                    provider="llama_cpp",
                    base_url=base_url,
                    model=model,
                    ready=True,
                    readiness_key="llama_cpp",
                    execution_key="llama_cpp",
                    **self._resolution_settings(config),
                )
            return ConsoleProviderResolution(
                provider="llama_cpp",
                base_url=base_url,
                model=model,
                ready=False,
                visible_copy=self._unreachable_copy(base_url),
                readiness_key="llama_cpp",
                execution_key="llama_cpp",
                **self._resolution_settings(config),
            )

        try:
            response = await self._active_http_client().get(
                f"{base_url.rstrip('/')}/v1/models",
                headers=self._authorization_headers(config.api_key),
                timeout=PROBE_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError:
            return ConsoleProviderResolution(
                provider="llama_cpp",
                base_url=base_url,
                model=None,
                ready=False,
                visible_copy=self._unreachable_copy(base_url),
                readiness_key="llama_cpp",
                execution_key="llama_cpp",
                **self._resolution_settings(config),
            )

        model = self._first_model_id(response)
        if model is None:
            return ConsoleProviderResolution(
                provider="llama_cpp",
                base_url=base_url,
                model=None,
                ready=False,
                visible_copy="Provider blocked: select or configure a llama.cpp model.",
                readiness_key="llama_cpp",
                execution_key="llama_cpp",
                **self._resolution_settings(config),
            )
        return ConsoleProviderResolution(
            provider="llama_cpp",
            base_url=base_url,
            model=model,
            ready=True,
            readiness_key="llama_cpp",
            execution_key="llama_cpp",
            **self._resolution_settings(config),
        )

    async def resolve_for_send(
        self, selection: ConsoleProviderSelection
    ) -> ConsoleProviderResolution:
        """Resolve the provider selected by Console before sending.

        Args:
            selection: Current Console provider, model, endpoint, and sampling
                settings.

        Returns:
            Provider resolution used to either send or render recovery copy.
        """
        if not selection.provider.strip():
            return self._blocked_resolution(
                selection,
                provider=selection.provider,
                visible_copy="Select a provider and model before sending.",
            )

        identity = resolve_console_provider_identity(selection.provider)
        if identity.uses_direct_llama_path:
            app_config = self._config_provider() or {}
            readiness = get_provider_readiness(
                identity.readiness_key,
                app_config,
                environ=self._environ,
            )
            if not readiness.ready:
                return self._blocked_resolution(
                    selection,
                    provider=identity.execution_key,
                    visible_copy=readiness.user_message,
                    readiness_key=identity.readiness_key,
                    execution_key=identity.execution_key,
                    api_key_source=readiness.api_key_source,
                )
            resolved = await self.resolve_llamacpp(
                LlamaCppProviderConfig(
                    base_url=selection.base_url or DEFAULT_LLAMACPP_BASE_URL,
                    explicit_model=selection.explicit_model,
                    configured_model=selection.configured_model,
                    api_key=readiness.api_key,
                    api_key_source=readiness.api_key_source,
                    temperature=selection.temperature,
                    top_p=selection.top_p,
                    min_p=selection.min_p,
                    top_k=selection.top_k,
                    max_tokens=selection.max_tokens,
                    seed=selection.seed,
                    presence_penalty=selection.presence_penalty,
                    frequency_penalty=selection.frequency_penalty,
                    reasoning_effort=selection.reasoning_effort,
                    reasoning_summary=selection.reasoning_summary,
                    verbosity=selection.verbosity,
                    thinking_effort=selection.thinking_effort,
                    thinking_budget_tokens=selection.thinking_budget_tokens,
                    streaming=selection.streaming,
                )
            )
            return replace(
                resolved,
                provider=identity.execution_key,
                readiness_key=identity.readiness_key,
                execution_key=identity.execution_key,
            )

        if not identity.is_supported:
            return self._blocked_resolution(
                selection,
                provider=selection.provider,
                visible_copy=(
                    f"Provider blocked: '{selection.provider}' is not available in Console yet. "
                    "Choose a supported provider."
                ),
                readiness_key=identity.readiness_key,
                execution_key=identity.execution_key,
            )

        app_config = self._config_provider() or {}
        try:
            provider_settings = _provider_settings(app_config, identity.readiness_key)
        except ProviderSettingsError:
            return self._blocked_resolution(
                selection,
                provider=selection.provider,
                visible_copy=(
                    "QwenCloud blocked: provider settings must be a configuration "
                    "table under api_settings.qwencloud."
                ),
                readiness_key=identity.readiness_key,
                execution_key=identity.execution_key,
            )
        model = _first_string(
            selection.explicit_model,
            selection.configured_model,
            provider_settings.get("model"),
            provider_settings.get("api_model"),
            provider_settings.get("default_model"),
        )
        if model is None:
            return self._blocked_resolution(
                selection,
                provider=selection.provider,
                visible_copy="Select a model before sending.",
                readiness_key=identity.readiness_key,
                execution_key=identity.execution_key,
            )

        api_mode: str | None = None
        qwencloud_configured_base_url: str | None = None
        effective_base_url: str | None
        if identity.execution_key == "qwencloud":
            try:
                api_mode = normalize_qwencloud_api_mode(
                    None,
                    provider_settings=provider_settings,
                )
            except ChatConfigurationError:
                return self._blocked_resolution(
                    selection,
                    provider=selection.provider,
                    model=model,
                    visible_copy=(
                        "QwenCloud blocked: invalid API mode setting. Choose "
                        "'responses' or 'chat_completions' in Settings."
                    ),
                    readiness_key=identity.readiness_key,
                    execution_key=identity.execution_key,
                )

            configured_base_url: str | None = None
            if "api_base_url" in provider_settings:
                raw_configured_base_url = provider_settings["api_base_url"]
                if not isinstance(raw_configured_base_url, str) or not (
                    raw_configured_base_url.strip()
                ):
                    return self._blocked_resolution(
                        selection,
                        provider=selection.provider,
                        model=model,
                        visible_copy=(
                            "QwenCloud blocked: invalid API base URL setting. Enter "
                            "an absolute HTTP(S) compatible-mode base URL in Settings."
                        ),
                        readiness_key=identity.readiness_key,
                        execution_key=identity.execution_key,
                    )
                configured_base_url = raw_configured_base_url

            try:
                qwencloud_configured_base_url = normalize_qwencloud_base_url(
                    configured_base_url
                )
                selected_base_url = selection.base_url
                if selected_base_url is None or (
                    isinstance(selected_base_url, str) and not selected_base_url.strip()
                ):
                    effective_base_url = qwencloud_configured_base_url
                else:
                    effective_base_url = normalize_qwencloud_base_url(selected_base_url)
            except ChatConfigurationError:
                return self._blocked_resolution(
                    selection,
                    provider=selection.provider,
                    model=model,
                    visible_copy=(
                        "QwenCloud blocked: invalid API base URL setting. Enter an "
                        "absolute HTTP(S) compatible-mode base URL in Settings."
                    ),
                    readiness_key=identity.readiness_key,
                    execution_key=identity.execution_key,
                )

        elif identity.execution_key == "deepseek":
            try:
                api_mode = _normalize_deepseek_api_mode(provider_settings)
            except ChatConfigurationError:
                return self._blocked_resolution(
                    selection,
                    provider=selection.provider,
                    model=model,
                    visible_copy=(
                        "DeepSeek blocked: invalid API mode setting. Choose "
                        "'responses' or 'chat_completions' in Settings."
                    ),
                    readiness_key=identity.readiness_key,
                    execution_key=identity.execution_key,
                )

            effective_base_url = effective_provider_endpoint(
                identity.readiness_key,
                selection.base_url,
                provider_settings,
            )
        else:
            effective_base_url = effective_provider_endpoint(
                identity.readiness_key,
                selection.base_url,
                provider_settings,
            )

        if identity.execution_key == "qwencloud":
            endpoint_differs = (
                qwencloud_configured_base_url is None
                or effective_base_url != qwencloud_configured_base_url
            )
        else:
            endpoint_differs = generic_endpoint_differs(
                selection.base_url, provider_settings
            )

        if (
            provider_uses_endpoint(identity.readiness_key, provider_settings)
            and endpoint_differs
        ):
            return self._blocked_resolution(
                selection,
                provider=selection.provider,
                model=model,
                visible_copy=unsaved_endpoint_copy(
                    selection.base_url, provider_settings
                ),
                readiness_key=identity.readiness_key,
                execution_key=identity.execution_key,
            )

        readiness = get_provider_readiness(
            identity.readiness_key, app_config, environ=self._environ
        )
        if not readiness.ready:
            return self._blocked_resolution(
                selection,
                provider=selection.provider,
                model=model,
                visible_copy=readiness.user_message,
                readiness_key=identity.readiness_key,
                execution_key=identity.execution_key,
                api_key_source=readiness.api_key_source,
            )

        # Console sends are multi-turn by construction, so they -- and ONLY
        # they -- opt into the Anthropic per-turn cache_control breakpoint.
        # A one-shot caller of `chat_with_anthropic` (summarization, evals,
        # websearch) would pay the 1.25x cache-write premium on its whole
        # prefix and never read it back, so the flag is stamped here rather
        # than defaulted on inside the provider.
        #
        # NOTE: this flag does not gate the kill-switch end-to-end by
        # itself -- `chat_with_anthropic` ANDs it with
        # `_anthropic_caching_enabled()`, which reads `[caching]
        # anthropic_enabled` directly via `get_cli_setting()` and already
        # disables every cache_control breakpoint (system, tool, AND this
        # per-turn one) when the switch is off, regardless of what this
        # resolution carries. This value only needs to be *truthful* for
        # whatever inspects `ConsoleProviderResolution.prompt_caching`
        # (introspection/tests/telemetry), which is why the config-shape
        # bug below mattered even though sends were never at risk.
        prompt_caching: bool | None = None
        if identity.execution_key == "anthropic":
            prompt_caching = bool(
                _caching_config_value(app_config).get("anthropic_enabled", True)
            )
        continuation_protocol = (
            "chat_completions"
            if identity.execution_key in {"moonshot", "zai"}
            else api_mode
            if identity.execution_key == "deepseek"
            else None
        )
        request_timeout: float | None = None
        request_retries: int | None = None
        request_retry_delay: float | None = None
        if identity.execution_key in {"moonshot", "zai"}:
            try:
                (
                    request_timeout,
                    request_retries,
                    request_retry_delay,
                ) = _hosted_transport_policy(
                    provider_settings,
                    provider=identity.execution_key,
                )
            except ChatConfigurationError:
                return self._blocked_resolution(
                    selection,
                    provider=selection.provider,
                    model=model,
                    visible_copy=(
                        f"{selection.provider} blocked: invalid timeout or retry "
                        "settings. Correct the provider transport policy in Settings."
                    ),
                    readiness_key=identity.readiness_key,
                    execution_key=identity.execution_key,
                )

        return ConsoleProviderResolution(
            provider=selection.provider,
            base_url=effective_base_url or "",
            model=model,
            ready=True,
            readiness_key=identity.readiness_key,
            execution_key=identity.execution_key,
            api_key=readiness.api_key,
            api_key_source=readiness.api_key_source,
            prompt_caching=prompt_caching,
            api_mode=api_mode,
            continuation_protocol=continuation_protocol,
            request_timeout=request_timeout,
            request_retries=request_retries,
            request_retry_delay=request_retry_delay,
            temperature=selection.temperature,
            top_p=selection.top_p,
            min_p=selection.min_p,
            top_k=selection.top_k,
            max_tokens=selection.max_tokens,
            seed=selection.seed,
            presence_penalty=selection.presence_penalty,
            frequency_penalty=selection.frequency_penalty,
            reasoning_effort=selection.reasoning_effort,
            reasoning_summary=selection.reasoning_summary,
            verbosity=selection.verbosity,
            thinking_effort=selection.thinking_effort,
            thinking_budget_tokens=selection.thinking_budget_tokens,
            streaming=selection.streaming,
        )

    async def stream_llamacpp_chat(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[Mapping[str, Any]],
        temperature: float | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        thinking_budget_tokens: int | None = None,
        api_key: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream OpenAI-compatible chat completion chunks from llama.cpp.

        Args:
            base_url: llama.cpp server endpoint.
            model: Model identifier to send.
            messages: OpenAI-compatible chat messages.
            temperature: Optional sampling temperature.
            top_p: Optional nucleus sampling value.
            min_p: Optional min-p sampling value.
            top_k: Optional top-k sampling value.
            max_tokens: Optional response token limit.
            reasoning_effort: Optional thinking level forwarded as
                ``chat_template_kwargs.reasoning_effort``.
            thinking_budget_tokens: Optional thinking token budget sent as
                the top-level ``reasoning_budget_tokens`` field.

        Yields:
            Assistant-visible content chunks.
        """
        normalized_base_url = normalize_llamacpp_base_url(base_url)
        if not validate_url(normalized_base_url):
            raise ValueError("invalid llama.cpp base URL")

        payload = build_llamacpp_chat_payload(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            thinking_budget_tokens=thinking_budget_tokens,
        )
        think_filter = StartAnchoredThinkFilter()
        emitted_content = False
        received_content = False
        stream_error: httpx.HTTPError | None = None
        try:
            async with self._active_http_client().stream(
                "POST",
                f"{normalized_base_url.rstrip('/')}/v1/chat/completions",
                json=payload,
                headers=self._authorization_headers(api_key),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    chunk = self._content_from_sse_line(line)
                    if chunk:
                        received_content = True
                        visible = think_filter.feed(chunk)
                        if visible:
                            emitted_content = True
                            yield visible
        except httpx.HTTPError as exc:
            if emitted_content:
                raise
            stream_error = exc

        if emitted_content:
            # flush() contractually returns "" (unterminated start-anchored
            # think tails are dropped), so there is no tail to yield.
            return
        if received_content:
            # Think-only reply: the filter removed every chunk, so a
            # non-streaming retry would return the same text — skip it and
            # surface any stream error that followed the content instead.
            if stream_error is not None:
                raise stream_error
            return

        fallback = await self.complete_llamacpp_chat(
            base_url=normalized_base_url,
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            thinking_budget_tokens=thinking_budget_tokens,
            api_key=api_key,
        )
        if fallback:
            yield fallback
            return
        if stream_error is not None:
            raise stream_error

    async def complete_llamacpp_chat(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[Mapping[str, Any]],
        temperature: float | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        reasoning_effort: str | None = None,
        thinking_budget_tokens: int | None = None,
        strict_response: bool = False,
        api_key: str | None = None,
    ) -> str:
        """Request a non-streaming OpenAI-compatible chat completion.

        Args:
            base_url: llama.cpp server endpoint.
            model: Model identifier to send.
            messages: OpenAI-compatible chat messages.
            temperature: Optional sampling temperature.
            top_p: Optional nucleus sampling value.
            min_p: Optional min-p sampling value.
            top_k: Optional top-k sampling value.
            max_tokens: Optional response token limit.
            seed: Optional deterministic generation seed.
            presence_penalty: Optional presence penalty value.
            frequency_penalty: Optional frequency penalty value.
            reasoning_effort: Optional thinking level forwarded as
                ``chat_template_kwargs.reasoning_effort``.
            thinking_budget_tokens: Optional thinking token budget sent as
                the top-level ``reasoning_budget_tokens`` field.
            strict_response: Raise when the provider response has no supported
                assistant-content shape instead of treating it as empty.

        Returns:
            Assistant-visible completion text.
        """
        normalized_base_url = normalize_llamacpp_base_url(base_url)
        if not validate_url(normalized_base_url):
            raise ValueError("invalid llama.cpp base URL")

        request_url = f"{normalized_base_url.rstrip('/')}/v1/chat/completions"
        payload = build_llamacpp_chat_payload(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            reasoning_effort=reasoning_effort,
            thinking_budget_tokens=thinking_budget_tokens,
        )
        client = self._active_http_client()
        response = (
            await self._post_without_high_level_http_log(
                client,
                request_url,
                json_payload=payload,
                headers=self._authorization_headers(api_key),
            )
            if is_sensitive_llm_request()
            else await client.post(
                request_url,
                json=payload,
                headers=self._authorization_headers(api_key),
            )
        )
        response.raise_for_status()
        content = self._content_from_completion_response(response)
        if content is None and strict_response:
            raise ChatProviderError(
                "Provider returned an unsupported auxiliary response.",
                provider="llama_cpp",
            )
        think_filter = StartAnchoredThinkFilter()
        return think_filter.feed(content or "") + think_filter.flush()

    @staticmethod
    async def _post_without_high_level_http_log(
        client: httpx.AsyncClient,
        url: str,
        *,
        json_payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """POST through this client's transport without HTTPX's URL-bearing INFO log."""

        request = client.build_request("POST", url, json=json_payload, headers=headers)
        transport = client._transport_for_url(request.url)
        response = await transport.handle_async_request(request)
        response.request = request
        try:
            await response.aread()
        finally:
            await response.aclose()
        return response

    async def complete_auxiliary(
        self,
        request: AuxiliaryCompletionRequest,
    ) -> AuxiliaryCompletionResult:
        """Run exactly one sensitive, non-streaming completion.

        The captured resolution is the sole provider authority. This path
        deliberately bypasses normal Console history, tools, streaming
        normalization, fallback copy, and persistence.
        """

        if not isinstance(request, AuxiliaryCompletionRequest):
            raise TypeError("request must be an AuxiliaryCompletionRequest")
        resolution = replace(
            request.resolution,
            streaming=False,
            max_tokens=request.max_output_tokens,
        )
        provider = resolution.provider
        model = cast(str, resolution.model)
        messages = cast(
            list[Mapping[str, Any]], _thaw_auxiliary_value(request.messages)
        )
        response: Any = _UNSUPPORTED_RESPONSE
        try:
            with sensitive_llm_request():
                if resolution.provider in {"llama_cpp", "local_llamacpp"}:
                    # Thinking controls follow ADR-066: level via
                    # chat_template_kwargs, budget via top-level
                    # reasoning_budget_tokens. Auxiliary requests inherit session
                    # thinking settings (documented parity with cloud
                    # providers).
                    text = await self.complete_llamacpp_chat(
                        base_url=resolution.base_url,
                        model=model,
                        messages=messages,
                        temperature=resolution.temperature,
                        top_p=resolution.top_p,
                        min_p=resolution.min_p,
                        top_k=resolution.top_k,
                        max_tokens=request.max_output_tokens,
                        seed=resolution.seed,
                        presence_penalty=resolution.presence_penalty,
                        frequency_penalty=resolution.frequency_penalty,
                        reasoning_effort=resolution.reasoning_effort,
                        thinking_budget_tokens=resolution.thinking_budget_tokens,
                        strict_response=True,
                        api_key=resolution.api_key,
                    )
                else:
                    kwargs = self._auxiliary_chat_api_kwargs(request, resolution)
                    context = copy_context()
                    response = await asyncio.to_thread(
                        context.run,
                        self._complete_sensitive_sync,
                        kwargs,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            status_code = getattr(exc, "status_code", 502)
            raise ChatProviderError(
                safe_provider_error_copy(provider, exc),
                provider=provider,
                status_code=status_code if isinstance(status_code, int) else 502,
            ) from None

        usage: ProviderUsage | None = None
        if response is not _UNSUPPORTED_RESPONSE:
            text = self._auxiliary_response_text(response)
            if isinstance(response, Mapping):
                usage = ProviderUsage.from_provider_payload(
                    response.get("usage"),
                    provider=provider,
                    model=model,
                )

        if not isinstance(text, str):
            raise ChatProviderError(
                "Provider returned an unsupported auxiliary response.",
                provider=provider,
            )
        return AuxiliaryCompletionResult(
            provider=provider,
            model=model,
            text=text,
            usage=usage,
        )

    def _complete_sensitive_sync(self, kwargs: Mapping[str, Any]) -> Any:
        """Invoke the final synchronous adapter under the sensitive policy."""

        with sensitive_llm_request():
            return self._chat_api_call(**dict(kwargs))

    @staticmethod
    def _auxiliary_response_text(response: Any) -> str:
        """Extract exact assistant text from supported non-streaming shapes."""

        if isinstance(response, str):
            return response
        if not isinstance(response, Mapping):
            raise ChatProviderError(
                "Provider returned an unsupported auxiliary response."
            )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ChatProviderError(
                "Provider returned an unsupported auxiliary response."
            )
        first = choices[0]
        if not isinstance(first, Mapping):
            raise ChatProviderError(
                "Provider returned an unsupported auxiliary response."
            )
        message = first.get("message")
        if not isinstance(message, Mapping) or not isinstance(
            message.get("content"), str
        ):
            raise ChatProviderError(
                "Provider returned an unsupported auxiliary response."
            )
        return cast(str, message["content"])

    @staticmethod
    def _auxiliary_chat_api_kwargs(
        request: AuxiliaryCompletionRequest,
        resolution: ConsoleProviderResolution,
    ) -> dict[str, Any]:
        """Build the isolated, tool-free kwargs for one auxiliary adapter."""

        payload = cast(list[dict[str, Any]], _thaw_auxiliary_value(request.messages))
        system_parts: list[str] = []
        while payload and payload[0].get("role") == "system":
            content = cast(str, payload.pop(0).get("content"))
            if content:
                system_parts.append(content)
        kwargs: dict[str, Any] = {
            "api_endpoint": resolution.execution_key,
            "api_base_url": resolution.base_url or None,
            "system_message": "\n\n".join(system_parts) or None,
            "messages_payload": payload,
            "api_key": resolution.api_key,
            "model": resolution.model,
            "streaming": False,
            "temp": resolution.temperature,
            "topp": resolution.top_p,
            "maxp": resolution.top_p,
            "topk": resolution.top_k,
            "minp": resolution.min_p,
            "max_tokens": request.max_output_tokens,
            "seed": resolution.seed,
            "presence_penalty": resolution.presence_penalty,
            "frequency_penalty": resolution.frequency_penalty,
            "reasoning_effort": resolution.reasoning_effort,
            "reasoning_summary": resolution.reasoning_summary,
            "verbosity": resolution.verbosity,
            "thinking_effort": resolution.thinking_effort,
            "thinking_budget_tokens": resolution.thinking_budget_tokens,
            "response_format": (
                _thaw_auxiliary_value(request.response_format)
                if request.response_format is not None
                else None
            ),
        }
        if resolution.execution_key == "qwencloud":
            kwargs["api_mode"] = resolution.api_mode
            kwargs["api_base_url"] = resolution.base_url or None
        elif resolution.execution_key in _CUSTOM_CREDENTIAL_DECISION_PROVIDERS:
            kwargs["api_key_resolved"] = True
        return {key: value for key, value in kwargs.items() if value is not None}

    async def stream_chat(
        self,
        resolution: ConsoleProviderResolution,
        messages: list[Mapping[str, Any]]
        | PreparedConsoleRequest
        | PreparedProviderRequest,
        tools: list | None = None,
        signals: _ProviderStreamSignals | None = None,
    ) -> AsyncIterator[str | ProviderToolCalls]:
        """Dispatch streaming for a resolved Console provider.

        Args:
            resolution: Provider resolution produced by ``resolve_for_send``.
            messages: Raw OpenAI-compatible messages, a semantic request, or
                the already serialized provider request. Raw/semantic inputs
                are prepared exactly once before accounting and dispatch.
            tools: Optional OpenAI-shape tool definitions. When omitted,
                behavior is byte-identical to a plain Console send. When
                provided, yields str chunks as before; if the provider
                returned native tool-calls, the final item is a
                ``ProviderToolCalls`` instead of a str.
            signals: Optional out-of-band stream provenance signals.

        Yields:
            Assistant-visible content chunks, and -- only when ``tools`` was
            passed and the provider returned native tool-calls -- a final
            ``ProviderToolCalls``.
        """
        # ONE invocation of this method == ONE provider call. A turn (agent
        # runs especially) makes N of them through the SAME signals object,
        # so the in-flight usage payload is closed out here, at the only
        # seam that knows where a call ends -- never in the consumer, which
        # cannot see the boundary at all.
        call_signals = (
            signals
            if isinstance(signals, ConsoleProviderCallSignals)
            else signals.new_usage_call()
            if signals is not None
            else None
        )
        try:
            if not resolution.ready or not resolution.model:
                return
            prepared = (
                messages
                if isinstance(messages, PreparedProviderRequest)
                else self.prepare_chat_request(resolution, messages, tools=tools)
            )
            if isinstance(messages, PreparedProviderRequest) and tools is not None:
                raise ValueError("tools are already owned by PreparedProviderRequest")
            if prepared.provider and prepared.provider != resolution.provider:
                raise ValueError("Prepared request provider does not match resolution.")
            if prepared.model and prepared.model != resolution.model:
                raise ValueError("Prepared request model does not match resolution.")
            if prepared.known_overflow:
                ceiling = prepared.capacity.effective_input_ceiling_tokens
                raise ChatBadRequestError(
                    "Mandatory Console request material exceeds the effective "
                    f"input ceiling ({prepared.accounting.total_input_tokens} > "
                    f"{ceiling}). Compaction cannot remove this material.",
                    provider=resolution.provider,
                )
            effective_resolution = replace(
                resolution,
                max_tokens=(
                    prepared.capacity.effective_response_tokens
                    if resolution.max_tokens is not None
                    else None
                ),
            )
            if resolution.provider in {"llama_cpp", "local_llamacpp"}:
                wire_messages = [thaw_json(item) for item in prepared.messages]
                if not resolution.streaming:
                    completion = await self.complete_llamacpp_chat(
                        base_url=resolution.base_url,
                        model=resolution.model,
                        messages=wire_messages,
                        temperature=resolution.temperature,
                        top_p=resolution.top_p,
                        min_p=resolution.min_p,
                        top_k=resolution.top_k,
                        max_tokens=effective_resolution.max_tokens,
                        reasoning_effort=resolution.reasoning_effort,
                        thinking_budget_tokens=resolution.thinking_budget_tokens,
                        api_key=resolution.api_key,
                    )
                    if completion:
                        yield completion
                    return
                async for chunk in self.stream_llamacpp_chat(
                    base_url=resolution.base_url,
                    model=resolution.model,
                    messages=wire_messages,
                    temperature=resolution.temperature,
                    top_p=resolution.top_p,
                    min_p=resolution.min_p,
                    top_k=resolution.top_k,
                    max_tokens=effective_resolution.max_tokens,
                    reasoning_effort=resolution.reasoning_effort,
                    thinking_budget_tokens=resolution.thinking_budget_tokens,
                    api_key=resolution.api_key,
                ):
                    yield chunk
                return
            if resolution.execution_key:
                async for chunk in self._stream_generic_chat(
                    effective_resolution, prepared, signals=call_signals
                ):
                    yield chunk
                return
        finally:
            if call_signals is not None:
                call_signals.close_usage_call()

    async def _stream_generic_chat(
        self,
        resolution: ConsoleProviderResolution,
        request: PreparedProviderRequest,
        signals: _ProviderStreamSignals | None = None,
    ) -> AsyncIterator[str | ProviderToolCalls]:
        """Bridge synchronous chat_api_call responses into async Console chunks."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        stop_event = threading.Event()
        response_lock = threading.Lock()
        retained_response: Any = None
        close_requested = False
        response_close_attempted = False

        def retain_response(response: Any) -> bool:
            nonlocal retained_response, response_close_attempted
            close = None
            with response_lock:
                retained_response = response
                iteration_permitted = not close_requested
                if close_requested and not response_close_attempted:
                    response_close_attempted = True
                    close = getattr(retained_response, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
            return iteration_permitted

        def close_response() -> None:
            nonlocal close_requested, response_close_attempted
            with response_lock:
                close_requested = True
                if response_close_attempted or retained_response is None:
                    return
                response_close_attempted = True
                close = getattr(retained_response, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()

        def enqueue(item: _QueueItem) -> None:
            if stop_event.is_set():
                return
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(queue.put_nowait, item)

        def worker() -> None:
            try:
                kwargs = self._chat_api_kwargs_from_prepared(resolution, request)
                response = self._chat_api_call(**kwargs)
                provider_response = response
                accumulator = _ToolCallAccumulator() if request.tools else None
                if accumulator is not None:
                    response = _tee_tool_calls(response, accumulator)
                if not retain_response(response) or stop_event.is_set():
                    return
                emitted_content = False
                # tools= runs: fallback UI copy must never leak into agent
                # history, so it is suppressed at GENERATION (not filtered
                # by string equality — review minor m4: a real answer that
                # happens to equal the copy text now flows through).
                normalized_response = self.normalize_provider_response(
                    response,
                    suppress_fallback_copy=accumulator is not None,
                    signals=signals,
                )
                while not stop_event.is_set():
                    try:
                        text = next(normalized_response)
                    except StopIteration:
                        break
                    if text:
                        emitted_content = True
                    enqueue(_QueueItem.content(text))
                if stop_event.is_set():
                    return
                if accumulator is not None:
                    calls = accumulator.calls()
                    metadata = _provider_turn_metadata(provider_response)
                    if calls or metadata is not None:
                        enqueue(_QueueItem.native_tool_calls(calls, metadata))
                    elif not emitted_content:
                        # PR #648 review Minor 1: the turn produced NEITHER
                        # visible content NOR tool-calls. On the fence path
                        # junk surfaces as diagnostic copy; silently
                        # completing here would make a misbehaving provider's
                        # junk 200-body indistinguishable from a legitimate
                        # empty answer. Surface it as a provider error,
                        # feeding the run's existing honest RUN_ERROR path.
                        no_content_exc = ChatProviderError(
                            "Provider returned no content and no tool calls.",
                            provider=resolution.provider,
                        )
                        enqueue(
                            _QueueItem.error(
                                self._safe_error_copy(
                                    resolution.provider, no_content_exc
                                ),
                                status_code=no_content_exc.status_code,
                            )
                        )
            except BaseException as exc:
                raw_status = getattr(exc, "status_code", None)
                status_code = raw_status if isinstance(raw_status, int) else None
                error_copy = _provider_error_copy_with_model_recovery(
                    self._safe_error_copy(resolution.provider, exc),
                    model=resolution.model,
                    status_code=status_code,
                )
                enqueue(
                    _QueueItem.error(
                        error_copy,
                        status_code=status_code,
                    )
                )
            finally:
                close_response()
                enqueue(_QueueItem.done())

        worker_task = asyncio.create_task(asyncio.to_thread(worker))
        try:
            while True:
                item = await queue.get()
                if item.kind == "done":
                    break
                if item.kind == "error":
                    # F5: carry the real status the worker captured -- never
                    # re-derive it by parsing item.text back out (that text
                    # is redacted prose, not a machine-readable status).
                    raise ChatProviderError(
                        item.text
                        or safe_provider_error_copy(
                            resolution.provider, ChatProviderError()
                        ),
                        provider=resolution.provider,
                        status_code=item.status_code
                        if isinstance(item.status_code, int)
                        else 502,
                    )
                if item.kind == "tool_calls":
                    yield cast(ProviderToolCalls, item.payload)
                    continue
                if item.text:
                    yield item.text
        finally:
            stop_event.set()
            close_response()
            if not worker_task.done():
                worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(worker_task), timeout=0)

    @staticmethod
    def normalize_provider_response(
        response: Any,
        suppress_fallback_copy: bool = False,
        signals: _ProviderStreamSignals | None = None,
    ) -> Iterator[str]:
        """Yield safe assistant-visible chunks from generic provider output.

        Args:
            response: Raw return value from ``chat_api_call``.
            suppress_fallback_copy: When True (tools= agent runs), the
                NO_PROVIDER_CONTENT / UNSUPPORTED fallback UI copy is never
                GENERATED instead of being string-filtered downstream — so a
                real model answer that happens to equal the copy text flows
                through untouched (review minor m4, PR #648 line).
            signals: Optional out-of-band stream provenance signals.

        Yields:
            Assistant-visible text chunks (and, unless suppressed,
            normalized fallback copy).
        """
        content = _content_from_provider_item(response, signals=signals)
        if isinstance(content, str):
            if content:
                yield content
            elif not suppress_fallback_copy:
                if signals is not None:
                    signals.mark_synthetic_fallback()
                yield NO_PROVIDER_CONTENT_COPY
            return
        if content is _UNSUPPORTED_RESPONSE:
            if _is_iterable_response(response):
                emitted = False
                for item in response:
                    item_content = _content_from_provider_item(item, signals=signals)
                    if isinstance(item_content, str):
                        if item_content:
                            emitted = True
                            yield item_content
                        continue
                    if item_content is _EMPTY_RESPONSE:
                        continue
                    emitted = True
                    if not suppress_fallback_copy:
                        if signals is not None:
                            signals.mark_synthetic_fallback()
                        yield UNSUPPORTED_PROVIDER_RESPONSE_COPY
                if not emitted and not suppress_fallback_copy:
                    if signals is not None:
                        signals.mark_synthetic_fallback()
                    yield NO_PROVIDER_CONTENT_COPY
                return
            if not suppress_fallback_copy:
                if signals is not None:
                    signals.mark_synthetic_fallback()
                yield UNSUPPORTED_PROVIDER_RESPONSE_COPY
            return
        if not suppress_fallback_copy:
            if signals is not None:
                signals.mark_synthetic_fallback()
            yield NO_PROVIDER_CONTENT_COPY

    def _chat_api_call(self, **kwargs: Any) -> Any:
        if self._chat_api_call_fn is None:
            from tldw_chatbook.Chat.Chat_Functions import chat_api_call

            return chat_api_call(**kwargs)
        return self._chat_api_call_fn(**kwargs)

    @staticmethod
    def _chat_api_kwargs_from_prepared(
        resolution: ConsoleProviderResolution,
        request: PreparedProviderRequest,
    ) -> dict[str, Any]:
        """Build adapter kwargs without re-serializing the prepared payload."""

        kwargs = {
            "api_endpoint": resolution.execution_key,
            "system_message": request.system_message,
            "messages_payload": [thaw_json(item) for item in request.messages_payload],
            "api_key": resolution.api_key,
            "model": resolution.model,
            "streaming": resolution.streaming,
            "temp": resolution.temperature,
            "topp": resolution.top_p,
            "maxp": resolution.top_p,
            "topk": resolution.top_k,
            "minp": resolution.min_p,
            "max_tokens": resolution.max_tokens,
            "seed": resolution.seed,
            "presence_penalty": resolution.presence_penalty,
            "frequency_penalty": resolution.frequency_penalty,
            "reasoning_effort": resolution.reasoning_effort,
            "reasoning_summary": resolution.reasoning_summary,
            "verbosity": resolution.verbosity,
            "thinking_effort": resolution.thinking_effort,
            "thinking_budget_tokens": resolution.thinking_budget_tokens,
            "tools": thaw_json(request.tools) if request.tools else None,
            "response_format": (
                thaw_json(request.response_format)
                if request.response_format is not None
                else None
            ),
            "prompt_caching": resolution.prompt_caching,
        }
        if resolution.execution_key == "qwencloud":
            kwargs["api_mode"] = resolution.api_mode
            kwargs["api_base_url"] = resolution.base_url or None
        elif resolution.execution_key in {"moonshot", "zai"}:
            kwargs["api_base_url"] = resolution.base_url or None
            kwargs["request_timeout"] = resolution.request_timeout
            kwargs["request_retries"] = resolution.request_retries
            kwargs["request_retry_delay"] = resolution.request_retry_delay
            if request.continuation_groups:
                kwargs["provider_continuations"] = [
                    group.checkpoint for group in request.continuation_groups
                ]
        elif resolution.execution_key in {
            "anthropic",
            "custom-openai-api",
            "custom-openai-api-2",
            "mistral",
            "mistralai",
        }:
            kwargs["api_base_url"] = resolution.base_url or None
            if resolution.execution_key in _CUSTOM_CREDENTIAL_DECISION_PROVIDERS:
                kwargs["api_key_resolved"] = True
        elif (
            resolution.execution_key == "openai" and request.response_format is not None
        ):
            # Evaluator-only structured-output requests pin the endpoint that
            # was resolved and capability-checked. Ordinary Console sends have
            # no response_format and retain their existing adapter behavior.
            kwargs["api_base_url"] = resolution.base_url or None
        return {key: value for key, value in kwargs.items() if value is not None}

    @staticmethod
    def _chat_api_kwargs(
        resolution: ConsoleProviderResolution,
        messages: list[Mapping[str, Any]],
        tools: list | None = None,
    ) -> dict[str, Any]:
        # Extract the contiguous LEADING system rows into chat_api_call's
        # `system_message` parameter (PR #1112 Qodo finding 3): Anthropic and
        # Gemini adapters accept system content only via their dedicated
        # parameter and do not honor `role="system"` rows in the message
        # array, so a payload-only system prompt (and the task-1531 folded
        # greeting riding it) would never reach those providers. The
        # OpenAI-compatible adapters prepend `system_message` as a system row
        # themselves when the payload has none, so the extraction is
        # provider-neutral. Mid-array system rows never occur on this path
        # (`_provider_message_payloads` emits user/assistant only).
        #
        # The join deliberately normalizes: each row is `.strip()`ed and the
        # rows are joined with "\n\n". The result is therefore NOT byte-verbatim
        # with respect to the source rows -- but it IS a pure function of them,
        # so the same rows always produce the same bytes. That determinism is
        # what Anthropic prefix caching needs (a stable system block across
        # consecutive turns); see the cache-stability test in
        # Tests/Chat/test_console_provider_gateway.py.
        payload = list(messages)
        system_parts: list[str] = []
        while payload and payload[0].get("role") == "system":
            content = str(payload[0].get("content") or "").strip()
            if content:
                system_parts.append(content)
            payload = payload[1:]
        kwargs = {
            "api_endpoint": resolution.execution_key,
            "system_message": "\n\n".join(system_parts) or None,
            "messages_payload": payload,
            "api_key": resolution.api_key,
            "model": resolution.model,
            "streaming": resolution.streaming,
            "temp": resolution.temperature,
            "topp": resolution.top_p,
            "maxp": resolution.top_p,
            "topk": resolution.top_k,
            "minp": resolution.min_p,
            "max_tokens": resolution.max_tokens,
            "seed": resolution.seed,
            "presence_penalty": resolution.presence_penalty,
            "frequency_penalty": resolution.frequency_penalty,
            "reasoning_effort": resolution.reasoning_effort,
            "reasoning_summary": resolution.reasoning_summary,
            "verbosity": resolution.verbosity,
            "thinking_effort": resolution.thinking_effort,
            "thinking_budget_tokens": resolution.thinking_budget_tokens,
            "tools": tools,
            # None for every non-Anthropic resolution, so the strip below
            # removes the key entirely and other providers' kwargs are byte
            # for byte what they were before prompt caching existed.
            "prompt_caching": resolution.prompt_caching,
        }
        if resolution.execution_key == "qwencloud":
            kwargs["api_mode"] = resolution.api_mode
            kwargs["api_base_url"] = resolution.base_url or None
        elif resolution.execution_key in {
            "anthropic",
            "custom-openai-api",
            "custom-openai-api-2",
            "mistral",
            "mistralai",
        }:
            # These adapters otherwise consult process-global config after
            # Console has resolved a provider-scoped endpoint and credential.
            # Pinning the resolved base keeps that pair intact, including the
            # custom aliases and distinct mistral config owners.
            kwargs["api_base_url"] = resolution.base_url or None
            if resolution.execution_key in _CUSTOM_CREDENTIAL_DECISION_PROVIDERS:
                kwargs["api_key_resolved"] = True
        return {key: value for key, value in kwargs.items() if value is not None}

    @staticmethod
    def _raise_for_sse_error(line: str) -> None:
        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return
        if isinstance(payload, Mapping) and "error" in payload:
            raise RuntimeError("Provider stream error.")

    @staticmethod
    def _resolution_settings(config: LlamaCppProviderConfig) -> dict[str, Any]:
        return {
            "api_key": config.api_key,
            "api_key_source": config.api_key_source,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "min_p": config.min_p,
            "top_k": config.top_k,
            "max_tokens": config.max_tokens,
            "seed": config.seed,
            "presence_penalty": config.presence_penalty,
            "frequency_penalty": config.frequency_penalty,
            "reasoning_effort": config.reasoning_effort,
            "reasoning_summary": config.reasoning_summary,
            "verbosity": config.verbosity,
            "thinking_effort": config.thinking_effort,
            "thinking_budget_tokens": config.thinking_budget_tokens,
            "streaming": config.streaming,
        }

    @staticmethod
    def _authorization_headers(api_key: str | None) -> dict[str, str] | None:
        return {"Authorization": f"Bearer {api_key}"} if api_key else None

    async def _is_reachable(self, base_url: str, *, api_key: str | None = None) -> bool:
        try:
            await self._active_http_client().get(
                f"{base_url.rstrip('/')}/health",
                headers=self._authorization_headers(api_key),
                timeout=PROBE_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError:
            return False
        return True

    @staticmethod
    def _first_model_id(response: httpx.Response) -> str | None:
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        if not isinstance(data, list):
            return None
        for item in data:
            if (
                isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and item["id"]
            ):
                return item["id"]
        return None

    @staticmethod
    def _content_from_sse_line(line: str) -> str | None:
        if not line.startswith("data:"):
            return None
        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            return None
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        delta = first.get("delta")
        if not isinstance(delta, dict):
            return None
        content = delta.get("content")
        return content if isinstance(content, str) else None

    @staticmethod
    def _content_from_completion_response(response: httpx.Response) -> str | None:
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        text = first.get("text")
        return text if isinstance(text, str) else None

    @staticmethod
    def _unreachable_copy(base_url: str) -> str:
        return (
            f"Provider blocked: llama.cpp server is not reachable at {base_url}. "
            "Start llama.cpp or update Console provider settings."
        )

    @staticmethod
    def _blocked_resolution(
        selection: ConsoleProviderSelection,
        *,
        provider: str,
        visible_copy: str,
        model: str | None = None,
        readiness_key: str = "",
        execution_key: str = "",
        api_key_source: str | None = None,
    ) -> ConsoleProviderResolution:
        return ConsoleProviderResolution(
            provider=provider,
            base_url=selection.base_url or "",
            model=model
            if model is not None
            else selection.explicit_model or selection.configured_model,
            ready=False,
            visible_copy=visible_copy,
            readiness_key=readiness_key,
            execution_key=execution_key,
            api_key_source=api_key_source,
            temperature=selection.temperature,
            top_p=selection.top_p,
            min_p=selection.min_p,
            top_k=selection.top_k,
            max_tokens=selection.max_tokens,
            seed=selection.seed,
            presence_penalty=selection.presence_penalty,
            frequency_penalty=selection.frequency_penalty,
            reasoning_effort=selection.reasoning_effort,
            reasoning_summary=selection.reasoning_summary,
            verbosity=selection.verbosity,
            thinking_effort=selection.thinking_effort,
            thinking_budget_tokens=selection.thinking_budget_tokens,
            streaming=selection.streaming,
        )


def _mapping_value(source: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = source.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _caching_config_value(app_config: Mapping[str, object]) -> Mapping[str, object]:
    """Return the ``[caching]`` section from either config shape.

    Boot-time/live Console config (``load_settings()``) never projects
    ``[caching]`` to the top level the way it does ``api_settings`` or
    ``chat_defaults`` -- it only survives nested under
    ``COMPREHENSIVE_CONFIG_RAW`` (see ``config.py``'s ``load_settings``).
    A plain ``app_config.get("caching")`` therefore always misses on the
    live Console config and silently reads the kill-switch as always-on
    (Qodo finding, PR #1239). Prefer a top-level ``caching`` key when a
    caller supplies one directly (e.g. tests), and fall back to the
    nested raw-TOML shape otherwise.
    """
    top_level = _mapping_value(app_config, "caching")
    if top_level:
        return top_level
    raw = _mapping_value(app_config, "COMPREHENSIVE_CONFIG_RAW")
    return _mapping_value(raw, "caching")


def _is_iterable_response(response: Any) -> bool:
    return isinstance(response, (Iterator, GeneratorType)) and not isinstance(
        response, (str, bytes, Mapping, list, tuple)
    )


def _maybe_record_usage(
    payload: Mapping[str, Any],
    signals: "_ProviderStreamSignals | None",
) -> None:
    if signals is None:
        return
    usage = payload.get("usage")
    if isinstance(usage, Mapping) and usage:
        signals.record_usage_payload(usage)


def _content_from_provider_item(
    item: Any,
    *,
    signals: "_ProviderStreamSignals | None" = None,
) -> str | object:
    if isinstance(item, str):
        if item.startswith("data:"):
            return _content_from_sse_data(item, signals=signals)
        return item
    if isinstance(item, bytes):
        decoded = item.decode("utf-8", errors="replace")
        if decoded.startswith("data:"):
            return _content_from_sse_data(decoded, signals=signals)
        return decoded
    if isinstance(item, Mapping):
        _maybe_record_usage(item, signals)
        return _content_from_provider_mapping(item)
    return _UNSUPPORTED_RESPONSE


def _content_from_sse_data(
    line: str,
    *,
    signals: "_ProviderStreamSignals | None" = None,
) -> str | object:
    ConsoleProviderGateway._raise_for_sse_error(line)
    data = line.removeprefix("data:").strip()
    if not data or data == "[DONE]":
        return _EMPTY_RESPONSE
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return _EMPTY_RESPONSE
    if not isinstance(payload, Mapping):
        return _EMPTY_RESPONSE
    _maybe_record_usage(payload, signals)
    content = _content_from_provider_mapping(payload)
    return _EMPTY_RESPONSE if content is _UNSUPPORTED_RESPONSE else content


def _content_from_provider_mapping(item: Mapping[str, Any]) -> str | object:
    choices = item.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            delta = first.get("delta")
            if isinstance(delta, Mapping) and isinstance(delta.get("content"), str):
                return delta["content"]
            message = first.get("message")
            if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                return message["content"]
            text = first.get("text")
            if isinstance(text, str):
                return text
    elif choices == [] and isinstance(item.get("usage"), Mapping):
        return _EMPTY_RESPONSE

    candidates = item.get("candidates")
    if isinstance(candidates, list) and candidates:
        first_candidate = candidates[0]
        if isinstance(first_candidate, Mapping):
            content = first_candidate.get("content")
            if isinstance(content, Mapping):
                parts = content.get("parts")
                if isinstance(parts, list):
                    text_parts = [
                        part["text"]
                        for part in parts
                        if isinstance(part, Mapping)
                        and isinstance(part.get("text"), str)
                    ]
                    if text_parts:
                        return "".join(text_parts)
            text = first_candidate.get("text")
            if isinstance(text, str):
                return text

    message = item.get("message")
    if isinstance(message, Mapping) and isinstance(message.get("content"), str):
        return message["content"]

    for key in ("content", "text", "response", "generated_text"):
        value = item.get(key)
        if isinstance(value, str):
            return value

    return _UNSUPPORTED_RESPONSE


def _provider_settings(
    app_config: Mapping[str, object], provider_key: str
) -> Mapping[str, object]:
    api_settings = _mapping_value(app_config, "api_settings")
    return provider_settings_for_key(api_settings, provider_key)


def _hosted_transport_policy(
    settings: Mapping[str, object],
    *,
    provider: str,
) -> tuple[float, int, float]:
    delay_default = 1.0 if provider == "moonshot" else 5.0
    timeout = settings.get("timeout", 90.0)
    retries = settings.get("retries", 3)
    retry_delay = settings.get("retry_delay", delay_default)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout <= 0
        or type(retries) is not int
        or retries < 0
        or isinstance(retry_delay, bool)
        or not isinstance(retry_delay, (int, float))
        or not math.isfinite(float(retry_delay))
        or retry_delay < 0
    ):
        raise ChatConfigurationError("Hosted provider transport policy is invalid.")
    return float(timeout), retries, float(retry_delay)


def _first_string(*values: object) -> str | None:
    for value in values:
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return None
