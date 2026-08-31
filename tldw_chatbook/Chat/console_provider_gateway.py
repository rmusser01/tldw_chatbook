"""Console-native provider resolution and streaming gateway."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import math
import threading
import uuid
import weakref
from collections.abc import Awaitable, Iterator, Mapping, Sequence
from contextvars import copy_context
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import GeneratorType, MappingProxyType
from typing import Any, AsyncIterator, Callable, Literal, TypeVar, cast
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
from tldw_chatbook.Chat.console_dispatch_checkpoint import ConsoleResolvedDestination
from tldw_chatbook.Chat.console_exchange_capture import (
    CaptureBudget,
    CaptureDetail,
    ExchangeCapture,
    build_request_capture,
    compact_safe_history_rows,
    sanitize_capture_value_with_omission,
)
from tldw_chatbook.Chat.console_project_instructions import (
    EPHEMERAL_ORIGIN_KEY,
    canonical_provider_endpoint_identity,
)
from tldw_chatbook.Chat.console_library_destination import resolve_console_destination
from tldw_chatbook.Chat.console_provider_endpoints import (
    effective_provider_endpoint,
    generic_endpoint_differs,
    normalize_generic_endpoint_for_compare,
    provider_uses_endpoint,
    unsaved_endpoint_copy,
)
from tldw_chatbook.Chat.console_prepared_request import (
    CONTINUATION_OWNER_KEY,
    THINKING_OWNER_KEY,
    PreparedConsoleRequest,
    PreparedProviderRequest,
    WireStyle,
    attach_thinking_history,
    build_console_request,
    prepare_provider_request,
    resolve_request_capacity,
    thaw_json,
)
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
    ProviderRequestProvenance,
    RequestRouteTraceProvenance,
    TraceProvenanceAlignmentError,
    TraceProvenance,
    request_route_provenance,
)
from tldw_chatbook.Chat.console_trace_final_values import (
    ProviderOverlayProvenance,
    ProviderRequestShadowBundle,
    reconstruct_provider_gateway_kwargs,
    verify_provider_request_shadow,
)
from tldw_chatbook.Chat.console_trace_redaction import (
    CredentialSanitizer,
    PII_DETECTOR_UNAVAILABLE,
    redact_pii_value,
)
from tldw_chatbook.Chat.console_trace_models import TraceCallState
from tldw_chatbook.Chat.console_trace_service import TraceCallPersistenceError
from tldw_chatbook.Chat.console_trace_settlement import (
    MAX_TRACE_RESPONSE_BYTES,
    TraceResponseOmission,
)
from tldw_chatbook.Chat.console_thinking_history import (
    ProviderThinkingSidecar,
    ThinkingReplayTarget,
    resolve_thinking_history,
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
from tldw_chatbook.Chat.llamacpp_think_filter import StartAnchoredThinkSplitter
from tldw_chatbook.Chat.thinking_blocks import (
    MAX_THINKING_PROVENANCE_CHARS,
    MAX_THINKING_TEXT_BYTES,
    THINKING_ENVELOPE_VERSION,
    ThinkingHistoryPolicy,
)
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.console_session_settings import reasoning_effort_hint_for_model
from tldw_chatbook.LLM_Calls.qwencloud import (
    normalize_qwencloud_api_mode,
    normalize_qwencloud_base_url,
)
from tldw_chatbook.LLM_Calls.hosted_chat import (
    HostedChatTurn,
    ReasoningDisposition,
)
from tldw_chatbook.LLM_Calls.moonshot import MoonshotFinishPolicy
from tldw_chatbook.LLM_Calls.zai import ZAIFinishPolicy
from tldw_chatbook.config import ProviderSettingsError, provider_settings_for_key
from tldw_chatbook.Utils.input_validation import validate_url
from tldw_chatbook.Utils.sensitive_llm_logging import (
    is_sensitive_llm_request,
    sensitive_llm_request,
)
from tldw_chatbook.Utils.tls_trust import build_httpx_async_client


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
MAX_TRACE_RESPONSE_ITEMS = 1_024
_MAX_TRACE_ACCUMULATED_BYTES = MAX_TRACE_RESPONSE_BYTES - 262_144
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
_DISPLAYABLE_THINKING_EXECUTION_KEYS = frozenset(
    {"llama_cpp", "local_llamacpp", "vllm", "local_vllm"}
)
_HOSTED_THINKING_FINISH_POLICIES = MappingProxyType(
    {
        "moonshot": MoonshotFinishPolicy,
        "zai": ZAIFinishPolicy,
    }
)
_AdapterResult = TypeVar("_AdapterResult")


class _ProviderAdapterEntryCancelled(Exception):
    """Internal signal that stream cancellation won the adapter-entry claim."""


class _ProviderAdapterEntryGate:
    """Linearize cancellation against one adapter authority consumption."""

    __slots__ = ("_cancelled", "_claimed", "_lock")

    def __init__(self) -> None:
        self._cancelled = False
        self._claimed = False
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Cancel adapter entry unless the worker already owns the boundary."""

        with self._lock:
            if not self._claimed:
                self._cancelled = True

    def consume_or_cancel(
        self,
        admission: "_ProviderAdapterAdmission",
        issuer: object,
    ) -> None:
        """Consume the token and decide cancellation versus provider ownership."""

        with self._lock, admission._lock:
            if admission._issuer is not issuer or admission._consumed:
                raise TraceCallPersistenceError()
            admission._consumed = True
            if self._cancelled:
                raise _ProviderAdapterEntryCancelled()
            self._claimed = True


class TemporaryCaptureRequiresSave(RuntimeError):
    """A temporary conversation attempted durable Capture On."""


def require_durable_capture_admission(
    *,
    capture_mode: ConsoleTraceCaptureMode,
    ephemeral: bool,
) -> None:
    """Reject temporary Capture On before request serialization or adapter entry.

    Args:
        capture_mode: Frozen trace capture mode for the provider call.
        ephemeral: Whether the owning conversation lacks durable storage.

    Raises:
        TypeError: If either argument is not its exact boundary type.
        TemporaryCaptureRequiresSave: If Capture On is requested for a temporary
            conversation.
    """

    if type(capture_mode) is not ConsoleTraceCaptureMode:
        raise TypeError("capture_mode must be ConsoleTraceCaptureMode")
    if type(ephemeral) is not bool:
        raise TypeError("ephemeral must be a bool")
    if ephemeral and capture_mode is ConsoleTraceCaptureMode.CAPTURE_ON:
        raise TemporaryCaptureRequiresSave(
            "Temporary Capture On requires Save & Send or explicit Capture Off."
        )


class _ProviderAdapterAdmission:
    """Single-use proof that this gateway admitted one adapter entry."""

    __slots__ = ("_consumed", "_issuer", "_lock", "capture_mode", "route")

    def __init__(
        self,
        issuer: object,
        capture_mode: ConsoleTraceCaptureMode,
        route: ConsoleRequestRoute | None,
    ) -> None:
        self.capture_mode = capture_mode
        self.route = route
        self._issuer = issuer
        self._consumed = False
        self._lock = threading.Lock()

    def consume(self, issuer: object) -> None:
        with self._lock:
            if self._issuer is not issuer or self._consumed:
                raise TraceCallPersistenceError()
            self._consumed = True


def _validate_request_trace_binding(
    request: PreparedConsoleRequest | PreparedProviderRequest,
    *,
    route: ConsoleRequestRoute | None,
    route_actor_id: str | None,
    route_chain_id: str | None,
    capture_mode: ConsoleTraceCaptureMode,
) -> None:
    """Fail closed unless preparation provenance matches this dispatch."""

    if type(capture_mode) is not ConsoleTraceCaptureMode:
        raise TypeError("capture_mode must be ConsoleTraceCaptureMode")
    provenance = request.provenance
    if capture_mode is ConsoleTraceCaptureMode.CAPTURE_OFF:
        if provenance is not None:
            raise TraceProvenanceAlignmentError(
                "Capture Off cannot dispatch a capture-on prepared request"
            )
        return
    if provenance is None:
        raise TraceProvenanceAlignmentError(
            "Capture On requires prepared request provenance"
        )
    route_descriptor = (
        request_route_provenance(
            route,
            actor_id=route_actor_id,
            chain_id=route_chain_id,
        )
        if route is not None
        else None
    )
    route_descriptors = tuple(
        item
        for item in provenance.metadata
        if type(item) is RequestRouteTraceProvenance
    )
    if (
        route_descriptor is None
        or len(route_descriptors) != 1
        or route_descriptors[0] != route_descriptor
    ):
        raise TraceProvenanceAlignmentError(
            "capture-on request route provenance is missing or mismatched"
        )


def _thinking_stream_capability(
    execution_key: str,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, ReasoningDisposition | int | None]:
    key = execution_key.strip().lower()
    if key in _DISPLAYABLE_THINKING_EXECUTION_KEYS:
        effort = str(reasoning_effort or "").strip().lower()
        disposition: ReasoningDisposition = (
            "displayable"
            if effort != "none"
            and (bool(effort) or reasoning_effort_hint_for_model(model) is not None)
            else "ignored"
        )
        return {
            "thinking_stream_disposition": disposition,
            "thinking_round_trip_version": (
                THINKING_ENVELOPE_VERSION if disposition == "displayable" else None
            ),
        }
    policy = _HOSTED_THINKING_FINISH_POLICIES.get(key)
    disposition: ReasoningDisposition = (
        policy.reasoning_disposition if policy is not None else "ignored"
    )
    return {
        "thinking_stream_disposition": disposition,
        "thinking_round_trip_version": (
            THINKING_ENVELOPE_VERSION if disposition != "ignored" else None
        ),
    }


class ProviderThinkingCaptureError(RuntimeError):
    """A provider-local thinking capture failed without exposing its content."""


def _is_strict_utf8_text(value: str) -> bool:
    """Return whether text contains no unencodable surrogate code points."""
    return all(not 0xD800 <= ord(character) <= 0xDFFF for character in value)


def _is_valid_provider_thinking_identity(value: object) -> bool:
    return (
        type(value) is str
        and bool(value.strip())
        and len(value) <= MAX_THINKING_PROVENANCE_CHARS
        and _is_strict_utf8_text(value)
    )


def _is_valid_provider_thinking_text(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and _is_strict_utf8_text(value)
        and len(value.encode("utf-8")) <= MAX_THINKING_TEXT_BYTES
    )


@dataclass(frozen=True, slots=True, init=False)
class ProviderThinkingDelta:
    """One bounded displayable thinking fragment from an approved adapter."""

    text: str = field(repr=False)
    provider: str
    model: str
    protocol: str
    source_format: str

    def __init__(
        self,
        text: str,
        provider: str,
        model: str,
        protocol: str,
        source_format: str,
    ) -> None:
        # Initialize only content-free state before validation so a rejected
        # identity cannot survive through constructor traceback locals.
        object.__setattr__(self, "text", "")
        object.__setattr__(self, "provider", "")
        object.__setattr__(self, "model", "")
        object.__setattr__(self, "protocol", "")
        object.__setattr__(self, "source_format", "")
        valid = (
            _is_valid_provider_thinking_text(text)
            and _is_valid_provider_thinking_identity(provider)
            and _is_valid_provider_thinking_identity(model)
            and _is_valid_provider_thinking_identity(protocol)
            and _is_valid_provider_thinking_identity(source_format)
        )
        if not valid:
            del text, provider, model, protocol, source_format
            raise ValueError("Invalid provider thinking event.")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "source_format", source_format)


@dataclass(frozen=True, slots=True, init=False)
class ProviderProprietaryThinkingEvidence:
    """Content-free proof that an approved adapter observed private reasoning."""

    provider: str
    model: str
    protocol: str
    source_format: str

    def __init__(
        self,
        provider: str,
        model: str,
        protocol: str,
        source_format: str,
    ) -> None:
        object.__setattr__(self, "provider", "")
        object.__setattr__(self, "model", "")
        object.__setattr__(self, "protocol", "")
        object.__setattr__(self, "source_format", "")
        valid = (
            _is_valid_provider_thinking_identity(provider)
            and _is_valid_provider_thinking_identity(model)
            and _is_valid_provider_thinking_identity(protocol)
            and _is_valid_provider_thinking_identity(source_format)
        )
        if not valid:
            del provider, model, protocol, source_format
            raise ValueError("Invalid provider thinking event.")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "source_format", source_format)


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
    model_retry_callback: Callable[[], None] | None = field(
        default=None,
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
    _trace_settlement_sink: Callable[[object], object] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _trace_settlement_lock: threading.Lock = field(
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

    def mark_model_retry(self) -> None:
        """Report an observed provider retry without coupling to its owner."""
        callback = self.model_retry_callback
        if callback is None:
            return
        try:
            callback()
        except Exception:
            logger.warning("model_retry_callback_failed")

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

    def bind_trace_settlement_sink(self, sink: Callable[[object], object]) -> None:
        """Bind one run-owned explicit handoff sink before provider dispatch."""

        if not callable(sink):
            raise TypeError("sink")
        with self._trace_settlement_lock:
            self._trace_settlement_sink = sink

    async def _publish_trace_settlement(self, handoff: object) -> bool:
        with self._trace_settlement_lock:
            sink = self._trace_settlement_sink
        if sink is None:
            return False
        try:
            result = sink(handoff)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            await self._settle_caller_owned_trace_handoff(handoff)
            raise
        except Exception as exc:
            logger.warning("trace_settlement_handoff_failed: {}", type(exc).__name__)
            await self._settle_caller_owned_trace_handoff(handoff)
            # A bound sink failure remains caller-owned here. Returning true
            # prevents the legacy no-sink path from repeating SQLite work on
            # the event-loop thread after the awaited off-thread attempt.
            return True
        return True

    @staticmethod
    async def _settle_caller_owned_trace_handoff(handoff: object) -> None:
        settle = getattr(handoff, "settle", None)
        if not callable(settle):
            return
        try:
            await asyncio.to_thread(settle, None)
        except Exception as exc:
            logger.warning(
                "trace_response_handoff_fallback_failed: {}",
                type(exc).__name__,
            )

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

    run_tag: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Fail-safe default: OFF. A bare `ConsoleProviderStreamSignals()` (every
    # construction site that does not explicitly opt in -- visual
    # evaluation, the agent-bridge fallback) must never capture. Only
    # `_new_run_stream_signals()` (console_chat_controller.py) opts in,
    # reading the actual `[console] exchange_capture` config gate (review
    # finding I1: the two bare-construction sites used to inherit `True`
    # and capture unconditionally, for output nobody ever reads).
    exchange_capture_enabled: bool = False
    capture_detail: CaptureDetail = field(default=CaptureDetail.SAFE, repr=False)
    pii_redaction_enabled: bool = field(default=False, repr=False)
    completed_exchanges: list["ExchangeCapture"] = field(
        default_factory=list, repr=False
    )
    _active_exchanges: dict[object, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _exchange_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def _begin_scoped_exchange(self, token: object, flight: dict[str, Any]) -> None:
        with self._exchange_lock:
            self._active_exchanges[token] = flight

    def _mutate_scoped_exchange(self, token: object, key: str, items: list) -> None:
        """Never raises (review finding M4): capture is diagnostic tooling
        layered over the real send path -- the gateway's own call sites
        (record_exchange_content/record_exchange_tool_calls) are NOT all
        wrapped in their own try/except, and three of them sit inside
        ``_stream_generic_chat``'s worker ``try``, whose ``except
        BaseException`` would otherwise convert a capture-bookkeeping bug
        into a fabricated provider error, turning a good turn into a failed
        one. No exception text/traceback logged -- ``items`` can hold raw
        captured request/response content."""
        try:
            with self._exchange_lock:
                flight = self._active_exchanges.get(token)
                if flight is not None:
                    retained = flight[key]
                    sanitized, omitted = sanitize_capture_value_with_omission(
                        items,
                        known_credentials=flight["known_credentials"],
                    )
                    if omitted:
                        path = f"response.{key}"
                        if path not in flight["credential_omission_inventory"]:
                            flight["credential_omission_inventory"].append(path)
                    if not isinstance(sanitized, list):
                        return
                    for item in sanitized:
                        if flight["capture_budget"].retain(item):
                            retained.append(item)
                        elif key not in flight["response_truncation_inventory"]:
                            flight["response_truncation_inventory"].append(key)
        except Exception as exc:
            logger.warning(f"exchange_capture_mutate_failed: {type(exc).__name__}")

    def _mark_scoped_exchange_synthetic(self, token: object) -> None:
        """Stamp one call's in-flight record as carrying locally
        synthesized fallback UI copy, not provider output (review finding
        M3). Never raises -- same M4 contract as ``_mutate_scoped_
        exchange``."""
        try:
            with self._exchange_lock:
                flight = self._active_exchanges.get(token)
                if flight is not None:
                    flight["synthetic_fallback"] = True
        except Exception as exc:
            logger.warning(
                f"exchange_capture_mark_synthetic_failed: {type(exc).__name__}"
            )

    def _complete_scoped_exchange(
        self,
        token: object,
        status: str,
        usage_payload: dict[str, Any] | None,
    ) -> None:
        """Never raises (review finding M4) -- same "never break send"
        contract as ``_mutate_scoped_exchange``: this is the ``close_
        exchange`` call site's own implementation, called at both a
        `finally` (stream_chat) and inside ``_stream_generic_chat``'s
        worker `try`/`except` (twice), where an uncaught raise here would
        either mask the real cleanup or itself be relabeled a provider
        error. No exception text/traceback logged -- ``flight`` holds raw
        captured request/response content."""
        try:
            with self._exchange_lock:
                flight = self._active_exchanges.pop(token, None)
                if flight is None:
                    return
                self.completed_exchanges.append(
                    _flight_capture(
                        self.run_tag,
                        len(self.completed_exchanges),
                        flight,
                        status,
                        usage_payload,
                    )
                )
        except Exception as exc:
            logger.warning(f"exchange_capture_complete_failed: {type(exc).__name__}")

    def exchange_captures(self) -> list["ExchangeCapture"]:
        """Completed calls + in-flight tails (as "stopped") — tails cover
        aborted streams whose generator never reached its own close-out,
        mirroring usage_payloads()."""
        with self._exchange_lock:
            captures = list(self.completed_exchanges)
            for flight in self._active_exchanges.values():
                captures.append(
                    _flight_capture(
                        self.run_tag, len(captures), flight, "stopped", None
                    )
                )
            return captures


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
    # Review finding M3: set by mark_synthetic_fallback(), consumed by the
    # very next record_exchange_content() call in the generic stream loop --
    # NOT the aggregate's own sticky Event (that one never resets, and is
    # shared across every call this signals object ever makes; this one is
    # per-call and self-clearing, so only the ONE chunk actually generated
    # as fallback UI copy gets labeled, never a later real answer).
    _synthetic_pending: bool = field(default=False, init=False, repr=False)
    _synthetic_emitted: bool = field(default=False, init=False, repr=False)

    @property
    def synthetic_fallback_emitted(self) -> bool:
        """Return whether the aggregate emitted synthetic fallback usage."""
        return self._aggregate.synthetic_fallback_emitted

    @property
    def synthetic_copy_emitted(self) -> bool:
        """Return whether this call emitted locally synthesized UI copy."""

        return self._synthetic_emitted

    @property
    def exchange_capture_enabled(self) -> bool:
        """Return whether the aggregate has exchange capture enabled.

        Callers check this BEFORE doing any capture-building work (allowlist
        filtering, ``json.dumps``, ``stub_binary_strings``'s recursive
        walk) -- ``begin_exchange`` below also checks it, but only after
        that work is already done, so it cannot save the cost on its own
        (review finding I1).
        """
        return self._aggregate.exchange_capture_enabled

    @property
    def capture_detail(self) -> CaptureDetail:
        """Return the admission-frozen detail shared by this run."""
        return self._aggregate.capture_detail

    def mark_synthetic_fallback(self) -> None:
        """Mark synthetic fallback usage on the aggregate signal, and flag
        this call's NEXT recorded content chunk as synthetic (review
        finding M3 -- consumed once by ``take_synthetic_pending()``)."""
        self._synthetic_pending = True
        self._aggregate.mark_synthetic_fallback()

    def take_synthetic_pending(self) -> bool:
        """Consume (and clear) whether ``mark_synthetic_fallback()`` fired
        for the chunk about to be recorded. Self-clearing so only the one
        chunk actually generated as fallback UI copy is ever labeled."""
        pending = self._synthetic_pending
        self._synthetic_pending = False
        self._synthetic_emitted = self._synthetic_emitted or pending
        return pending

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

    async def publish_trace_settlement(self, handoff: object) -> bool:
        """Publish one sanitized explicit-call handoff to terminal persistence."""

        return await self._aggregate._publish_trace_settlement(handoff)

    def begin_exchange(
        self,
        *,
        provider: str,
        model: str,
        endpoint: str | None,
        request: dict,
        omitted_keys: tuple[str, ...],
        capture_budget: CaptureBudget | None = None,
        known_credentials: tuple[str, ...] = (),
        request_credentials_filtered: bool = False,
    ) -> None:
        """Open this call's capture. ONE stream_chat invocation == one
        exchange; close_exchange in stream_chat's finally is the close site.

        ``request`` must be a freshly built, allowlisted dict -- i.e.
        ``build_request_capture``'s output -- never raw ``chat_api_call``
        kwargs, which would alias live state and re-admit credentials.
        """
        if not self._aggregate.exchange_capture_enabled:
            return
        if endpoint is not None:
            try:
                endpoint = canonical_provider_endpoint_identity(endpoint)
            except ValueError:
                endpoint = "[invalid endpoint]"
        safe_provider, provider_omitted = sanitize_capture_value_with_omission(
            provider,
            known_credentials=known_credentials,
        )
        safe_model, model_omitted = sanitize_capture_value_with_omission(
            model,
            known_credentials=known_credentials,
        )
        safe_endpoint, endpoint_omitted = sanitize_capture_value_with_omission(
            endpoint,
            known_credentials=known_credentials,
        )
        safe_request, request_omitted = sanitize_capture_value_with_omission(
            request,
            # Request builders already applied the known-credential filter
            # before adding content-free structural capture markers. Avoid
            # interpreting a one-character test/local key inside those
            # markers while still applying the full recognized filter here.
            known_credentials=() if request_credentials_filtered else known_credentials,
        )
        omitted = set(omitted_keys)
        omitted.update(
            name
            for name, failed in (
                ("provider", provider_omitted),
                ("model", model_omitted),
                ("endpoint", endpoint_omitted),
                ("request", request_omitted),
            )
            if failed
        )
        self._aggregate._begin_scoped_exchange(
            self._token,
            {
                "provider": safe_provider if isinstance(safe_provider, str) else "",
                "model": safe_model if isinstance(safe_model, str) else "",
                "endpoint": safe_endpoint if isinstance(safe_endpoint, str) else None,
                "request": safe_request
                if isinstance(safe_request, dict)
                else {"omitted": True},
                "omitted_keys": tuple(sorted(omitted)),
                "content": [],
                "tool_calls": [],
                "synthetic_fallback": False,
                "response_truncation_inventory": [],
                "credential_omission_inventory": [],
                "known_credentials": known_credentials,
                "capture_detail": self.capture_detail,
                "pii_redaction_enabled": self._aggregate.pii_redaction_enabled,
                "capture_budget": capture_budget or CaptureBudget(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def record_exchange_content(self, text: str, *, synthetic: bool = False) -> None:
        """Append one content chunk to this call's in-flight capture.

        Args:
            synthetic: True when ``text`` is locally synthesized fallback
                UI copy (``NO_PROVIDER_CONTENT_COPY``/``UNSUPPORTED_
                PROVIDER_RESPONSE_COPY``), never actual provider output --
                stamped into the capture's response so the Exchange tab can
                label it instead of presenting UI copy as a model answer
                (review finding M3).
        """
        if text:
            self._aggregate._mutate_scoped_exchange(self._token, "content", [text])
            if synthetic:
                self._aggregate._mark_scoped_exchange_synthetic(self._token)

    def record_exchange_tool_calls(self, calls: "Sequence[Mapping[str, Any]]") -> None:
        # Review finding M9: `dict(c)` is a SHALLOW copy -- the nested
        # `function` dict (and any other nested mapping/list) stays aliased
        # to the live object the caller passed in until this flush reaches
        # `close_exchange`/`_flight_capture`, seconds later on a real turn.
        # `deepcopy` closes that window permanently.
        self._aggregate._mutate_scoped_exchange(
            self._token, "tool_calls", [deepcopy(dict(c)) for c in calls]
        )

    def close_exchange(self, status: str = "complete") -> None:
        """Publish this call's capture exactly once (token pop = move
        semantics; a second close finds nothing)."""
        self._aggregate._complete_scoped_exchange(
            self._token, status, self.usage_snapshot()
        )


_ProviderStreamSignals = ConsoleProviderStreamSignals | ConsoleProviderCallSignals

_PROVIDER_REQUEST_FAILED_COPY = "Provider request failed."


def _sanitized_provider_diagnostic(
    value: object,
    *,
    known_credentials: tuple[str, ...] = (),
) -> str:
    """Return diagnostic copy only when credential filtering changed nothing."""
    result = CredentialSanitizer(known_credentials=known_credentials).sanitize(value)
    if not result.available or result.redacted or type(result.value) is not str:
        return _PROVIDER_REQUEST_FAILED_COPY
    return result.value


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
    provider_copy = _sanitized_provider_diagnostic(provider or "unknown")
    if provider_copy == _PROVIDER_REQUEST_FAILED_COPY:
        return provider_copy
    status_code = getattr(exc, "status_code", None)
    status_copy = f" Status: {status_code}." if type(status_code) is int else ""
    return _sanitized_provider_diagnostic(
        f"Provider error from {provider_copy}: {category}.{status_copy}"
    )


def _flight_capture(
    run_tag: str,
    seq: int,
    flight: dict[str, Any],
    status: str,
    usage_payload: dict[str, Any] | None,
) -> ExchangeCapture:
    """Build the immutable capture for one call's in-flight record.

    Normalizes THIS call's usage payload on its own (never a cross-call
    merge — the same disjoint-buckets rule the aggregate documents).
    """
    usage_json = None
    if usage_payload:
        try:
            usage = ProviderUsage.from_provider_payload(
                usage_payload, provider=flight["provider"], model=flight["model"]
            )
            usage_json = usage.to_json() if usage is not None else None
        except Exception:
            usage_json = None
    content, content_omitted = sanitize_capture_value_with_omission(
        "".join(flight["content"]),
        known_credentials=flight["known_credentials"],
    )
    tool_calls, tools_omitted = sanitize_capture_value_with_omission(
        deepcopy(flight["tool_calls"]),
        known_credentials=flight["known_credentials"],
    )
    credential_omissions = list(flight.get("credential_omission_inventory", ()))
    if content_omitted:
        credential_omissions.append("response.content")
    if tools_omitted:
        credential_omissions.append("response.tool_calls")
    request = flight["request"]
    if flight.get("pii_redaction_enabled") is True:
        request_redaction = redact_pii_value(request)
        content_redaction = redact_pii_value(content)
        tool_redaction = redact_pii_value(tool_calls)
        request = (
            request_redaction.value
            if request_redaction.available
            else {"omitted": PII_DETECTOR_UNAVAILABLE}
        )
        content = (
            content_redaction.value
            if content_redaction.available
            else f"[omitted: {PII_DETECTOR_UNAVAILABLE}]"
        )
        tool_calls = (
            tool_redaction.value
            if tool_redaction.available
            else [{"omitted": PII_DETECTOR_UNAVAILABLE}]
        )
        for path, result in (
            ("request", request_redaction),
            ("response.content", content_redaction),
            ("response.tool_calls", tool_redaction),
        ):
            if not result.available:
                credential_omissions.append(path + ".pii_unavailable")
    return ExchangeCapture(
        run_tag=run_tag,
        seq=seq,
        created_at=flight["created_at"],
        provider=flight["provider"],
        model=flight["model"],
        endpoint=flight["endpoint"],
        request=request,
        response={
            # Sanitize once more after aggregation: individually harmless
            # sub-threshold chunks can form one data URI/base64 body.
            "content": content,
            "tool_calls": tool_calls,
            "synthetic_fallback": bool(flight.get("synthetic_fallback", False)),
            "truncation_inventory": tuple(
                flight.get("response_truncation_inventory", ())
            ),
            "credential_omission_inventory": tuple(sorted(set(credential_omissions))),
        },
        status=status,
        usage_json=usage_json,
        omitted_keys=tuple(
            sorted(set(flight["omitted_keys"]).union(credential_omissions))
        ),
        capture_detail=flight["capture_detail"],
    )


def _provider_error_copy_with_model_recovery(
    copy: str,
    *,
    model: str | None,
    status_code: int | None,
) -> str:
    """Add safe model-specific recovery to provider bad-request copy."""
    if status_code != 400:
        return copy
    model_result = CredentialSanitizer().sanitize(model or "")
    if (
        not model_result.available
        or model_result.redacted
        or type(model_result.value) is not str
    ):
        return copy
    model_id = "".join(
        character for character in model_result.value.strip() if character.isprintable()
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
    resolved_destination: ConsoleResolvedDestination | None = None
    thinking_stream_disposition: ReasoningDisposition = "ignored"
    thinking_round_trip_version: int | None = None

    def __post_init__(self) -> None:
        valid_disposition = self.thinking_stream_disposition in {
            "displayable",
            "proprietary",
            "ignored",
        }
        valid_version = (
            self.thinking_round_trip_version is None
            if self.thinking_stream_disposition == "ignored"
            else type(self.thinking_round_trip_version) is int
            and self.thinking_round_trip_version == THINKING_ENVELOPE_VERSION
        )
        if not valid_disposition or not valid_version:
            raise ValueError("Invalid provider thinking capability.")

    @property
    def may_emit_thinking(self) -> bool:
        """Whether this frozen adapter target can emit typed thinking evidence."""
        return self.thinking_stream_disposition != "ignored"


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


def _validate_auxiliary_content(role: str, content: Any) -> None:
    """Accept text or the exact provider-visible Console image-part shape."""

    if isinstance(content, str):
        return
    if not isinstance(content, (list, tuple)) or not content:
        raise TypeError("Auxiliary message content must be text or content parts.")
    if role != "user":
        raise ValueError("Auxiliary multimodal content must use the user role.")
    for part in content:
        if not isinstance(part, Mapping):
            raise TypeError("Auxiliary content parts must be mappings.")
        part_type = part.get("type")
        if part_type == "text":
            if set(part) != {"type", "text"} or not isinstance(
                part.get("text"), str
            ):
                raise ValueError("Auxiliary text parts are invalid.")
            continue
        if part_type != "image_url" or set(part) != {"type", "image_url"}:
            raise ValueError("Auxiliary content part type is unsupported.")
        image_url = part.get("image_url")
        if not isinstance(image_url, Mapping) or set(image_url) != {"url"}:
            raise ValueError("Auxiliary image parts are invalid.")
        url = image_url.get("url")
        if not isinstance(url, str):
            raise ValueError("Auxiliary image parts are invalid.")
        header, separator, encoded = url.partition(",")
        if (
            not separator
            or not header.startswith("data:image/")
            or not header.endswith(";base64")
            or not encoded
        ):
            raise ValueError("Auxiliary image parts require inline image data.")


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
            _validate_auxiliary_content(role, content)
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
    synthetic: bool = False
    # F5: the real HTTP status, carried alongside the (already-redacted)
    # text -- never re-derived by parsing that text back out. `None` means
    # "no real status available" (a bare RuntimeError, say), which the
    # consumer maps to ChatProviderError's own upstream-error default.
    status_code: int | None = None

    @classmethod
    def content(cls, text: str, *, synthetic: bool = False) -> "_QueueItem":
        return cls("content", text, synthetic=synthetic)

    @classmethod
    def error(cls, text: str, status_code: int | None = None) -> "_QueueItem":
        return cls("error", text, status_code=status_code)

    @classmethod
    def trace_verification_error(cls) -> "_QueueItem":
        """Carry only a typed, content-free verification failure."""

        return cls("trace_verification_error")

    @classmethod
    def trace_persistence_error(cls, boundary: object | None = None) -> "_QueueItem":
        """Carry only a typed, content-free pre-dispatch write failure."""

        return cls("trace_persistence_error", payload=boundary)

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

    @classmethod
    def thinking(cls, event: ProviderStreamItem) -> "_QueueItem":
        return cls("thinking", payload=event)


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


ProviderStreamItem = (
    str
    | ProviderToolCalls
    | ProviderThinkingDelta
    | ProviderProprietaryThinkingEvidence
)


@dataclass(frozen=True, slots=True)
class _ProviderStreamEmission:
    """One internal stream item plus whether its UI copy is synthetic."""

    item: ProviderStreamItem = field(repr=False)
    synthetic: bool = False


@dataclass(slots=True)
class _TraceResponseAccumulator:
    """Retain only a bounded semantic prefix for trace settlement."""

    _items: list[ProviderStreamItem] = field(default_factory=list, repr=False)
    _retained_bytes: int = 0
    omission_reason: str | None = None
    semantic_observed: bool = False
    synthetic_observed: bool = False

    @property
    def items(self) -> tuple[ProviderStreamItem, ...]:
        return tuple(self._items)

    def observe(self, item: ProviderStreamItem, *, synthetic: bool) -> bool:
        """Return whether this is the first real provider semantic item."""

        if synthetic:
            self.synthetic_observed = True
            return False
        first_semantic = not self.semantic_observed
        self.semantic_observed = True
        if self.omission_reason is not None:
            return first_semantic
        if len(self._items) >= MAX_TRACE_RESPONSE_ITEMS:
            self._omit("response_item_limit")
            return first_semantic
        item_bytes = _trace_response_item_bytes(item)
        if (
            item_bytes is None
            or self._retained_bytes + item_bytes > _MAX_TRACE_ACCUMULATED_BYTES
        ):
            self._omit("response_accumulation_limit")
            return first_semantic
        self._items.append(_retained_trace_response_item(item))
        self._retained_bytes += item_bytes
        return first_semantic

    def _omit(self, reason: str) -> None:
        self._items.clear()
        self._retained_bytes = 0
        self.omission_reason = reason


def _mark_trace_response_started(boundary: object | None) -> None:
    marker = getattr(boundary, "mark_response_started", None)
    if not callable(marker):
        return
    try:
        marker()
    except Exception as exc:
        logger.warning("trace_response_checkpoint_failed: {}", type(exc).__name__)


async def _settle_trace_response(
    boundary: object | None,
    items: Sequence[ProviderStreamItem],
    *,
    outcome: TraceCallState,
    usage: Mapping[str, object] | None,
    response_omission: str | None = None,
    signals: ConsoleProviderCallSignals | None = None,
) -> None:
    envelope = (
        TraceResponseOmission(response_omission)
        if response_omission is not None
        else None
        if outcome in {TraceCallState.ERROR, TraceCallState.STOPPED} and not items
        else _provider_response_envelope(items)
    )
    preparer = getattr(boundary, "prepare_response_settlement", None)
    if callable(preparer):
        try:
            handoff = preparer(envelope, outcome, usage)
            if handoff is not None:
                if signals is not None and await signals.publish_trace_settlement(
                    handoff
                ):
                    return
                settle = getattr(handoff, "settle", None)
                if callable(settle):
                    settle(None)
                    return
        except Exception as exc:
            logger.warning("trace_response_handoff_failed: {}", type(exc).__name__)
    settler = getattr(boundary, "settle_response", None)
    if not callable(settler):
        return
    try:
        settler(envelope, outcome, usage)
    except Exception as exc:
        logger.warning("trace_response_settlement_failed: {}", type(exc).__name__)


def _provider_response_envelope(
    items: Sequence[ProviderStreamItem],
) -> dict[str, object] | None:
    """Assemble one normalized provider-facing response without UI copy."""

    if not items:
        return {"role": "assistant", "content": ""}
    envelope: dict[str, object] = {
        "role": "assistant",
        "content": "".join(item for item in items if isinstance(item, str)),
    }
    tool_calls = [
        deepcopy(call)
        for item in items
        if isinstance(item, ProviderToolCalls)
        for call in item.tool_calls
    ]
    if tool_calls:
        envelope["tool_calls"] = tool_calls
    thinking = [
        {
            "text": item.text,
            "provider": item.provider,
            "model": item.model,
            "protocol": item.protocol,
            "source_format": item.source_format,
        }
        for item in items
        if isinstance(item, ProviderThinkingDelta)
    ]
    if thinking:
        envelope["thinking"] = thinking
    proprietary = [
        {
            "provider": item.provider,
            "model": item.model,
            "protocol": item.protocol,
            "source_format": item.source_format,
        }
        for item in items
        if isinstance(item, ProviderProprietaryThinkingEvidence)
    ]
    if proprietary:
        envelope["proprietary_thinking_evidence"] = proprietary
    return envelope


def _trace_response_item_bytes(item: ProviderStreamItem) -> int | None:
    """Measure one normalized semantic item without retaining its raw value."""

    try:
        envelope = _provider_response_envelope((item,))
        return len(
            json.dumps(
                envelope,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
    except (TypeError, ValueError):
        return None


def _retained_trace_response_item(item: ProviderStreamItem) -> ProviderStreamItem:
    """Detach bounded semantic tool data from non-envelope metadata."""

    if isinstance(item, ProviderToolCalls):
        return ProviderToolCalls(
            tuple(deepcopy(call) for call in item.tool_calls),
            metadata=None,
        )
    return item


@dataclass(frozen=True, slots=True)
class _LocalCompletionResult:
    items: tuple[ProviderStreamItem, ...] = field(repr=False)
    capture_failed: bool = False


def _unpack_local_completion_result(
    result: str | _LocalCompletionResult | tuple[ProviderStreamItem, ...],
) -> tuple[tuple[ProviderStreamItem, ...], bool]:
    if isinstance(result, _LocalCompletionResult):
        return result.items, result.capture_failed
    if isinstance(result, str):
        return (result,), False
    return result, False


def _local_thinking_delta(
    text: str,
    *,
    provider: str,
    model: str,
    protocol: str,
) -> ProviderThinkingDelta:
    return ProviderThinkingDelta(
        text=text,
        provider=provider,
        model=model,
        protocol=protocol,
        source_format="start_anchored_think",
    )


def _split_local_completion_items(
    text: str,
    *,
    provider: str,
    model: str,
    protocol: str,
) -> _LocalCompletionResult:
    splitter = StartAnchoredThinkSplitter()
    update = splitter.feed(text)
    terminal = splitter.flush()
    items: list[ProviderStreamItem] = []
    thinking = update.thinking + terminal.thinking
    content = update.content + terminal.content
    if thinking:
        items.append(
            _local_thinking_delta(
                thinking,
                provider=provider,
                model=model,
                protocol=protocol,
            )
        )
    if content:
        items.append(content)
    return _LocalCompletionResult(
        items=tuple(items),
        capture_failed=terminal.status == "failed",
    )


def _thinking_protocol(resolution: ConsoleProviderResolution) -> str:
    return resolution.continuation_protocol or resolution.api_mode or "chat_completions"


def _proprietary_thinking_event(
    response: Any,
    resolution: ConsoleProviderResolution,
) -> ProviderProprietaryThinkingEvidence | None:
    if resolution.thinking_stream_disposition != "proprietary":
        return None
    try:
        turn = response.terminal_turn
    except AttributeError:
        return None
    if not isinstance(turn, HostedChatTurn):
        raise ChatProviderError("Provider terminal metadata is malformed.")
    if not turn.reasoning_content:
        return None
    return ProviderProprietaryThinkingEvidence(
        provider=resolution.execution_key or resolution.provider,
        model=cast(str, resolution.model),
        protocol=_thinking_protocol(resolution),
        source_format="reasoning_content",
    )


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
        "messages": [
            {
                key: value
                for key, value in message.items()
                if key != EPHEMERAL_ORIGIN_KEY
            }
            for message in messages
        ],
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
        trace_shadow_sink: Optional in-memory consumer for verified, sanitized
            Capture-On provider values. Slice B does not persist this bundle.
        trace_call_boundary_factory: Optional hard-off normalized-writer seam.
            When supplied, each Capture-On call must reserve and commit
            ``dispatch_started`` before adapter entry.
    """

    deferred_dispatch_boundary = True

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        config_provider: Callable[[], Mapping[str, object]] | None = None,
        environ: Mapping[str, str] | None = None,
        chat_api_call_fn: Callable[..., Any] | None = None,
        safe_error_copy: Callable[[str, BaseException], str] | None = None,
        trace_shadow_sink: Callable[[ProviderRequestShadowBundle], None] | None = None,
        trace_call_boundary_factory: (
            Callable[
                [
                    PreparedProviderRequest,
                    ConsoleProviderResolution,
                    ConsoleRequestRoute | None,
                ],
                object,
            ]
            | None
        ) = None,
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
        self._trace_shadow_sink = trace_shadow_sink
        self._trace_call_boundary_factory = trace_call_boundary_factory
        self._adapter_admission_issuer = object()

    def _capture_off_admission(
        self, route: ConsoleRequestRoute | None
    ) -> _ProviderAdapterAdmission:
        """Explicitly admit one untraced adapter entry."""

        return _ProviderAdapterAdmission(
            self._adapter_admission_issuer,
            ConsoleTraceCaptureMode.CAPTURE_OFF,
            route,
        )

    def _enter_provider_adapter(
        self,
        admission: _ProviderAdapterAdmission,
        adapter: Callable[..., _AdapterResult],
        *args: Any,
        **kwargs: Any,
    ) -> _AdapterResult:
        """Consume one gateway-issued admission immediately before adapter entry."""

        if type(admission) is not _ProviderAdapterAdmission:
            raise TraceCallPersistenceError()
        _entry_gate = kwargs.pop("_console_adapter_entry_gate", None)
        if _entry_gate is not None:
            if type(_entry_gate) is not _ProviderAdapterEntryGate:
                raise TraceCallPersistenceError()
            _entry_gate.consume_or_cancel(
                admission,
                self._adapter_admission_issuer,
            )
            return adapter(*args, **kwargs)
        # Consumption is owned by the gateway, not dynamically dispatched to
        # the presented object's method.  Otherwise a subclass can override
        # ``consume`` and forge entry without possessing this issuer.
        with admission._lock:
            if (
                admission._issuer is not self._adapter_admission_issuer
                or admission._consumed
            ):
                raise TraceCallPersistenceError()
            admission._consumed = True
        return adapter(*args, **kwargs)

    def _reserve_trace_call(
        self,
        request: PreparedProviderRequest,
        resolution: ConsoleProviderResolution,
        route: ConsoleRequestRoute | None,
    ) -> object:
        """Create and reserve one distinct Capture-On call boundary."""

        if self._trace_call_boundary_factory is None:
            raise TraceCallPersistenceError(reservation_status="not_established")
        boundary: object | None = None
        try:
            boundary = self._trace_call_boundary_factory(request, resolution, route)
            reserve = getattr(boundary, "reserve", None)
            if not callable(reserve):
                raise TraceCallPersistenceError()
            reserve()
            return boundary
        except TraceCallPersistenceError as exc:
            if exc.boundary is None and boundary is not None:
                raise TraceCallPersistenceError(boundary=boundary) from None
            raise
        except Exception:
            raise TraceCallPersistenceError(reservation_status="unknown") from None

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
            others: list[tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]] = []
            still_live: list[tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]] = []
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
        thinking_sidecar: tuple[ProviderThinkingSidecar, ...] = (),
        thinking_policy: ThinkingHistoryPolicy | None = None,
        thinking_owner_key: str | None = None,
        route: ConsoleRequestRoute | None = None,
        route_actor_id: str | None = None,
        route_chain_id: str | None = None,
        capture_mode: ConsoleTraceCaptureMode = ConsoleTraceCaptureMode.CAPTURE_OFF,
        ephemeral: bool = False,
    ) -> PreparedProviderRequest:
        """Prepare the one immutable payload later consumed by dispatch.

        Model capability facts are read once here.  Unknown models remain
        explicitly unverified; an optional user override is enforced as a
        bound but never labeled as provider-verified.
        """

        require_durable_capture_admission(
            capture_mode=capture_mode,
            ephemeral=ephemeral,
        )
        if isinstance(messages, PreparedConsoleRequest) and tools is not None:
            raise ValueError("tools are already owned by PreparedConsoleRequest")
        sidecar = tuple(continuation_sidecar)
        thinking_sidecars = tuple(thinking_sidecar)
        if sidecar and (continuation_target is None or not continuation_owner_key):
            raise ValueError(
                "continuation target and owner key are required for private history"
            )
        if thinking_sidecars and not thinking_owner_key:
            raise ValueError("thinking owner key is required for thinking history")
        if continuation_target is not None and (
            continuation_target.provider,
            continuation_target.model,
            normalize_generic_endpoint_for_compare(continuation_target.api_base_url),
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
            semantic = (
                replace(messages, effective_thinking_policy="required")
                if continuation_groups
                and messages.effective_thinking_policy != "required"
                else messages
            )
            if thinking_sidecars:
                assert thinking_owner_key is not None
                selected_thinking_owner_ids = {
                    message.get(thinking_owner_key)
                    for message in semantic.flattened_messages()
                    if type(message.get(thinking_owner_key)) is str
                }
                thinking = resolve_thinking_history(
                    target=ThinkingReplayTarget(
                        provider=resolution.execution_key or resolution.provider,
                        model=resolution.model or "",
                        protocol=_thinking_protocol(resolution),
                        disposition=resolution.thinking_stream_disposition,
                        round_trip_version=resolution.thinking_round_trip_version,
                    ),
                    policy=thinking_policy,
                    sidecars=tuple(
                        item
                        for item in thinking_sidecars
                        if item.owner_message_id in selected_thinking_owner_ids
                    ),
                    continuation_required=bool(continuation_groups),
                )
                semantic = attach_thinking_history(
                    semantic,
                    groups=thinking.groups,
                    owner_key=thinking_owner_key,
                    thinking_policy=thinking.saved_policy,
                    effective_thinking_policy=thinking.effective_policy,
                )
        elif not sidecar and not thinking_sidecars:
            if any("provider_continuation" in message for message in messages):
                raise ValueError(
                    "continuation_target is required for provider continuation history"
                )
            semantic = build_console_request(
                messages,
                tools=tools or (),
                capture_mode=capture_mode,
            )
        else:
            continuation_groups = ()
            if sidecar:
                assert continuation_target is not None
                assert continuation_owner_key is not None
                selected_owner_ids = {
                    message.get(continuation_owner_key)
                    for message in messages
                    if not is_deleted_history_value(message.get("deleted"))
                    and type(message.get(continuation_owner_key)) is str
                }
                continuation_groups = provider_continuation_owner_groups(
                    tuple(
                        item
                        for item in sidecar
                        if item.owner_message_id in selected_owner_ids
                    ),
                    target=continuation_target,
                )
            selected_thinking_owner_ids = {
                message.get(thinking_owner_key)
                for message in messages
                if thinking_owner_key is not None
                and not is_deleted_history_value(message.get("deleted"))
                and type(message.get(thinking_owner_key)) is str
            }
            thinking = resolve_thinking_history(
                target=ThinkingReplayTarget(
                    provider=resolution.execution_key or resolution.provider,
                    model=resolution.model or "",
                    protocol=_thinking_protocol(resolution),
                    disposition=resolution.thinking_stream_disposition,
                    round_trip_version=resolution.thinking_round_trip_version,
                ),
                policy=thinking_policy,
                sidecars=tuple(
                    item
                    for item in thinking_sidecars
                    if item.owner_message_id in selected_thinking_owner_ids
                ),
                continuation_required=bool(continuation_groups),
            )
            continuation_owner_ids = {
                group.owner_message_id for group in continuation_groups
            }
            thinking_owner_ids = {group.owner_message_id for group in thinking.groups}
            visible_messages: list[dict[str, Any]] = []
            for message in messages:
                if is_deleted_history_value(message.get("deleted")):
                    continue
                row = dict(message)
                if (
                    continuation_owner_key is not None
                    and continuation_owner_key == thinking_owner_key
                ):
                    shared_owner_id = row.pop(continuation_owner_key, None)
                    continuation_owner_id = shared_owner_id
                    thinking_owner_id = shared_owner_id
                else:
                    continuation_owner_id = (
                        row.pop(continuation_owner_key, None)
                        if continuation_owner_key is not None
                        else None
                    )
                    thinking_owner_id = (
                        row.pop(thinking_owner_key, None)
                        if thinking_owner_key is not None
                        else None
                    )
                row.pop("provider_continuation", None)
                row.pop("deleted", None)
                if (
                    type(continuation_owner_id) is str
                    and continuation_owner_id in continuation_owner_ids
                ):
                    row[CONTINUATION_OWNER_KEY] = continuation_owner_id
                if (
                    type(thinking_owner_id) is str
                    and thinking_owner_id in thinking_owner_ids
                ):
                    row[THINKING_OWNER_KEY] = thinking_owner_id
                visible_messages.append(row)
            semantic = build_console_request(
                visible_messages,
                tools=tools or (),
                continuation_groups=continuation_groups,
                thinking_groups=thinking.groups,
                thinking_policy=thinking.saved_policy,
                effective_thinking_policy=thinking.effective_policy,
                capture_mode=capture_mode,
            )

        _validate_request_trace_binding(
            semantic,
            route=route,
            route_actor_id=route_actor_id,
            route_chain_id=route_chain_id,
            capture_mode=capture_mode,
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
        return build_httpx_async_client(
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
                logger.bind(error_type=type(exc).__name__).warning(
                    "console_provider_stale_client_close_failed"
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
            **self._resolution_settings(config, model=model),
        )

    async def resolve_for_send(
        self, selection: ConsoleProviderSelection
    ) -> ConsoleProviderResolution:
        """Resolve readiness and attach the credential-free destination."""
        resolution = await self._resolve_for_send_unclassified(selection)
        return replace(
            resolution,
            resolved_destination=resolve_console_destination(resolution),
        )

    async def _resolve_for_send_unclassified(
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
            **_thinking_stream_capability(
                identity.execution_key,
                model=model,
                reasoning_effort=selection.reasoning_effort,
            ),
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
        seed: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        reasoning_effort: str | None = None,
        thinking_budget_tokens: int | None = None,
        api_key: str | None = None,
        provider: str = "llama_cpp",
        protocol: str = "chat_completions",
        thinking_stream_disposition: ReasoningDisposition = "ignored",
        on_fallback_retry_started: "Callable[[], None] | None" = None,
        on_fallback_transition: "Callable[[bool], Awaitable[None]] | None" = None,
        on_fallback_request: "Callable[[str, Mapping[str, Any]], None] | None" = None,
        on_fallback_retry: "Callable[[dict[str, Any], str, bool], None] | None" = None,
        on_synthetic_output: "Callable[[], None] | None" = None,
        before_adapter: Callable[[], Awaitable[_ProviderAdapterAdmission]]
        | None = None,
        before_fallback_adapter: (
            Callable[[str, Mapping[str, Any]], Awaitable[_ProviderAdapterAdmission]]
            | None
        ) = None,
    ) -> AsyncIterator[ProviderStreamItem]:
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
            seed: Optional deterministic generation seed.
            presence_penalty: Optional presence penalty value.
            frequency_penalty: Optional frequency penalty value.
            reasoning_effort: Optional thinking level forwarded as
                ``chat_template_kwargs.reasoning_effort``.
            thinking_budget_tokens: Optional thinking token budget sent as
                the top-level ``reasoning_budget_tokens`` field.
            thinking_stream_disposition: Frozen adapter decision controlling
                whether start-anchored thinking is split into typed events.
            before_adapter: Required callback issuing authority for the initial
                adapter entry.
            before_fallback_adapter: Required callback issuing distinct authority
                for a stream-to-completion retry.

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
            seed=seed,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            reasoning_effort=reasoning_effort,
            thinking_budget_tokens=thinking_budget_tokens,
        )
        think_splitter = (
            StartAnchoredThinkSplitter()
            if thinking_stream_disposition == "displayable"
            else None
        )
        emitted_content = False
        received_content = False
        stream_error: httpx.HTTPError | None = None
        client = self._active_http_client()
        request_url = f"{normalized_base_url.rstrip('/')}/v1/chat/completions"
        headers = self._authorization_headers(api_key)
        if before_adapter is None:
            raise TraceCallPersistenceError()
        admission = await before_adapter()
        try:
            stream_context = self._enter_provider_adapter(
                admission,
                client.stream,
                "POST",
                request_url,
                json=payload,
                headers=headers,
            )
            async with stream_context as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    chunk = self._content_from_sse_line(line)
                    if chunk:
                        received_content = True
                        split = think_splitter.feed(chunk) if think_splitter else None
                        if split is None:
                            emitted_content = True
                            yield chunk
                            continue
                        if split.thinking:
                            yield _local_thinking_delta(
                                split.thinking,
                                provider=provider,
                                model=model,
                                protocol=protocol,
                            )
                        if split.content:
                            emitted_content = True
                            yield split.content
        except httpx.HTTPError as exc:
            if emitted_content:
                raise
            stream_error = exc

        if stream_error is None and think_splitter is not None:
            terminal = think_splitter.flush()
            if terminal.thinking:
                yield _local_thinking_delta(
                    terminal.thinking,
                    provider=provider,
                    model=model,
                    protocol=protocol,
                )
            if terminal.content:
                emitted_content = True
                yield terminal.content
            if terminal.status == "failed":
                raise ProviderThinkingCaptureError("Provider thinking capture failed.")
        if emitted_content:
            return
        if received_content:
            # Think-only reply: the filter removed every chunk, so a
            # non-streaming retry would return the same text — skip it and
            # surface any stream error that followed the content instead.
            if stream_error is not None:
                raise stream_error
            return

        if on_fallback_retry_started is not None:
            try:
                on_fallback_retry_started()
            except Exception:
                logger.warning("model_retry_capture_failed")
        if on_fallback_transition is not None:
            await on_fallback_transition(stream_error is not None)
        fallback_payload = build_llamacpp_chat_payload(
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
        fallback_endpoint = f"{normalized_base_url.rstrip('/')}/v1/chat/completions"
        if on_fallback_request is not None:
            on_fallback_request(fallback_endpoint, fallback_payload)
        if before_fallback_adapter is None:
            raise TraceCallPersistenceError()
        fallback_admission = await before_fallback_adapter(
            fallback_endpoint, fallback_payload
        )
        fallback_result = await self.complete_llamacpp_chat(
            base_url=normalized_base_url,
            model=model,
            messages=messages,
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
            api_key=api_key,
            provider=provider,
            protocol=protocol,
            thinking_stream_disposition=thinking_stream_disposition,
            include_thinking_events=True,
            adapter_admission=fallback_admission,
        )
        fallback_items, fallback_capture_failed = _unpack_local_completion_result(
            fallback_result
        )
        fallback = "".join(item for item in fallback_items if isinstance(item, str))
        # task-19324: this retry is a SECOND HTTP request to the server. It
        # is made below the Console capture seam (which wraps stream_chat's
        # one call), so without this hook a turn that really made two calls
        # showed only one in the Inspector -- understating what was sent, on
        # exactly the degraded turn a user opens the Inspector to inspect.
        if on_fallback_retry is not None:
            try:
                on_fallback_retry(
                    fallback_payload,
                    fallback or "",
                    fallback_capture_failed,
                )
            except Exception as exc:
                # Capture must never break a send (task-18300 contract) -- but
                # a constant message made the degraded path this exists to
                # EXPLAIN undiagnosable (Qodo #6). The type name is enough to
                # act on and, unlike a traceback, cannot carry payload from
                # the frame's locals.
                logger.warning(
                    f"exchange_capture_fallback_failed: {type(exc).__name__}"
                )
        if fallback_items:
            for item in fallback_items:
                yield item
        if fallback_capture_failed:
            raise ProviderThinkingCaptureError("Provider thinking capture failed.")
        if fallback_items:
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
        provider: str = "llama_cpp",
        protocol: str = "chat_completions",
        thinking_stream_disposition: ReasoningDisposition = "ignored",
        include_thinking_events: bool = False,
        before_dispatch: "Callable[[str, Mapping[str, Any]], None] | None" = None,
        before_adapter: Callable[[], Awaitable[_ProviderAdapterAdmission]]
        | None = None,
        adapter_admission: _ProviderAdapterAdmission | None = None,
    ) -> str | _LocalCompletionResult:
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
            thinking_stream_disposition: Frozen adapter decision controlling
                whether start-anchored thinking is split into typed events.
            before_adapter: Callback issuing adapter-entry authority when
                ``adapter_admission`` is not supplied.
            adapter_admission: Explicit issuer-bound authority for this adapter
                entry.

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
        headers = self._authorization_headers(api_key)
        sensitive_request = is_sensitive_llm_request()
        if before_dispatch is not None:
            before_dispatch(request_url, payload)
        admission = adapter_admission
        if admission is None and before_adapter is not None:
            admission = await before_adapter()
        if admission is None:
            raise TraceCallPersistenceError()
        request_call = (
            self._post_without_high_level_http_log if sensitive_request else client.post
        )
        request_kwargs = (
            {"json_payload": payload, "headers": headers}
            if sensitive_request
            else {"json": payload, "headers": headers}
        )
        if sensitive_request:
            response = await self._enter_provider_adapter(
                admission,
                request_call,
                client,
                request_url,
                **request_kwargs,
            )
        else:
            response = await self._enter_provider_adapter(
                admission,
                request_call,
                request_url,
                **request_kwargs,
            )
        response.raise_for_status()
        content = self._content_from_completion_response(response)
        if content is None and strict_response:
            raise ChatProviderError(
                "Provider returned an unsupported auxiliary response.",
                provider="llama_cpp",
            )
        result = (
            _split_local_completion_items(
                content or "",
                provider=provider,
                model=model,
                protocol=protocol,
            )
            if thinking_stream_disposition == "displayable"
            else _LocalCompletionResult(items=(content,) if content else ())
        )
        if include_thinking_events:
            return result
        if result.capture_failed:
            return ""
        return "".join(item for item in result.items if isinstance(item, str))

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
        *,
        route: ConsoleRequestRoute | None = None,
    ) -> AuxiliaryCompletionResult:
        """Run exactly one sensitive, non-streaming completion.

        The captured resolution is the sole provider authority. This path
        deliberately bypasses normal Console history, tools, streaming
        normalization, fallback copy, and persistence.
        """

        if not isinstance(request, AuxiliaryCompletionRequest):
            raise TypeError("request must be an AuxiliaryCompletionRequest")
        if route not in {None, ConsoleRequestRoute.AUTO_COMPACTION}:
            raise TraceProvenanceAlignmentError(
                "auxiliary completion route is not capture-off"
            )
        if route is not None:
            request_route_provenance(route)
        admission = self._capture_off_admission(route)
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
                        thinking_stream_disposition=(
                            resolution.thinking_stream_disposition
                        ),
                        include_thinking_events=True,
                        adapter_admission=admission,
                    )
                else:
                    kwargs = self._auxiliary_chat_api_kwargs(request, resolution)
                    context = copy_context()
                    response = await asyncio.to_thread(
                        context.run,
                        self._complete_sensitive_sync,
                        kwargs,
                        admission,
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

        if not isinstance(text, (str, _LocalCompletionResult)):
            raise ChatProviderError(
                "Provider returned an unsupported auxiliary response.",
                provider=provider,
            )
        try:
            text = self._normalize_auxiliary_thinking(text, resolution)
        except ProviderThinkingCaptureError as exc:
            raise ChatProviderError(
                safe_provider_error_copy(provider, exc),
                provider=provider,
                status_code=502,
            ) from None
        return AuxiliaryCompletionResult(
            provider=provider,
            model=model,
            text=text,
            usage=usage,
        )

    @staticmethod
    def _normalize_auxiliary_thinking(
        text: str | _LocalCompletionResult,
        resolution: ConsoleProviderResolution,
    ) -> str:
        """Return assistant-visible text under the frozen adapter disposition."""

        if isinstance(text, _LocalCompletionResult):
            result = text
        elif resolution.thinking_stream_disposition == "displayable":
            result = _split_local_completion_items(
                text,
                provider=resolution.provider,
                model=cast(str, resolution.model),
                protocol=_thinking_protocol(resolution),
            )
        else:
            return text
        if result.capture_failed:
            raise ProviderThinkingCaptureError("Provider thinking capture failed.")
        return "".join(item for item in result.items if isinstance(item, str))

    def _complete_sensitive_sync(
        self,
        kwargs: Mapping[str, Any],
        admission: _ProviderAdapterAdmission,
    ) -> Any:
        """Invoke the final synchronous adapter under the sensitive policy."""

        with sensitive_llm_request():
            return self._enter_provider_adapter(
                admission,
                self._chat_api_call,
                **dict(kwargs),
            )

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
        *,
        route: ConsoleRequestRoute | None = None,
        route_actor_id: str | None = None,
        route_chain_id: str | None = None,
        capture_mode: ConsoleTraceCaptureMode = ConsoleTraceCaptureMode.CAPTURE_OFF,
        ephemeral: bool = False,
        before_provider_dispatch: Callable[[], Awaitable[None]] | None = None,
    ) -> AsyncIterator[ProviderStreamItem]:
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
        require_durable_capture_admission(
            capture_mode=capture_mode,
            ephemeral=ephemeral,
        )
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
            else ConsoleProviderStreamSignals().new_usage_call()
            if capture_mode is ConsoleTraceCaptureMode.CAPTURE_ON
            else None
        )
        # Tracks whether the generator drained a provider call normally, vs.
        # being torn down early (consumer Stop/cancel -> GeneratorExit /
        # CancelledError thrown into a suspended `yield`). Read only in the
        # `finally` below to pick the exchange's terminal status -- an
        # in-flight worker error already closed its own exchange as "error"
        # before enqueueing (token-pop move semantics make the second close
        # here a no-op), so this flag only decides "complete" vs "stopped".
        completed = False
        provider_failed = False
        trace_call_boundary: object | None = None
        response_accumulator = _TraceResponseAccumulator()

        def observe_response(
            item: ProviderStreamItem, *, synthetic: bool = False
        ) -> None:
            if response_accumulator.observe(item, synthetic=synthetic):
                _mark_trace_response_started(trace_call_boundary)

        try:
            if not resolution.ready or not resolution.model:
                return
            prepared = (
                messages
                if isinstance(messages, PreparedProviderRequest)
                else self.prepare_chat_request(
                    resolution,
                    messages,
                    tools=tools,
                    route=route,
                    route_actor_id=route_actor_id,
                    route_chain_id=route_chain_id,
                    capture_mode=capture_mode,
                )
            )
            if isinstance(messages, PreparedProviderRequest) and tools is not None:
                raise ValueError("tools are already owned by PreparedProviderRequest")
            _validate_request_trace_binding(
                prepared,
                route=route,
                route_actor_id=route_actor_id,
                route_chain_id=route_chain_id,
                capture_mode=capture_mode,
            )
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
            capture_off_admission: _ProviderAdapterAdmission | None = None
            if capture_mode is ConsoleTraceCaptureMode.CAPTURE_ON:
                trace_call_boundary = self._reserve_trace_call(
                    prepared,
                    effective_resolution,
                    route,
                )
            else:
                capture_off_admission = self._capture_off_admission(route)
            if resolution.provider in {"llama_cpp", "local_llamacpp"}:
                wire_messages = [thaw_json(item) for item in prepared.messages]

                def capture_wire_payload(
                    raw_wire: Mapping[str, Any], detail: CaptureDetail
                ) -> tuple[Any, tuple[str, ...]]:
                    """Returns the sanitized wire capture plus the Safe
                    history-elision inventory (task-23026) — the same
                    O(n²)-copy bound the generic path gets from
                    ``build_request_capture``, applied to this branch's
                    literal ``messages`` list."""
                    captured = deepcopy(raw_wire)
                    if detail is not CaptureDetail.SAFE:
                        sanitized, omitted = sanitize_capture_value_with_omission(
                            captured,
                            known_credentials=(resolution.api_key,)
                            if resolution.api_key
                            else (),
                        )
                        return sanitized, (("wire_payload",) if omitted else ())
                    captured_messages = captured.get("messages")
                    if not isinstance(captured_messages, list):
                        sanitized, omitted = sanitize_capture_value_with_omission(
                            captured,
                            known_credentials=(resolution.api_key,)
                            if resolution.api_key
                            else (),
                        )
                        return sanitized, (("wire_payload",) if omitted else ())
                    semantic_messages = [
                        thaw_json(item)
                        for item in prepared.semantic.flattened_messages()
                    ]
                    if prepared.wire_style == "single_preamble":
                        system_parts: list[str] = []
                        for row in semantic_messages:
                            if row.get("role") != "system":
                                break
                            content = str(row.get("content") or "").strip()
                            if row.get(EPHEMERAL_ORIGIN_KEY) == "project_instructions":
                                content = (
                                    "[project instruction body omitted by "
                                    f"capture policy -- {len(content)} chars]"
                                )
                            if content:
                                system_parts.append(content)
                        if captured_messages and system_parts:
                            captured_messages[0]["content"] = "\n\n".join(system_parts)
                    else:
                        for index, source in enumerate(semantic_messages):
                            if (
                                index < len(captured_messages)
                                and source.get(EPHEMERAL_ORIGIN_KEY)
                                == "project_instructions"
                            ):
                                content = str(source.get("content") or "")
                                captured_messages[index]["content"] = (
                                    "[project instruction body omitted by "
                                    f"capture policy -- {len(content)} chars]"
                                )
                    sanitized, credential_omitted = (
                        sanitize_capture_value_with_omission(
                            captured,
                            known_credentials=(resolution.api_key,)
                            if resolution.api_key
                            else (),
                        )
                    )
                    if credential_omitted:
                        return sanitized, ("wire_payload",)
                    if isinstance(sanitized, dict):
                        compacted_rows, elided_paths = compact_safe_history_rows(
                            sanitized.get("messages"),
                            detail,
                            path="wire_payload.messages",
                        )
                        if elided_paths:
                            sanitized["messages"] = compacted_rows
                        return sanitized, elided_paths
                    return sanitized, ()

                verified_wire: dict[str, Any] | None = None
                verified_bundle: ProviderRequestShadowBundle | None = None
                if capture_mode is ConsoleTraceCaptureMode.CAPTURE_ON:
                    verified_wire = build_llamacpp_chat_payload(
                        model=resolution.model,
                        messages=wire_messages,
                        stream=resolution.streaming,
                        temperature=resolution.temperature,
                        top_p=resolution.top_p,
                        min_p=resolution.min_p,
                        top_k=resolution.top_k,
                        max_tokens=effective_resolution.max_tokens,
                        seed=resolution.seed,
                        presence_penalty=resolution.presence_penalty,
                        frequency_penalty=resolution.frequency_penalty,
                        reasoning_effort=resolution.reasoning_effort,
                        thinking_budget_tokens=resolution.thinking_budget_tokens,
                    )
                    trace_kwargs = self._trace_surface_kwargs(
                        trace_call_boundary,
                        self._chat_api_kwargs_from_prepared(
                            effective_resolution, prepared
                        ),
                    )
                    verified_bundle = self._verify_trace_shadow(
                        effective_resolution,
                        prepared,
                        trace_kwargs,
                        capture_mode=capture_mode,
                        literal_payload=verified_wire,
                        endpoint_identity=(
                            f"{normalize_llamacpp_base_url(resolution.base_url).rstrip('/')}"
                            "/v1/chat/completions"
                        ),
                        trace_call_boundary=trace_call_boundary,
                    )

                async def commit_llama_dispatch() -> _ProviderAdapterAdmission:
                    admission = self._trace_dispatch_admission(
                        trace_call_boundary,
                        verified_bundle,
                        prepared.provenance,
                        route=route,
                        capture_off_admission=capture_off_admission,
                    )
                    if before_provider_dispatch is not None:
                        try:
                            await before_provider_dispatch()
                        except BaseException:
                            self._commit_trace_dispatch_unknown(trace_call_boundary)
                            raise
                    return admission

                # This branch builds its own HTTP body -- the one place
                # capture IS the literal wire payload (spec Non-goals).
                # `api_key` never enters `build_llamacpp_chat_payload`'s
                # signature, so it structurally cannot leak into the
                # captured request even though it rides `stream_llamacpp_
                # chat`/`complete_llamacpp_chat`'s kwargs as auth headers.
                if call_signals is not None and call_signals.exchange_capture_enabled:
                    try:
                        budget = CaptureBudget()
                        wire = verified_wire or build_llamacpp_chat_payload(
                            model=resolution.model,
                            messages=wire_messages,
                            stream=resolution.streaming,
                            temperature=resolution.temperature,
                            top_p=resolution.top_p,
                            min_p=resolution.min_p,
                            top_k=resolution.top_k,
                            max_tokens=effective_resolution.max_tokens,
                            seed=resolution.seed,
                            presence_penalty=resolution.presence_penalty,
                            frequency_penalty=resolution.frequency_penalty,
                            reasoning_effort=resolution.reasoning_effort,
                            thinking_budget_tokens=resolution.thinking_budget_tokens,
                        )
                        capture_request, omitted = build_request_capture(
                            {"model": resolution.model},
                            capture_detail=call_signals.capture_detail,
                            budget=budget,
                            known_credentials=(resolution.api_key,)
                            if resolution.api_key
                            else (),
                        )
                        sanitized_wire, wire_elided = capture_wire_payload(
                            wire, call_signals.capture_detail
                        )
                        capture_request["wire_payload"] = (
                            sanitized_wire
                            if budget.retain(sanitized_wire)
                            else {"truncated": True}
                        )
                        omitted = tuple(sorted(set(omitted).union(wire_elided)))
                        call_signals.begin_exchange(
                            provider=str(resolution.provider or ""),
                            model=str(resolution.model or ""),
                            endpoint=normalize_llamacpp_base_url(resolution.base_url),
                            request=capture_request,
                            omitted_keys=omitted,
                            capture_budget=budget,
                            known_credentials=(resolution.api_key,)
                            if resolution.api_key
                            else (),
                            request_credentials_filtered=True,
                        )
                    except Exception as exc:
                        logger.warning(
                            "exchange_capture_begin_failed: {}", type(exc).__name__
                        )
                if not resolution.streaming:
                    # M1: an HTTP failure here must close the exchange as
                    # "error" -- left to the outer `finally` below, it would
                    # see `completed` still False and close as "stopped"
                    # (a real send failure misreported as a user-initiated
                    # stop), unlike the generic path's own explicit
                    # close_exchange(status="error") before it re-raises.
                    try:
                        completion_result = await self.complete_llamacpp_chat(
                            base_url=resolution.base_url,
                            model=resolution.model,
                            messages=wire_messages,
                            temperature=resolution.temperature,
                            top_p=resolution.top_p,
                            min_p=resolution.min_p,
                            top_k=resolution.top_k,
                            max_tokens=effective_resolution.max_tokens,
                            seed=resolution.seed,
                            presence_penalty=resolution.presence_penalty,
                            frequency_penalty=resolution.frequency_penalty,
                            reasoning_effort=resolution.reasoning_effort,
                            thinking_budget_tokens=resolution.thinking_budget_tokens,
                            api_key=resolution.api_key,
                            provider=resolution.execution_key or resolution.provider,
                            protocol=_thinking_protocol(resolution),
                            thinking_stream_disposition=(
                                resolution.thinking_stream_disposition
                            ),
                            include_thinking_events=True,
                            before_adapter=commit_llama_dispatch,
                        )
                    except Exception:
                        if call_signals is not None:
                            call_signals.close_exchange(status="error")
                        raise
                    completion_items, completion_capture_failed = (
                        _unpack_local_completion_result(completion_result)
                    )
                    for item in completion_items:
                        if call_signals is not None and isinstance(item, str):
                            call_signals.record_exchange_content(item)
                        observe_response(item)
                        yield item
                    if completion_capture_failed:
                        if call_signals is not None:
                            call_signals.close_exchange(status="error")
                        raise ProviderThinkingCaptureError(
                            "Provider thinking capture failed."
                        )
                    completed = True
                    return

                def _capture_llamacpp_fallback(
                    wire_payload: dict[str, Any], text: str, capture_failed: bool
                ) -> None:
                    """Give the stream->complete retry its own capture (task-19324).

                    The retry is a second HTTP request issued *inside*
                    ``stream_llamacpp_chat``, below the seam that captures
                    ``stream_chat``'s own call. It gets a fresh call-scoped
                    signals view off the aggregate so it lands as its own
                    row rather than being folded into the streaming call it
                    replaced. Needs the aggregate: a caller that handed us
                    an already-scoped view has no second call to open.
                    """
                    fallback_provenance = request_route_provenance(
                        ConsoleRequestRoute.LLAMA_FALLBACK
                    )
                    logger.bind(
                        route=fallback_provenance.route.value,
                        predicate=fallback_provenance.predicate,
                    ).debug("console_llama_fallback_request")
                    if signals is None or isinstance(
                        signals, ConsoleProviderCallSignals
                    ):
                        return
                    retry_signals = signals.new_usage_call()
                    budget = CaptureBudget()
                    capture_request, omitted = build_request_capture(
                        {"model": resolution.model},
                        capture_detail=retry_signals.capture_detail,
                        budget=budget,
                        known_credentials=(resolution.api_key,)
                        if resolution.api_key
                        else (),
                    )
                    sanitized_wire, wire_elided = capture_wire_payload(
                        wire_payload, retry_signals.capture_detail
                    )
                    capture_request["wire_payload"] = (
                        sanitized_wire
                        if budget.retain(sanitized_wire)
                        else {"truncated": True}
                    )
                    omitted = tuple(sorted(set(omitted).union(wire_elided)))
                    capture_request["retry_of"] = (
                        "llama.cpp stream produced no content; retried non-streaming"
                    )
                    retry_signals.begin_exchange(
                        provider=str(resolution.provider or ""),
                        model=str(resolution.model or ""),
                        endpoint=normalize_llamacpp_base_url(resolution.base_url),
                        request=capture_request,
                        omitted_keys=omitted,
                        capture_budget=budget,
                        known_credentials=(resolution.api_key,)
                        if resolution.api_key
                        else (),
                        request_credentials_filtered=True,
                    )
                    if text:
                        retry_signals.record_exchange_content(text)
                    retry_signals.close_exchange(
                        status="error" if capture_failed else "complete"
                    )
                    # Qodo #4: `new_usage_call()` registers this call in the
                    # aggregate's `_active_usage_payloads`; without the
                    # matching close it stays there forever. Harmless while
                    # the retry records no usage, but the moment one is added
                    # the stuck entry is billed by `usage_payloads()`'s
                    # in-flight tail. Closing here keeps the pairing local and
                    # obvious instead of load-bearing on a future reader.
                    retry_signals.close_usage_call()

                async def _authorize_llamacpp_fallback(
                    endpoint: str,
                    wire_payload: Mapping[str, Any],
                ) -> _ProviderAdapterAdmission:
                    nonlocal trace_call_boundary
                    if capture_mode is ConsoleTraceCaptureMode.CAPTURE_OFF:
                        return self._capture_off_admission(
                            ConsoleRequestRoute.LLAMA_FALLBACK
                        )
                    fallback_resolution = replace(
                        effective_resolution,
                        streaming=False,
                    )
                    fallback_boundary = self._reserve_trace_call(
                        prepared,
                        fallback_resolution,
                        ConsoleRequestRoute.LLAMA_FALLBACK,
                    )
                    trace_call_boundary = fallback_boundary
                    fallback_provenance = self._provenance_for_route(
                        prepared.provenance,
                        ConsoleRequestRoute.LLAMA_FALLBACK,
                    )
                    fallback_kwargs = self._trace_surface_kwargs(
                        fallback_boundary,
                        self._chat_api_kwargs_from_prepared(
                            fallback_resolution, prepared
                        ),
                    )
                    fallback_bundle = self._verify_trace_shadow(
                        fallback_resolution,
                        prepared,
                        fallback_kwargs,
                        capture_mode=capture_mode,
                        literal_payload=wire_payload,
                        endpoint_identity=endpoint,
                        route=ConsoleRequestRoute.LLAMA_FALLBACK,
                        extra_overlays=(
                            ProviderOverlayProvenance(
                                "llama_fallback_retry", "structural"
                            ),
                        ),
                        trace_call_boundary=fallback_boundary,
                        provenance_override=fallback_provenance,
                    )
                    return self._trace_dispatch_admission(
                        fallback_boundary,
                        fallback_bundle,
                        fallback_provenance,
                        route=ConsoleRequestRoute.LLAMA_FALLBACK,
                    )

                async def _transition_to_llamacpp_fallback(
                    _stream_failed: bool,
                ) -> None:
                    nonlocal trace_call_boundary
                    initial_boundary = trace_call_boundary
                    trace_call_boundary = None
                    await _settle_trace_response(
                        initial_boundary,
                        (),
                        outcome=TraceCallState.ERROR,
                        usage=None,
                        signals=call_signals,
                    )

                try:
                    async for chunk in self.stream_llamacpp_chat(
                        base_url=resolution.base_url,
                        model=resolution.model,
                        messages=wire_messages,
                        temperature=resolution.temperature,
                        top_p=resolution.top_p,
                        min_p=resolution.min_p,
                        top_k=resolution.top_k,
                        max_tokens=effective_resolution.max_tokens,
                        seed=resolution.seed,
                        presence_penalty=resolution.presence_penalty,
                        frequency_penalty=resolution.frequency_penalty,
                        reasoning_effort=resolution.reasoning_effort,
                        thinking_budget_tokens=resolution.thinking_budget_tokens,
                        api_key=resolution.api_key,
                        provider=resolution.execution_key or resolution.provider,
                        protocol=_thinking_protocol(resolution),
                        thinking_stream_disposition=(
                            resolution.thinking_stream_disposition
                        ),
                        on_fallback_retry_started=(
                            signals.mark_model_retry
                            if isinstance(signals, ConsoleProviderStreamSignals)
                            else None
                        ),
                        on_fallback_transition=_transition_to_llamacpp_fallback,
                        on_fallback_request=None,
                        before_fallback_adapter=_authorize_llamacpp_fallback,
                        on_fallback_retry=_capture_llamacpp_fallback,
                        on_synthetic_output=(
                            call_signals.mark_synthetic_fallback
                            if call_signals is not None
                            else None
                        ),
                        before_adapter=commit_llama_dispatch,
                    ):
                        synthetic = (
                            call_signals.take_synthetic_pending()
                            if call_signals is not None
                            else False
                        )
                        if call_signals is not None and isinstance(chunk, str):
                            call_signals.record_exchange_content(
                                chunk, synthetic=synthetic
                            )
                        observe_response(
                            chunk,
                            synthetic=synthetic,
                        )
                        yield chunk
                except Exception:
                    # Only real provider/HTTP failures land here -- a
                    # consumer abort throws GeneratorExit/CancelledError
                    # (BaseException, not Exception) into this suspended
                    # `yield`, so it still falls through to the outer
                    # `finally`'s "stopped" close, unchanged.
                    if call_signals is not None:
                        call_signals.close_exchange(status="error")
                    raise
                completed = True
                return
            if resolution.execution_key:
                async for emission in self._stream_generic_chat(
                    effective_resolution,
                    prepared,
                    signals=call_signals,
                    capture_mode=capture_mode,
                    trace_call_boundary=trace_call_boundary,
                    capture_off_admission=capture_off_admission,
                    route=route,
                    before_provider_dispatch=before_provider_dispatch,
                ):
                    observe_response(emission.item, synthetic=emission.synthetic)
                    yield emission.item
                completed = True
                return
        except Exception:
            provider_failed = True
            await _settle_trace_response(
                trace_call_boundary,
                response_accumulator.items,
                outcome=TraceCallState.ERROR,
                usage=(
                    call_signals.usage_snapshot()
                    if isinstance(call_signals, ConsoleProviderCallSignals)
                    else None
                ),
                response_omission=response_accumulator.omission_reason,
                signals=call_signals,
            )
            raise
        finally:
            if not provider_failed:
                await _settle_trace_response(
                    trace_call_boundary,
                    response_accumulator.items,
                    outcome=(
                        TraceCallState.ERROR
                        if completed and not response_accumulator.semantic_observed
                        else TraceCallState.COMPLETE
                        if completed
                        else TraceCallState.STOPPED
                    ),
                    usage=(
                        call_signals.usage_snapshot()
                        if isinstance(call_signals, ConsoleProviderCallSignals)
                        else None
                    ),
                    response_omission=response_accumulator.omission_reason,
                    signals=call_signals,
                )
            if call_signals is not None:
                call_signals.close_exchange(
                    status="complete" if completed else "stopped"
                )
                call_signals.close_usage_call()

    async def _stream_generic_chat(
        self,
        resolution: ConsoleProviderResolution,
        request: PreparedProviderRequest,
        signals: ConsoleProviderCallSignals | None = None,
        capture_mode: ConsoleTraceCaptureMode = ConsoleTraceCaptureMode.CAPTURE_OFF,
        trace_call_boundary: object | None = None,
        capture_off_admission: _ProviderAdapterAdmission | None = None,
        route: ConsoleRequestRoute | None = None,
        before_provider_dispatch: Callable[[], Awaitable[None]] | None = None,
    ) -> AsyncIterator[_ProviderStreamEmission]:
        """Bridge synchronous chat_api_call responses into async Console chunks."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        stop_event = threading.Event()
        adapter_entry_gate = _ProviderAdapterEntryGate()
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
                kwargs = self._trace_surface_kwargs(trace_call_boundary, kwargs)
                bundle = self._verify_trace_shadow(
                    resolution,
                    request,
                    kwargs,
                    capture_mode=capture_mode,
                    trace_call_boundary=trace_call_boundary,
                )
                if signals is not None and signals.exchange_capture_enabled:
                    try:
                        budget = CaptureBudget()
                        capture_kwargs = dict(kwargs)
                        semantic_messages = [
                            thaw_json(item)
                            for item in request.semantic.flattened_messages()
                        ]
                        has_project_instructions = any(
                            row.get(EPHEMERAL_ORIGIN_KEY) == "project_instructions"
                            for row in semantic_messages
                        )
                        if (
                            has_project_instructions
                            and signals.capture_detail is CaptureDetail.SAFE
                        ):
                            capture_kwargs["messages_payload"] = semantic_messages
                        if (
                            has_project_instructions
                            and signals.capture_detail is CaptureDetail.SAFE
                        ):
                            system_parts: list[str] = []
                            for row in semantic_messages:
                                if row.get("role") != "system":
                                    break
                                content = str(row.get("content") or "").strip()
                                if (
                                    row.get(EPHEMERAL_ORIGIN_KEY)
                                    == "project_instructions"
                                ):
                                    content = (
                                        "[project instruction body omitted by "
                                        f"capture policy -- {len(content)} chars]"
                                    )
                                if content:
                                    system_parts.append(content)
                            capture_kwargs["system_message"] = (
                                "\n\n".join(system_parts) or None
                            )
                        capture_request, omitted = build_request_capture(
                            capture_kwargs,
                            capture_detail=signals.capture_detail,
                            budget=budget,
                            known_credentials=(resolution.api_key,)
                            if resolution.api_key
                            else (),
                        )
                        signals.begin_exchange(
                            provider=str(resolution.provider or ""),
                            model=str(resolution.model or ""),
                            endpoint=getattr(resolution, "base_url", None),
                            request=capture_request,
                            omitted_keys=omitted,
                            capture_budget=budget,
                            known_credentials=(resolution.api_key,)
                            if resolution.api_key
                            else (),
                            request_credentials_filtered=True,
                        )
                    except Exception as exc:
                        logger.warning(
                            "exchange_capture_begin_failed: {}", type(exc).__name__
                        )
                admission = self._trace_dispatch_admission(
                    trace_call_boundary,
                    bundle,
                    request.provenance,
                    route=route,
                    capture_off_admission=capture_off_admission,
                )
                if before_provider_dispatch is not None:

                    async def await_provider_dispatch() -> None:
                        await before_provider_dispatch()

                    dispatch_future = asyncio.run_coroutine_threadsafe(
                        await_provider_dispatch(), loop
                    )
                    try:
                        dispatch_future.result()
                    except BaseException:
                        self._commit_trace_dispatch_unknown(trace_call_boundary)
                        raise
                if stop_event.is_set():
                    # Normalized dispatch is already durable before the
                    # caller-owned checkpoint.  Cancellation after that wait
                    # prevents adapter entry, but the committed boundary can
                    # only be closed as dispatch-unknown without rewriting
                    # history or widening the persisted lifecycle schema.
                    self._commit_trace_dispatch_unknown(trace_call_boundary)
                    return
                try:
                    response = self._enter_provider_adapter(
                        admission,
                        self._chat_api_call,
                        _console_adapter_entry_gate=adapter_entry_gate,
                        **kwargs,
                    )
                except _ProviderAdapterEntryCancelled:
                    self._commit_trace_dispatch_unknown(trace_call_boundary)
                    return
                provider_response = response
                accumulator = _ToolCallAccumulator() if request.tools else None
                if accumulator is not None:
                    response = _tee_tool_calls(response, accumulator)
                if not retain_response(response) or stop_event.is_set():
                    return
                emitted_content = False
                think_splitter = (
                    StartAnchoredThinkSplitter()
                    if resolution.thinking_stream_disposition == "displayable"
                    else None
                )
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
                    split = think_splitter.feed(text) if think_splitter else None
                    thinking = split.thinking if split is not None else ""
                    visible = split.content if split is not None else text
                    if thinking:
                        enqueue(
                            _QueueItem.thinking(
                                _local_thinking_delta(
                                    thinking,
                                    provider=resolution.execution_key
                                    or resolution.provider,
                                    model=cast(str, resolution.model),
                                    protocol=_thinking_protocol(resolution),
                                )
                            )
                        )
                    if visible:
                        emitted_content = True
                    synthetic = (
                        signals.take_synthetic_pending()
                        if signals is not None and visible
                        else False
                    )
                    if signals is not None and visible:
                        # M3: the fallback UI copy this loop can receive
                        # from `normalize_provider_response` (NO_PROVIDER_
                        # CONTENT_COPY / UNSUPPORTED_PROVIDER_RESPONSE_COPY)
                        # is locally synthesized, never provider output --
                        # take_synthetic_pending() reports whether THIS
                        # specific chunk was one (set by mark_synthetic_
                        # fallback() just before that generator's yield),
                        # so the capture records it as such instead of
                        # presenting UI copy as a model answer.
                        signals.record_exchange_content(visible, synthetic=synthetic)
                    if visible:
                        enqueue(_QueueItem.content(visible, synthetic=synthetic))
                if stop_event.is_set():
                    return
                if think_splitter is not None:
                    terminal = think_splitter.flush()
                    if terminal.thinking:
                        enqueue(
                            _QueueItem.thinking(
                                _local_thinking_delta(
                                    terminal.thinking,
                                    provider=resolution.execution_key
                                    or resolution.provider,
                                    model=cast(str, resolution.model),
                                    protocol=_thinking_protocol(resolution),
                                )
                            )
                        )
                    if terminal.content:
                        emitted_content = True
                        if signals is not None:
                            signals.record_exchange_content(terminal.content)
                        enqueue(_QueueItem.content(terminal.content))
                    if terminal.status == "failed":
                        raise ProviderThinkingCaptureError(
                            "Provider thinking capture failed."
                        )
                proprietary_evidence = _proprietary_thinking_event(
                    provider_response, resolution
                )
                if proprietary_evidence is not None:
                    enqueue(_QueueItem.thinking(proprietary_evidence))
                if accumulator is not None:
                    calls = accumulator.calls()
                    if signals is not None and calls:
                        signals.record_exchange_tool_calls(calls)
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
                        if signals is not None:
                            signals.close_exchange(status="error")
                        try:
                            raw_error_copy = self._safe_error_copy(
                                resolution.provider, no_content_exc
                            )
                        except BaseException:  # failure context can contain credentials
                            raw_error_copy = _PROVIDER_REQUEST_FAILED_COPY
                        enqueue(
                            _QueueItem.error(
                                _sanitized_provider_diagnostic(
                                    raw_error_copy,
                                    known_credentials=(resolution.api_key or "",),
                                ),
                                status_code=no_content_exc.status_code,
                            )
                        )
            except TraceCallPersistenceError as exc:
                enqueue(_QueueItem.trace_persistence_error(exc.boundary))
            except TraceProvenanceAlignmentError:
                if signals is not None:
                    signals.close_exchange(status="error")
                enqueue(_QueueItem.trace_verification_error())
            except BaseException as exc:
                raw_status = getattr(exc, "status_code", None)
                status_code = raw_status if type(raw_status) is int else None
                try:
                    raw_error_copy = self._safe_error_copy(resolution.provider, exc)
                except BaseException:  # failure context can contain credentials
                    raw_error_copy = _PROVIDER_REQUEST_FAILED_COPY
                error_copy = _provider_error_copy_with_model_recovery(
                    raw_error_copy,
                    model=resolution.model,
                    status_code=status_code,
                )
                error_copy = _sanitized_provider_diagnostic(
                    error_copy,
                    known_credentials=(resolution.api_key or "",),
                )
                if signals is not None:
                    signals.close_exchange(status="error")
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
                if item.kind == "trace_verification_error":
                    raise TraceProvenanceAlignmentError(
                        "provider request trace verification failed"
                    )
                if item.kind == "trace_persistence_error":
                    raise TraceCallPersistenceError(boundary=item.payload)
                if item.kind == "tool_calls":
                    yield _ProviderStreamEmission(cast(ProviderToolCalls, item.payload))
                    continue
                if item.kind == "thinking":
                    yield _ProviderStreamEmission(
                        cast(
                            ProviderThinkingDelta | ProviderProprietaryThinkingEvidence,
                            item.payload,
                        )
                    )
                    continue
                if item.text:
                    yield _ProviderStreamEmission(item.text, synthetic=item.synthetic)
        finally:
            adapter_entry_gate.cancel()
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

    def _verify_trace_shadow(
        self,
        resolution: ConsoleProviderResolution,
        request: PreparedProviderRequest,
        actual_kwargs: Mapping[str, object],
        *,
        capture_mode: ConsoleTraceCaptureMode,
        literal_payload: object | None = None,
        endpoint_identity: str | None = None,
        route: ConsoleRequestRoute | None = None,
        extra_overlays: tuple[ProviderOverlayProvenance, ...] = (),
        trace_call_boundary: object | None = None,
        provenance_override: ProviderRequestProvenance | None = None,
    ) -> ProviderRequestShadowBundle | None:
        """Fail Capture On closed before any content-bearing shadow sink."""

        if capture_mode is ConsoleTraceCaptureMode.CAPTURE_OFF:
            return None
        if route is not None:
            request_route_provenance(route)
        provenance = provenance_override or request.provenance
        if provenance is None:
            raise TraceProvenanceAlignmentError(
                "provider request trace verification failed"
            )
        surface_boundary = getattr(trace_call_boundary, "surface_boundary", None)
        projected = getattr(surface_boundary, "provenance", None)
        if projected is not None:
            if not isinstance(projected, ProviderRequestProvenance):
                raise TraceProvenanceAlignmentError(
                    "provider request trace verification failed"
                )
            provenance = projected

        from tldw_chatbook.Chat.Chat_Functions import PROVIDER_PARAM_MAP

        handler_source_names = {
            provider_name: generic_name
            for generic_name, provider_name in PROVIDER_PARAM_MAP.get(
                resolution.execution_key or "", {}
            ).items()
        }
        handler_source_names.update(
            api_base_url="api_base_url",
            provider_name="api_endpoint",
        )

        def project(values: dict[str, object]) -> Mapping[str, object]:
            from tldw_chatbook.Chat.Chat_Functions import (
                project_chat_handler_kwargs,
            )

            endpoint = values.pop("api_endpoint", None)
            if not isinstance(endpoint, str):
                raise ValueError("missing endpoint")
            return project_chat_handler_kwargs(endpoint, values)

        expected = self._trace_surface_kwargs(
            trace_call_boundary,
            reconstruct_provider_gateway_kwargs(resolution, request),
        )
        known_credentials = (
            (resolution.api_key,)
            if isinstance(resolution.api_key, str) and resolution.api_key
            else ()
        )
        bundle = verify_provider_request_shadow(
            actual_kwargs=actual_kwargs,
            expected_kwargs=expected,
            provenance=provenance,
            project_handler_kwargs=project,
            handler_source_names=handler_source_names,
            known_credentials=known_credentials,
            literal_payload=literal_payload,
            endpoint_identity=endpoint_identity or resolution.base_url or None,
            extra_overlays=extra_overlays,
            preparation_identity=(
                getattr(trace_call_boundary, "preparation_identity", None)
                if trace_call_boundary is not None
                else None
            ),
            surface_boundary=surface_boundary,
        )
        if not bundle.available and trace_call_boundary is None:
            raise TraceProvenanceAlignmentError(
                "provider request trace verification failed"
            )
        if self._trace_shadow_sink is not None:
            try:
                self._trace_shadow_sink(bundle)
            except Exception:  # noqa: BLE001 - shadow sink context may be sensitive
                raise TraceProvenanceAlignmentError(
                    "provider request trace verification failed"
                ) from None
        return bundle

    @staticmethod
    def _trace_surface_kwargs(
        trace_call_boundary: object | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Use the service-issued surface objects at the final verifier seam."""

        surface_boundary = getattr(trace_call_boundary, "surface_boundary", None)
        issued_values = getattr(
            surface_boundary,
            "_provider_request_surface_values",
            None,
        )
        if not callable(issued_values):
            return kwargs
        values = issued_values()
        if not isinstance(values, Mapping):
            raise TraceCallPersistenceError()
        for name in ("messages_payload", "provider_continuations"):
            value = values.get(name)
            if not isinstance(value, tuple):
                raise TraceCallPersistenceError()
            kwargs[name] = value
        return kwargs

    @staticmethod
    def _provenance_for_route(
        provenance: ProviderRequestProvenance | None,
        route: ConsoleRequestRoute,
    ) -> ProviderRequestProvenance:
        """Replace the single route descriptor for an internal provider call."""

        if provenance is None:
            raise TraceProvenanceAlignmentError(
                "provider request trace verification failed"
            )
        descriptor = request_route_provenance(route)
        route_replaced = False
        metadata: list[TraceProvenance] = []
        for item in provenance.metadata:
            if type(item) is RequestRouteTraceProvenance:
                if route_replaced:
                    raise TraceProvenanceAlignmentError(
                        "provider request trace verification failed"
                    )
                metadata.append(descriptor)
                route_replaced = True
            else:
                metadata.append(item)
        if not route_replaced:
            raise TraceProvenanceAlignmentError(
                "provider request trace verification failed"
            )
        return replace(provenance, metadata=tuple(metadata))

    @staticmethod
    def _commit_trace_dispatch_started(
        trace_call_boundary: object | None,
        bundle: ProviderRequestShadowBundle | None,
        provenance: object,
    ) -> None:
        """Commit the normalized dispatch token as the adapter-adjacent step."""

        if trace_call_boundary is None:
            return
        if bundle is None:
            raise TraceCallPersistenceError()
        try:
            mark_dispatch_started = getattr(
                trace_call_boundary,
                "mark_dispatch_started",
                None,
            )
            if not callable(mark_dispatch_started):
                raise TraceCallPersistenceError()
            mark_dispatch_started(bundle, provenance)
        except TraceCallPersistenceError as exc:
            if exc.boundary is None:
                raise TraceCallPersistenceError(boundary=trace_call_boundary) from None
            raise
        except Exception:
            raise TraceCallPersistenceError() from None

    def _trace_dispatch_admission(
        self,
        trace_call_boundary: object | None,
        bundle: ProviderRequestShadowBundle | None,
        provenance: object,
        *,
        route: ConsoleRequestRoute | None,
        capture_off_admission: _ProviderAdapterAdmission | None = None,
    ) -> _ProviderAdapterAdmission:
        """Return a token only after Capture On commits or Capture Off admits."""

        if trace_call_boundary is None:
            if capture_off_admission is None:
                raise TraceCallPersistenceError()
            return capture_off_admission
        self._commit_trace_dispatch_started(
            trace_call_boundary,
            bundle,
            provenance,
        )
        return _ProviderAdapterAdmission(
            self._adapter_admission_issuer,
            ConsoleTraceCaptureMode.CAPTURE_ON,
            route,
        )

    @staticmethod
    def _commit_trace_dispatch_unknown(trace_call_boundary: object | None) -> None:
        """Make a post-normalized, pre-adapter caller-checkpoint failure honest."""

        if trace_call_boundary is None:
            return
        try:
            mark_dispatch_unknown = getattr(
                trace_call_boundary,
                "mark_dispatch_unknown",
                None,
            )
            if not callable(mark_dispatch_unknown):
                raise TraceCallPersistenceError()
            mark_dispatch_unknown()
        except TraceCallPersistenceError:
            raise
        except Exception:
            raise TraceCallPersistenceError() from None

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
    def _resolution_settings(
        config: LlamaCppProviderConfig,
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
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
            **_thinking_stream_capability(
                "llama_cpp",
                model=model or config.explicit_model or config.configured_model,
                reasoning_effort=config.reasoning_effort,
            ),
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
