"""Console-native provider resolution and streaming gateway."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from types import GeneratorType
from typing import Any, AsyncIterator, Callable
from urllib.parse import urlparse, urlunparse

import httpx

from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_provider_endpoints import (
    generic_endpoint_differs,
    provider_uses_endpoint,
    unsaved_endpoint_copy,
)
from tldw_chatbook.Chat.console_provider_support import resolve_console_provider_identity
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness, provider_config_key
from tldw_chatbook.Utils.input_validation import validate_url


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


@dataclass(frozen=True)
class _QueueItem:
    kind: str
    text: str = ""
    payload: Any = None

    @classmethod
    def content(cls, text: str) -> "_QueueItem":
        return cls("content", text)

    @classmethod
    def error(cls, text: str) -> "_QueueItem":
        return cls("error", text)

    @classmethod
    def done(cls) -> "_QueueItem":
        return cls("done")

    @classmethod
    def native_tool_calls(cls, calls: tuple) -> "_QueueItem":
        return cls("tool_calls", payload=calls)


@dataclass(frozen=True)
class ProviderToolCalls:
    """Accumulated native tool-calls, yielded as ``stream_chat``'s FINAL
    item -- and only when the caller passed ``tools=``. Plain Console sends
    never receive one. ``tool_calls`` entries are OpenAI-shape dicts with
    streaming fragments already merged."""

    tool_calls: tuple[dict, ...]


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
            self._by_index[index] = {"id": "", "type": "function",
                                     "function": {"name": "", "arguments": ""}}
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
        # Preserve provider-specific extra keys (last-wins) — e.g. Gemini 3
        # thought signatures carried as google_thought_signature, which the
        # request converter must echo back verbatim (task-266 live gate).
        for key, value in fragment.items():
            if key in ("index", "id", "type", "function"):
                continue
            if value:
                entry[key] = value

    def calls(self) -> tuple[dict, ...]:
        # Numeric index order, not first-seen order: the provider's index
        # field defines the batch's array position, and fragments may arrive
        # interleaved/out of order (PR #648 review).
        return tuple(self._by_index[i] for i in sorted(self._by_index)
                     if self._by_index[i]["function"]["name"])


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
        accumulator.feed_payload(response)
        return response
    if not _is_iterable_response(response):
        return response

    def generator():
        for item in response:
            accumulator.feed_payload(_decode_stream_item(item))
            yield item
    return generator()


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

    Returns:
        Request payload for the llama.cpp chat completions endpoint.
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
        self._client_loop: asyncio.AbstractEventLoop | None = None
        # Guards the check-and-swap in `_active_http_client` as a single
        # atomic critical section (PR #629 Fix 1(a)): concurrent callers on
        # different loops/threads -- e.g. the app loop's readiness probe
        # racing the agent worker thread's per-turn loop -- must never
        # interleave the read-then-write, which could otherwise desync
        # `http_client` from `_client_loop` (a client bound to one loop
        # while the recorded loop is a different one).
        self._client_lock = threading.Lock()
        self._config_provider = config_provider or (lambda: {})
        self._environ = environ
        self._chat_api_call_fn = chat_api_call_fn
        self._safe_error_copy = safe_error_copy or safe_provider_error_copy

    async def aclose(self) -> None:
        """Close the owned HTTP client.

        Returns:
            ``None``. Injected HTTP clients are left open for their owner.
        """
        if self._owns_http_client:
            await self.http_client.aclose()

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
        loop." Recreate the owned client whenever the running loop changes
        so each loop gets its own client; injected clients (tests) are
        trusted to manage their own loop lifecycle and are left untouched.

        The read-check-swap below is guarded by ``_client_lock`` (PR #629
        Fix 1(a)): without it, two concurrent callers on different
        loops/threads (the app loop's readiness probe racing the agent
        worker thread's per-turn loop) can each read the same stale
        ``(http_client, _client_loop)`` pair before either writes, then
        both swap -- whichever writer's ``self.http_client`` assignment
        loses the race ends up paired with the *other* writer's
        ``self._client_loop`` assignment, leaving the recorded loop and the
        actual client bound to different loops. The very next probe on the
        recorded loop then reuses a client bound elsewhere and crashes.
        Holding the lock across the whole check, creation, and both
        assignments makes the swap a single atomic step.
        """
        if not self._owns_http_client:
            return self.http_client
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self.http_client
        with self._client_lock:
            if self._client_loop is loop:
                return self.http_client
            stale_client, stale_loop = self.http_client, self._client_loop
            self.http_client = self._new_owned_http_client()
            self._client_loop = loop
            active_client = self.http_client
        # Best-effort close of whatever this swap replaced -- including the
        # very first swap, where `stale_loop` is still `None` (there is no
        # previous loop to close it on). A dropped owned client must never
        # simply leak (PR #629 Fix 1(b)): fall back to closing it on the
        # loop that just replaced it, which is safe since we are executing
        # inside that running loop right now. Scheduling the close outside
        # the lock keeps the critical section itself minimal.
        self._schedule_stale_client_close(
            stale_client, stale_loop if stale_loop is not None else loop
        )
        return active_client

    @staticmethod
    def _schedule_stale_client_close(
        client: httpx.AsyncClient, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Best-effort close of a client left behind by a loop swap.

        The previous loop may already be closed (a completed per-turn
        ``asyncio.run()``) or may still be running elsewhere (the app's
        main loop) -- either way this must never raise into the caller
        that triggered the swap.
        """
        try:
            future = asyncio.run_coroutine_threadsafe(client.aclose(), loop)
        except RuntimeError:
            return
        future.add_done_callback(lambda f: f.exception())

    async def resolve_llamacpp(self, config: LlamaCppProviderConfig) -> ConsoleProviderResolution:
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
            if await self._is_reachable(base_url):
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

    async def resolve_for_send(self, selection: ConsoleProviderSelection) -> ConsoleProviderResolution:
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
            resolved = await self.resolve_llamacpp(
                LlamaCppProviderConfig(
                    base_url=selection.base_url or DEFAULT_LLAMACPP_BASE_URL,
                    explicit_model=selection.explicit_model,
                    configured_model=selection.configured_model,
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
        provider_settings = _provider_settings(app_config, identity.readiness_key)
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

        if (
            provider_uses_endpoint(identity.readiness_key, provider_settings)
            and generic_endpoint_differs(selection.base_url, provider_settings)
        ):
            return self._blocked_resolution(
                selection,
                provider=selection.provider,
                model=model,
                visible_copy=unsaved_endpoint_copy(selection.base_url, provider_settings),
                readiness_key=identity.readiness_key,
                execution_key=identity.execution_key,
            )

        readiness = get_provider_readiness(identity.readiness_key, app_config, environ=self._environ)
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

        return ConsoleProviderResolution(
            provider=selection.provider,
            base_url=selection.base_url or "",
            model=model,
            ready=True,
            readiness_key=identity.readiness_key,
            execution_key=identity.execution_key,
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
        )
        emitted_content = False
        stream_error: httpx.HTTPError | None = None
        try:
            async with self._active_http_client().stream(
                "POST",
                f"{normalized_base_url.rstrip('/')}/v1/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    chunk = self._content_from_sse_line(line)
                    if chunk:
                        emitted_content = True
                        yield chunk
        except httpx.HTTPError as exc:
            if emitted_content:
                raise
            stream_error = exc

        if emitted_content:
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

        Returns:
            Assistant-visible completion text.
        """
        normalized_base_url = normalize_llamacpp_base_url(base_url)
        if not validate_url(normalized_base_url):
            raise ValueError("invalid llama.cpp base URL")

        response = await self._active_http_client().post(
            f"{normalized_base_url.rstrip('/')}/v1/chat/completions",
            json=build_llamacpp_chat_payload(
                model=model,
                messages=messages,
                stream=False,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                max_tokens=max_tokens,
            ),
        )
        response.raise_for_status()
        return self._content_from_completion_response(response) or ""

    async def stream_chat(
        self,
        resolution: ConsoleProviderResolution,
        messages: list[Mapping[str, Any]],
        tools: list | None = None,
    ) -> AsyncIterator[str | ProviderToolCalls]:
        """Dispatch streaming for a resolved Console provider.

        Args:
            resolution: Provider resolution produced by ``resolve_for_send``.
            messages: OpenAI-compatible chat messages.
            tools: Optional OpenAI-shape tool definitions. When omitted,
                behavior is byte-identical to a plain Console send. When
                provided, yields str chunks as before; if the provider
                returned native tool-calls, the final item is a
                ``ProviderToolCalls`` instead of a str.

        Yields:
            Assistant-visible content chunks, and -- only when ``tools`` was
            passed and the provider returned native tool-calls -- a final
            ``ProviderToolCalls``.
        """
        if not resolution.ready or not resolution.model:
            return
        if resolution.provider in {"llama_cpp", "local_llamacpp"}:
            if not resolution.streaming:
                completion = await self.complete_llamacpp_chat(
                    base_url=resolution.base_url,
                    model=resolution.model,
                    messages=messages,
                    temperature=resolution.temperature,
                    top_p=resolution.top_p,
                    min_p=resolution.min_p,
                    top_k=resolution.top_k,
                    max_tokens=resolution.max_tokens,
                )
                if completion:
                    yield completion
                return
            async for chunk in self.stream_llamacpp_chat(
                base_url=resolution.base_url,
                model=resolution.model,
                messages=messages,
                temperature=resolution.temperature,
                top_p=resolution.top_p,
                min_p=resolution.min_p,
                top_k=resolution.top_k,
                max_tokens=resolution.max_tokens,
            ):
                yield chunk
            return
        if resolution.execution_key:
            async for chunk in self._stream_generic_chat(resolution, messages, tools=tools):
                yield chunk
            return

    async def _stream_generic_chat(
        self,
        resolution: ConsoleProviderResolution,
        messages: list[Mapping[str, Any]],
        tools: list | None = None,
    ) -> AsyncIterator[str | ProviderToolCalls]:
        """Bridge synchronous chat_api_call responses into async Console chunks."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        stop_event = threading.Event()

        def enqueue(item: _QueueItem) -> None:
            if stop_event.is_set():
                return
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(queue.put_nowait, item)

        def worker() -> None:
            try:
                kwargs = self._chat_api_kwargs(resolution, messages, tools=tools)
                response = self._chat_api_call(**kwargs)
                accumulator = _ToolCallAccumulator() if tools else None
                if accumulator is not None:
                    response = _tee_tool_calls(response, accumulator)
                suppressed_fallback_copy = False
                emitted_content = False
                for text in self.normalize_provider_response(response):
                    if stop_event.is_set():
                        break
                    if accumulator is not None and text in (
                            NO_PROVIDER_CONTENT_COPY,
                            UNSUPPORTED_PROVIDER_RESPONSE_COPY):
                        # tools= runs: fallback UI copy must never leak into
                        # agent history -- a tool-call-only turn legitimately
                        # has no visible content.
                        suppressed_fallback_copy = True
                        continue
                    if text:
                        emitted_content = True
                    enqueue(_QueueItem.content(text))
                if accumulator is not None:
                    calls = accumulator.calls()
                    if calls:
                        enqueue(_QueueItem.native_tool_calls(calls))
                    elif suppressed_fallback_copy and not emitted_content:
                        # PR #648 review Minor 1: the turn produced NEITHER
                        # visible content NOR tool-calls -- only filtered
                        # fallback copy. On the fence path that copy surfaces
                        # as the (diagnostic) answer; silently completing here
                        # would make a misbehaving provider's junk 200-body
                        # indistinguishable from a legitimate empty answer.
                        # Surface it as a provider error instead, feeding the
                        # run's existing honest RUN_ERROR outcome path.
                        enqueue(_QueueItem.error(self._safe_error_copy(
                            resolution.provider,
                            ChatProviderError(
                                "Provider returned no content and no tool calls.",
                                provider=resolution.provider))))
            except BaseException as exc:
                enqueue(_QueueItem.error(self._safe_error_copy(resolution.provider, exc)))
            finally:
                enqueue(_QueueItem.done())

        worker_task = asyncio.create_task(asyncio.to_thread(worker))
        try:
            while True:
                item = await queue.get()
                if item.kind == "done":
                    break
                if item.kind == "error":
                    raise ChatProviderError(
                        item.text or safe_provider_error_copy(resolution.provider, ChatProviderError()),
                        provider=resolution.provider,
                    )
                if item.kind == "tool_calls":
                    yield ProviderToolCalls(tuple(item.payload))
                    continue
                if item.text:
                    yield item.text
        finally:
            stop_event.set()
            if not worker_task.done():
                worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(worker_task), timeout=0)

    @staticmethod
    def normalize_provider_response(response: Any) -> Iterator[str]:
        """Yield safe assistant-visible chunks from generic provider output.

        Args:
            response: Raw return value from ``chat_api_call``.

        Yields:
            Assistant-visible text chunks or normalized fallback copy.
        """
        content = _content_from_provider_item(response)
        if isinstance(content, str):
            yield content if content else NO_PROVIDER_CONTENT_COPY
            return
        if content is _UNSUPPORTED_RESPONSE:
            if _is_iterable_response(response):
                emitted = False
                for item in response:
                    item_content = _content_from_provider_item(item)
                    if isinstance(item_content, str):
                        if item_content:
                            emitted = True
                            yield item_content
                        continue
                    if item_content is _EMPTY_RESPONSE:
                        continue
                    emitted = True
                    yield UNSUPPORTED_PROVIDER_RESPONSE_COPY
                if not emitted:
                    yield NO_PROVIDER_CONTENT_COPY
                return
            yield UNSUPPORTED_PROVIDER_RESPONSE_COPY
            return
        yield NO_PROVIDER_CONTENT_COPY

    def _chat_api_call(self, **kwargs: Any) -> Any:
        if self._chat_api_call_fn is None:
            from tldw_chatbook.Chat.Chat_Functions import chat_api_call

            return chat_api_call(**kwargs)
        return self._chat_api_call_fn(**kwargs)

    @staticmethod
    def _chat_api_kwargs(
        resolution: ConsoleProviderResolution,
        messages: list[Mapping[str, Any]],
        tools: list | None = None,
    ) -> dict[str, Any]:
        kwargs = {
            "api_endpoint": resolution.execution_key,
            "messages_payload": list(messages),
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
        }
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

    async def _is_reachable(self, base_url: str) -> bool:
        try:
            await self._active_http_client().get(
                f"{base_url.rstrip('/')}/health",
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
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]:
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
            model=model if model is not None else selection.explicit_model or selection.configured_model,
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


def _is_iterable_response(response: Any) -> bool:
    return isinstance(response, (Iterator, GeneratorType)) and not isinstance(response, (str, bytes, Mapping, list, tuple))


def _content_from_provider_item(item: Any) -> str | object:
    if isinstance(item, str):
        if item.startswith("data:"):
            return _content_from_sse_data(item)
        return item
    if isinstance(item, bytes):
        decoded = item.decode("utf-8", errors="replace")
        if decoded.startswith("data:"):
            return _content_from_sse_data(decoded)
        return decoded
    if isinstance(item, Mapping):
        return _content_from_provider_mapping(item)
    return _UNSUPPORTED_RESPONSE


def _content_from_sse_data(line: str) -> str | object:
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
                        if isinstance(part, Mapping) and isinstance(part.get("text"), str)
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


def _provider_settings(app_config: Mapping[str, object], provider_key: str) -> Mapping[str, object]:
    api_settings = _mapping_value(app_config, "api_settings")
    for configured_provider, configured_value in api_settings.items():
        if provider_config_key(str(configured_provider)) == provider_key:
            return configured_value if isinstance(configured_value, Mapping) else {}
    return {}


def _first_string(*values: object) -> str | None:
    for value in values:
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return None
