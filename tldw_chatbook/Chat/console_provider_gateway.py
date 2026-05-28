"""Console-native provider resolution and streaming gateway."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Callable, Mapping
from urllib.parse import urlparse, urlunparse

import httpx

from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_provider_support import resolve_console_provider_identity
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness, provider_config_key
from tldw_chatbook.Utils.input_validation import validate_url


DEFAULT_LLAMACPP_BASE_URL = "http://127.0.0.1:9099"
INVALID_LLAMACPP_BASE_URL_COPY = (
    "Provider blocked: invalid llama.cpp base URL. "
    "Use an http(s) URL such as http://127.0.0.1:9099."
)


def normalize_llamacpp_base_url(api_url: str | None) -> str:
    """Return the llama.cpp origin root used before appending OpenAI paths."""
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
    """Configuration needed to resolve a llama.cpp-compatible provider."""

    base_url: str = DEFAULT_LLAMACPP_BASE_URL
    explicit_model: str | None = None
    configured_model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    min_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    streaming: bool = True


@dataclass(frozen=True)
class ConsoleProviderResolution:
    """Provider readiness result used by Console send and recovery UI."""

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
    streaming: bool = True


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
    """Build the OpenAI-compatible llama.cpp chat completion payload."""
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
    """Resolve Console providers and stream chat responses."""

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        config_provider: Callable[[], Mapping[str, object]] | None = None,
        environ: Mapping[str, str] | None = None,
        chat_api_call_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._owns_http_client = http_client is None
        self.http_client = http_client or httpx.AsyncClient(timeout=30.0)
        self._config_provider = config_provider or (lambda: {})
        self._environ = environ
        self._chat_api_call_fn = chat_api_call_fn

    async def aclose(self) -> None:
        """Close the owned HTTP client, leaving injected clients to their owner."""
        if self._owns_http_client:
            await self.http_client.aclose()

    async def resolve_llamacpp(self, config: LlamaCppProviderConfig) -> ConsoleProviderResolution:
        """Resolve llama.cpp readiness and the effective model."""
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
            response = await self.http_client.get(f"{base_url.rstrip('/')}/v1/models")
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
        """Resolve the provider selected by Console before sending."""
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

        if _has_different_generic_base_url(selection.base_url, provider_settings):
            return self._blocked_resolution(
                selection,
                provider=selection.provider,
                model=model,
                visible_copy="Provider blocked: save the endpoint in Settings before using it from Console.",
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
        """Stream OpenAI-compatible chat completion content chunks from llama.cpp."""
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
            async with self.http_client.stream(
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
        """Request a non-streaming OpenAI-compatible chat completion."""
        normalized_base_url = normalize_llamacpp_base_url(base_url)
        if not validate_url(normalized_base_url):
            raise ValueError("invalid llama.cpp base URL")

        response = await self.http_client.post(
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
    ) -> AsyncIterator[str]:
        """Dispatch streaming for a resolved Console provider."""
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

    @staticmethod
    def _resolution_settings(config: LlamaCppProviderConfig) -> dict[str, Any]:
        return {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "min_p": config.min_p,
            "top_k": config.top_k,
            "max_tokens": config.max_tokens,
            "streaming": config.streaming,
        }

    async def _is_reachable(self, base_url: str) -> bool:
        try:
            await self.http_client.get(f"{base_url.rstrip('/')}/health")
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
            streaming=selection.streaming,
        )


def _mapping_value(source: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = source.get(key, {})
    return value if isinstance(value, Mapping) else {}


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


def _has_different_generic_base_url(base_url: str | None, provider_settings: Mapping[str, object]) -> bool:
    selected_base_url = _normalize_generic_url_for_compare(base_url)
    if not selected_base_url:
        return False

    configured_base_url = _normalize_generic_url_for_compare(
        _first_string(
            provider_settings.get("api_url"),
            provider_settings.get("base_url"),
            provider_settings.get("api_base"),
            provider_settings.get("api_endpoint"),
            provider_settings.get("endpoint"),
        )
    )
    return selected_base_url != configured_base_url


def _normalize_generic_url_for_compare(url: str | None) -> str:
    raw_url = str(url or "").strip()
    if not raw_url:
        return ""
    try:
        parsed = urlparse(raw_url)
    except ValueError:
        return raw_url.rstrip("/")
    if parsed.scheme and parsed.netloc:
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        try:
            port = parsed.port
        except ValueError:
            return raw_url.rstrip("/")
        default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
        if default_port and parsed.hostname:
            hostname = parsed.hostname.lower()
            netloc = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
        return urlunparse((scheme, netloc, parsed.path.rstrip("/"), "", "", ""))
    return raw_url.rstrip("/")
