"""Console-native provider resolution and streaming gateway."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping

import httpx

from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection


DEFAULT_LLAMACPP_BASE_URL = "http://127.0.0.1:9099"


@dataclass(frozen=True)
class LlamaCppProviderConfig:
    """Configuration needed to resolve a llama.cpp-compatible provider."""

    base_url: str = DEFAULT_LLAMACPP_BASE_URL
    explicit_model: str | None = None
    configured_model: str | None = None


@dataclass(frozen=True)
class ConsoleProviderResolution:
    """Provider readiness result used by Console send and recovery UI."""

    provider: str
    base_url: str
    model: str | None
    ready: bool
    visible_copy: str = ""


class ConsoleProviderGateway:
    """Resolve Console providers and stream chat responses."""

    def __init__(self, *, http_client: httpx.AsyncClient | None = None) -> None:
        self.http_client = http_client or httpx.AsyncClient(timeout=30.0)

    async def resolve_llamacpp(self, config: LlamaCppProviderConfig) -> ConsoleProviderResolution:
        """Resolve llama.cpp readiness and the effective model."""
        model = config.explicit_model or config.configured_model
        if model is not None:
            if await self._is_reachable(config.base_url):
                return ConsoleProviderResolution(
                    provider="llama_cpp",
                    base_url=config.base_url,
                    model=model,
                    ready=True,
                )
            return ConsoleProviderResolution(
                provider="llama_cpp",
                base_url=config.base_url,
                model=model,
                ready=False,
                visible_copy=self._unreachable_copy(config.base_url),
            )

        try:
            response = await self.http_client.get(f"{config.base_url.rstrip('/')}/v1/models")
        except httpx.HTTPError:
            return ConsoleProviderResolution(
                provider="llama_cpp",
                base_url=config.base_url,
                model=None,
                ready=False,
                visible_copy=self._unreachable_copy(config.base_url),
            )

        model = self._first_model_id(response)
        if model is None:
            return ConsoleProviderResolution(
                provider="llama_cpp",
                base_url=config.base_url,
                model=None,
                ready=False,
                visible_copy="Provider blocked: select or configure a llama.cpp model.",
            )
        return ConsoleProviderResolution(
            provider="llama_cpp",
            base_url=config.base_url,
            model=model,
            ready=True,
        )

    async def resolve_for_send(self, selection: ConsoleProviderSelection) -> ConsoleProviderResolution:
        """Resolve the provider selected by Console before sending."""
        if selection.provider in {"llama_cpp", "local_llamacpp"}:
            return await self.resolve_llamacpp(
                LlamaCppProviderConfig(
                    base_url=selection.base_url or DEFAULT_LLAMACPP_BASE_URL,
                    explicit_model=selection.explicit_model,
                    configured_model=selection.configured_model,
                )
            )

        return ConsoleProviderResolution(
            provider=selection.provider,
            base_url=selection.base_url or "",
            model=selection.explicit_model or selection.configured_model,
            ready=False,
            visible_copy=(
                f"WIP: Console native provider '{selection.provider}' is not wired yet. "
                "Select llama.cpp for this slice."
            ),
        )

    async def stream_llamacpp_chat(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[Mapping[str, Any]],
    ) -> AsyncIterator[str]:
        """Stream OpenAI-compatible chat completion content chunks from llama.cpp."""
        payload = {
            "model": model,
            "messages": list(messages),
            "stream": True,
        }
        async with self.http_client.stream(
            "POST",
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                chunk = self._content_from_sse_line(line)
                if chunk:
                    yield chunk

    async def stream_chat(
        self,
        resolution: ConsoleProviderResolution,
        messages: list[Mapping[str, Any]],
    ) -> AsyncIterator[str]:
        """Dispatch streaming for a resolved Console provider."""
        if not resolution.ready or not resolution.model:
            return
        if resolution.provider in {"llama_cpp", "local_llamacpp"}:
            async for chunk in self.stream_llamacpp_chat(
                base_url=resolution.base_url,
                model=resolution.model,
                messages=messages,
            ):
                yield chunk
            return

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
    def _unreachable_copy(base_url: str) -> str:
        return (
            f"Provider blocked: llama.cpp server is not reachable at {base_url}. "
            "Start llama.cpp or update Console provider settings."
        )
