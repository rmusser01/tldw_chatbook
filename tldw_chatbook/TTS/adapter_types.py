from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

ProgressSink = Callable[["TTSProgress"], Awaitable[None]]
CleanupCallback = Callable[[], Awaitable[None]]
ProviderFactory = Callable[[Mapping[str, Any]], "TTSAdapter"]
ProviderState = Literal[
    "available", "unavailable", "not_configured", "reconfiguring", "closed"
]


class UnknownTTSProviderError(LookupError):
    """Raised when an exact canonical provider ID is not registered."""


class TTSRegistryClosedError(RuntimeError):
    """Raised when a registry no longer admits operations."""


class TTSProviderReconfiguringError(RuntimeError):
    """Raised when an exclusive provider handoff blocks new operations."""


@dataclass(frozen=True, slots=True)
class TTSRequest:
    provider_id: str
    model_id: str
    text: str
    voice: str | None
    response_format: str
    speed: float = 1.0
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True, slots=True)
class TTSProgress:
    status: str
    fraction: float | None = None
    processed: int | None = None
    total: int | None = None
    metrics: Mapping[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    state: ProviderState
    fresh: bool
    diagnostic: str | None = None
    retryable: bool = False
    recovery_action: str | None = None


@dataclass(frozen=True, slots=True)
class TTSModelInfo:
    model_id: str
    display_name: str
    family: str
    upstream_mode: str
    formats: tuple[str, ...]
    voices: tuple[str, ...]
    supports_speed: bool
    supports_options: tuple[str, ...] = ()
    omit_voice_uses_server_default: bool = False


@dataclass(frozen=True, slots=True)
class TTSProviderCatalog:
    provider_id: str
    revision: int
    health: ProviderHealth
    models: tuple[TTSModelInfo, ...]
    approximate: bool = False


@dataclass(frozen=True, slots=True)
class TTSProviderDescriptor:
    provider_id: str
    display_name: str
    native: bool


class TTSAudioResponse:
    def __init__(
        self,
        *,
        provider_id: str,
        model_id: str,
        audio_format: str,
        content_type: str,
        byte_stream: AsyncIterator[bytes],
        sample_rate: int | None = None,
        cleanup: CleanupCallback | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self.audio_format = audio_format
        self.content_type = content_type
        self.byte_stream = byte_stream
        self.sample_rate = sample_rate
        self._cleanup_callbacks = [cleanup] if cleanup is not None else []
        self._close_lock = asyncio.Lock()
        self._closed = False

    def add_cleanup(self, callback: CleanupCallback) -> None:
        if self._closed:
            raise RuntimeError("Cannot add cleanup to a closed audio response")
        self._cleanup_callbacks.append(callback)

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            first_error: BaseException | None = None
            stream_close = getattr(self.byte_stream, "aclose", None)
            if callable(stream_close):
                try:
                    await stream_close()
                except BaseException as error:
                    first_error = error
            for callback in self._cleanup_callbacks:
                try:
                    await callback()
                except BaseException as error:
                    first_error = first_error or error
            if first_error is not None:
                raise first_error

    async def __aenter__(self) -> "TTSAudioResponse":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()


class TTSAdapter(Protocol):
    async def ensure_ready(self) -> None:
        raise NotImplementedError

    async def get_catalog(self, refresh: bool = False) -> TTSProviderCatalog:
        raise NotImplementedError

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class TTSProviderSpec:
    descriptor: TTSProviderDescriptor
    factory: ProviderFactory
    initial_config: Mapping[str, Any]
    exclusive_reconfigure: bool = False
