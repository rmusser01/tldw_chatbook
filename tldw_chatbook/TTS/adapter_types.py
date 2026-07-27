from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

ProgressSink = Callable[["TTSProgress"], Awaitable[None]]
CleanupCallback = Callable[[], Awaitable[None]]
ProviderFactory = Callable[[Mapping[str, Any]], "TTSAdapter"]
ProviderState = Literal[
    "available", "unavailable", "not_configured", "reconfiguring", "closed"
]
TTSOperationCode = Literal[
    "configuration_invalid",
    "connection_unavailable",
    "contract_incompatible",
    "not_configured",
    "request_invalid",
    "model_invalid",
    "server_busy",
    "generation_failed",
    "audio_response_invalid",
    "generation_timeout",
]
VoiceDiscoveryState = Literal["complete", "model_missing", "unverified"]


class UnknownTTSProviderError(LookupError):
    """Raised when an exact canonical provider ID is not registered."""


class TTSRegistryClosedError(RuntimeError):
    """Raised when a registry no longer admits operations."""


class TTSProviderReconfiguringError(RuntimeError):
    """Raised when an exclusive provider handoff blocks new operations."""


class TTSConfigurationRevisionError(RuntimeError):
    """Raised when request selection and provider revision no longer match."""


class TTSProviderUnavailableError(RuntimeError):
    """Raised when a failed handoff has sealed a provider slot."""


class TTSOperationError(RuntimeError):
    """A provider-neutral operation failure with safe, stable details."""

    __slots__ = ("_code", "_retryable", "_operation_id", "_recovery_action")

    _code: TTSOperationCode
    _retryable: bool
    _operation_id: str
    _recovery_action: str | None
    _standard_exception_attributes = frozenset(
        {
            "__cause__",
            "__context__",
            "__notes__",
            "__suppress_context__",
            "__traceback__",
        }
    )

    def __init__(
        self,
        *,
        code: TTSOperationCode,
        message: str,
        retryable: bool,
        operation_id: str,
        recovery_action: str | None = None,
    ) -> None:
        super().__init__(message)
        object.__setattr__(self, "_code", code)
        object.__setattr__(self, "_retryable", retryable)
        object.__setattr__(self, "_operation_id", operation_id)
        object.__setattr__(self, "_recovery_action", recovery_action)

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._standard_exception_attributes:
            BaseException.__setattr__(self, name, value)
            return
        raise AttributeError("TTS operation error attributes are read-only")

    def __reduce__(
        self,
    ) -> tuple[Callable[..., TTSOperationError], tuple[Any, ...]]:
        notes = tuple(
            note for note in getattr(self, "__notes__", ()) if isinstance(note, str)
        )
        return (
            _restore_tts_operation_error,
            (
                self.code,
                str(self),
                self.retryable,
                self.operation_id,
                self.recovery_action,
                notes,
            ),
        )

    @property
    def code(self) -> TTSOperationCode:
        return self._code

    @property
    def retryable(self) -> bool:
        return self._retryable

    @property
    def operation_id(self) -> str:
        return self._operation_id

    @property
    def recovery_action(self) -> str | None:
        return self._recovery_action


def _restore_tts_operation_error(
    code: TTSOperationCode,
    message: str,
    retryable: bool,
    operation_id: str,
    recovery_action: str | None,
    notes: tuple[str, ...],
) -> TTSOperationError:
    error = TTSOperationError(
        code=code,
        message=message,
        retryable=retryable,
        operation_id=operation_id,
        recovery_action=recovery_action,
    )
    for note in notes:
        error.add_note(note)
    return error


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
class TTSVoiceDiscoveryResult:
    """An immutable, status-aware observation of one model's voices."""

    provider_id: str
    model_id: str
    catalog_revision: int
    voices: tuple[str, ...]
    state: VoiceDiscoveryState

    def __post_init__(self) -> None:
        if type(self.provider_id) is not str or not self.provider_id:
            raise ValueError("Voice discovery provider ID must be a non-empty string")
        if type(self.model_id) is not str or not self.model_id:
            raise ValueError("Voice discovery model ID must be a non-empty string")
        if type(self.catalog_revision) is not int:
            raise TypeError("Voice discovery catalog revision must be an integer")
        if self.catalog_revision < 0:
            raise ValueError("Voice discovery catalog revision must be nonnegative")
        if type(self.voices) is not tuple or any(
            type(voice) is not str or not voice for voice in self.voices
        ):
            raise TypeError("Voice discovery voices must be a tuple of non-empty strings")
        if type(self.state) is not str:
            raise TypeError("Voice discovery state must be a string")
        if self.state not in ("complete", "model_missing", "unverified"):
            raise ValueError("Voice discovery state is invalid")


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
        metadata: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self.audio_format = audio_format
        self.content_type = content_type
        self.byte_stream = byte_stream
        self.sample_rate = sample_rate
        metadata_copy = {} if metadata is None else dict(metadata)
        if any(
            type(value) not in (str, int, float, bool, type(None))
            for value in metadata_copy.values()
        ):
            raise TypeError(
                "TTS audio response metadata values must be immutable scalars"
            )
        self.metadata = MappingProxyType(metadata_copy)
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

    async def get_voices(
        self,
        model_id: str,
        refresh: bool = False,
    ) -> tuple[str, ...]:
        raise NotImplementedError

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


@runtime_checkable
class TTSStructuredVoiceAdapter(Protocol):
    """Optional adapter capability for authoritative voice discovery state."""

    async def observe_voices(
        self,
        model_id: str,
        refresh: bool = False,
    ) -> TTSVoiceDiscoveryResult:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class TTSProviderSpec:
    descriptor: TTSProviderDescriptor
    factory: ProviderFactory
    initial_config: Mapping[str, Any]
    exclusive_reconfigure: bool = False
