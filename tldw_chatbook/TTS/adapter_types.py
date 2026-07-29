from __future__ import annotations

import asyncio
import re
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
CapabilitySnapshotState = Literal["complete", "unverified"]
_VOICE_DISCOVERY_PROVIDER_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


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
    """An immutable, status-aware observation of one model's voices.

    ``complete`` may authoritatively contain no voices. ``model_missing`` is
    authoritative only when empty. ``unverified`` may retain validated partial
    voices for diagnostics, but compatibility callers must not project them.
    """

    provider_id: str
    model_id: str
    catalog_revision: int
    voices: tuple[str, ...]
    state: VoiceDiscoveryState

    def __post_init__(self) -> None:
        if type(self.provider_id) is not str or not (
            _VOICE_DISCOVERY_PROVIDER_ID_PATTERN.fullmatch(self.provider_id)
        ):
            raise ValueError("Voice discovery provider ID is invalid")
        _validate_voice_discovery_identifier(self.model_id, "model ID")
        if type(self.catalog_revision) is not int:
            raise TypeError("Voice discovery catalog revision must be an integer")
        if self.catalog_revision < 0:
            raise ValueError("Voice discovery catalog revision must be nonnegative")
        if type(self.voices) is not tuple:
            raise TypeError("Voice discovery voices must be a tuple")
        for voice in self.voices:
            _validate_voice_discovery_identifier(voice, "voice")
        if type(self.state) is not str:
            raise TypeError("Voice discovery state must be a string")
        if self.state not in ("complete", "model_missing", "unverified"):
            raise ValueError("Voice discovery state is invalid")
        if self.state == "model_missing" and self.voices:
            raise ValueError("Missing-model voice discovery must be empty")


@dataclass(frozen=True, slots=True)
class TTSNativeCapabilitySnapshot:
    """One configuration- and catalog-coherent native capability observation."""

    provider_id: str
    configuration_revision: int
    state: CapabilitySnapshotState
    catalog: TTSProviderCatalog | None
    voice_results: Mapping[str, TTSVoiceDiscoveryResult]

    def __post_init__(self) -> None:
        if type(self.provider_id) is not str or not (
            _VOICE_DISCOVERY_PROVIDER_ID_PATTERN.fullmatch(self.provider_id)
        ):
            raise ValueError("Capability snapshot provider ID is invalid")
        if type(self.configuration_revision) is not int:
            raise TypeError("Capability configuration revision must be an integer")
        if self.configuration_revision < 0:
            raise ValueError("Capability configuration revision must be nonnegative")
        if type(self.state) is not str:
            raise TypeError("Capability snapshot state must be a string")
        if self.state not in ("complete", "unverified"):
            raise ValueError("Capability snapshot state is invalid")
        if self.catalog is not None:
            if type(self.catalog) is not TTSProviderCatalog:
                raise TypeError("Capability catalog must be a provider catalog")
            if (
                type(self.catalog.provider_id) is not str
                or self.catalog.provider_id != self.provider_id
            ):
                raise ValueError("Capability catalog provider does not match")
            if type(self.catalog.revision) is not int:
                raise TypeError("Capability catalog revision must be an integer")
            if self.catalog.revision < 0:
                raise ValueError("Capability catalog revision must be nonnegative")
            if type(self.catalog.health) is not ProviderHealth:
                raise TypeError("Capability catalog health is invalid")
            if type(self.catalog.health.fresh) is not bool:
                raise TypeError("Capability catalog freshness must be a boolean")
            if type(self.catalog.models) is not tuple:
                raise TypeError("Capability catalog models must be a tuple")
        if not isinstance(self.voice_results, Mapping):
            raise TypeError("Capability voice results must be a mapping")

        catalog_model_ids: set[str] = set()
        if self.catalog is not None:
            for model in self.catalog.models:
                if type(model) is not TTSModelInfo:
                    raise TypeError("Capability catalog model is invalid")
                _validate_voice_discovery_identifier(model.model_id, "model ID")
                catalog_model_ids.add(model.model_id)

        frozen_results: dict[str, TTSVoiceDiscoveryResult] = {}
        for model_id, result in self.voice_results.items():
            _validate_voice_discovery_identifier(model_id, "model ID")
            if type(result) is not TTSVoiceDiscoveryResult:
                raise TypeError("Capability voice results must be discovery results")
            if result.provider_id != self.provider_id:
                raise ValueError("Capability voice result provider does not match")
            if result.model_id != model_id:
                raise ValueError("Capability voice result model does not match")
            if result.state != "unverified":
                if self.catalog is None:
                    raise ValueError("Authoritative voice result requires a catalog")
                if (
                    not self.catalog.health.fresh
                    or result.catalog_revision != self.catalog.revision
                ):
                    raise ValueError("Authoritative voice result catalog is stale")
                if result.state == "complete" and model_id not in catalog_model_ids:
                    raise ValueError("Complete voice result requires a catalog model")
                if result.state == "model_missing" and model_id in catalog_model_ids:
                    raise ValueError("Missing voice result contradicts the catalog")
            frozen_results[model_id] = result

        if self.state == "complete":
            if self.catalog is None:
                raise ValueError("Complete capability snapshot requires a catalog")
            if not self.catalog.health.fresh:
                raise ValueError("Complete capability snapshot requires fresh catalog")
            for result in frozen_results.values():
                if result.state == "unverified":
                    raise ValueError(
                        "Complete capability snapshot cannot contain unverified voices"
                    )
                if result.catalog_revision != self.catalog.revision:
                    raise ValueError(
                        "Complete capability snapshot revisions must match"
                    )

        object.__setattr__(
            self,
            "voice_results",
            MappingProxyType(frozen_results),
        )


def _validate_voice_discovery_identifier(value: object, label: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"Voice discovery {label} is invalid")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise ValueError(f"Voice discovery {label} is invalid") from None


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
