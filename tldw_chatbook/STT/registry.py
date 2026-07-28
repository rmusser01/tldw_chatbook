"""Dependency-free speech-to-text provider metadata and registry."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol, cast, runtime_checkable

from .contracts import (
    CancellationGranularity,
    ExecutionDevice,
    InputKind,
    LanguageInputMode,
    ProducedCapabilities,
    ResolvedTranscriptionRequest,
    TimestampGranularity,
    TranscriptionSegment,
    TranscriptionTask,
    TranscriptionTimings,
    TranscriptionWarningCode,
)

_LANGUAGE_PATTERN = re.compile(r"(?:auto|[a-z]{2,3}(?:-[a-z0-9]{1,8})*)")
_DETAIL_CODE_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*")
_MAX_DETAIL_CODE_LENGTH = 128
_SEMANTIC_CAPABILITY_FIELDS = (
    "languages",
    "automatic_language",
    "tasks",
    "inputs",
    "timestamps",
    "true_streaming",
    "batch",
    "cancellation",
    "vad",
    "diarization",
    "punctuation",
    "capitalization",
    "language_input_mode",
)


class ProviderRegistryError(Exception):
    """Base class for stable provider-registry validation errors."""


class CatalogDeclarationError(ProviderRegistryError):
    """Raised when catalog declarations are inconsistent."""


class DuplicateDeclarationError(CatalogDeclarationError):
    """Raised when an exact provider or model identity is declared twice."""


class AdapterRegistrationError(ProviderRegistryError):
    """Raised when a supplied adapter does not match the sealed catalog."""


class DuplicateAdapterError(AdapterRegistrationError):
    """Raised when more than one adapter serves an exact provider or model."""


class RuntimeCapabilityError(ProviderRegistryError):
    """Raised when a runtime observation conflicts with its declaration."""


def _require_string(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> None:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if value != value.strip():
        raise ValueError(f"{field_name} must not have surrounding whitespace")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a bool")


def _require_enum(value: object, enum_type: type[object], field_name: str) -> None:
    if type(value) is not enum_type:
        raise TypeError(f"{field_name} must be a {enum_type.__name__}")


def _require_frozenset(
    value: object,
    item_type: type[object],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> None:
    if type(value) is not frozenset:
        raise TypeError(f"{field_name} must be a frozenset")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must not be empty")
    if not all(type(item) is item_type for item in value):
        raise TypeError(f"{field_name} contains an invalid value")


def _require_language(language: object, field_name: str) -> None:
    _require_string(language, field_name)
    if not _LANGUAGE_PATTERN.fullmatch(cast(str, language)):
        raise ValueError(
            f"{field_name} must be 'auto' or a canonical lower-case language tag"
        )


@dataclass(frozen=True, slots=True)
class CapabilitySet:
    """Complete declared or runtime-observed model capabilities."""

    languages: frozenset[str]
    automatic_language: bool
    tasks: frozenset[TranscriptionTask]
    inputs: frozenset[InputKind]
    timestamps: frozenset[TimestampGranularity]
    true_streaming: bool
    batch: bool
    cancellation: CancellationGranularity
    vad: bool
    diarization: bool
    punctuation: bool
    capitalization: bool
    language_input_mode: LanguageInputMode
    execution_devices: frozenset[ExecutionDevice]
    precisions: frozenset[str]

    def __post_init__(self) -> None:
        _require_frozenset(
            self.languages,
            str,
            "languages",
            allow_empty=True,
        )
        for language in self.languages:
            _require_language(language, "languages")
            if language == "auto":
                raise ValueError(
                    "languages must contain explicit languages, not 'auto'"
                )
        _require_bool(self.automatic_language, "automatic_language")
        _require_frozenset(self.tasks, TranscriptionTask, "tasks")
        _require_frozenset(self.inputs, InputKind, "inputs")
        _require_frozenset(
            self.timestamps,
            TimestampGranularity,
            "timestamps",
        )
        _require_bool(self.true_streaming, "true_streaming")
        _require_bool(self.batch, "batch")
        _require_enum(
            self.cancellation,
            CancellationGranularity,
            "cancellation",
        )
        _require_bool(self.vad, "vad")
        _require_bool(self.diarization, "diarization")
        _require_bool(self.punctuation, "punctuation")
        _require_bool(self.capitalization, "capitalization")
        _require_enum(
            self.language_input_mode,
            LanguageInputMode,
            "language_input_mode",
        )
        if not self.automatic_language and self.language_input_mode in {
            LanguageInputMode.AUTOMATIC,
            LanguageInputMode.AUTOMATIC_ONLY,
        }:
            raise ValueError(
                "automatic language input modes require automatic_language"
            )
        _require_frozenset(
            self.execution_devices,
            ExecutionDevice,
            "execution_devices",
        )
        if ExecutionDevice.AUTO in self.execution_devices:
            raise ValueError("execution_devices must contain only concrete devices")
        _require_frozenset(self.precisions, str, "precisions")
        for precision in self.precisions:
            _require_string(precision, "precisions")


@dataclass(frozen=True, slots=True)
class ProviderMetadata:
    """One exact provider declaration."""

    provider_id: str
    display_name: str
    local_processing: bool

    def __post_init__(self) -> None:
        _require_string(self.provider_id, "provider_id")
        _require_string(self.display_name, "display_name")
        _require_bool(self.local_processing, "local_processing")


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """One exact model declaration."""

    provider_id: str
    model_id: str
    display_name: str
    capabilities: CapabilitySet
    default_precision: str
    semantic_default_eligible: bool
    enforces_language_hint: bool

    def __post_init__(self) -> None:
        _require_string(self.provider_id, "provider_id")
        _require_string(self.model_id, "model_id")
        _require_string(self.display_name, "display_name")
        if type(self.capabilities) is not CapabilitySet:
            raise TypeError("capabilities must be a CapabilitySet")
        _require_string(self.default_precision, "default_precision")
        if self.default_precision not in self.capabilities.precisions:
            raise ValueError("default_precision must be a declared precision")
        _require_bool(
            self.semantic_default_eligible,
            "semantic_default_eligible",
        )
        _require_bool(self.enforces_language_hint, "enforces_language_hint")


@dataclass(frozen=True, slots=True)
class CatalogDeclarations:
    """Immutable provider and model declarations used to seal a registry."""

    providers: tuple[ProviderMetadata, ...]
    models: tuple[ModelMetadata, ...]

    def __post_init__(self) -> None:
        if type(self.providers) is not tuple:
            raise TypeError("providers must be a tuple")
        if not self.providers:
            raise ValueError("providers must not be empty")
        if not all(type(provider) is ProviderMetadata for provider in self.providers):
            raise TypeError("providers must contain only ProviderMetadata values")
        if type(self.models) is not tuple:
            raise TypeError("models must be a tuple")
        if not self.models:
            raise ValueError("models must not be empty")
        if not all(type(model) is ModelMetadata for model in self.models):
            raise TypeError("models must contain only ModelMetadata values")


@dataclass(frozen=True, slots=True)
class RuntimeObservation:
    """Availability and capabilities observed for one exact model at runtime."""

    provider_id: str
    model_id: str
    available: bool
    capabilities: CapabilitySet | None
    detail_code: str | None = None

    def __post_init__(self) -> None:
        _require_string(self.provider_id, "provider_id")
        _require_string(self.model_id, "model_id")
        _require_bool(self.available, "available")
        if self.available and type(self.capabilities) is not CapabilitySet:
            raise ValueError("available observations require capabilities")
        if not self.available and self.capabilities is not None:
            raise ValueError("unavailable observations must not include capabilities")
        if self.detail_code is not None:
            _require_string(self.detail_code, "detail_code")
            if len(
                self.detail_code
            ) > _MAX_DETAIL_CODE_LENGTH or not _DETAIL_CODE_PATTERN.fullmatch(
                self.detail_code
            ):
                raise ValueError(
                    "detail_code must be a stable lower-case code of at most "
                    f"{_MAX_DETAIL_CODE_LENGTH} characters"
                )


@dataclass(frozen=True, slots=True)
class ProviderTranscriptionOutput:
    """Provider output before coordinator-owned provenance normalization."""

    text: str = field(repr=False)
    segments: tuple[TranscriptionSegment, ...] = field(repr=False)
    effective_language: str
    detected_language: str | None
    effective_device: ExecutionDevice
    produced_capabilities: ProducedCapabilities
    duration_seconds: float
    timings: TranscriptionTimings
    warnings: tuple[TranscriptionWarningCode, ...] = ()

    def __post_init__(self) -> None:
        _require_string(self.text, "text", allow_empty=True)
        if type(self.segments) is not tuple or not all(
            type(segment) is TranscriptionSegment for segment in self.segments
        ):
            raise TypeError("segments must be a tuple of TranscriptionSegment values")
        _require_language(self.effective_language, "effective_language")
        if self.detected_language is not None:
            _require_language(self.detected_language, "detected_language")
        _require_enum(
            self.effective_device,
            ExecutionDevice,
            "effective_device",
        )
        if self.effective_device is ExecutionDevice.AUTO:
            raise ValueError("effective_device must be a concrete device")
        if type(self.produced_capabilities) is not ProducedCapabilities:
            raise TypeError("produced_capabilities must be a ProducedCapabilities")
        if type(self.duration_seconds) not in (int, float):
            raise TypeError("duration_seconds must be a number")
        if (
            self.duration_seconds < 0
            or type(self.duration_seconds) is float
            and not math.isfinite(self.duration_seconds)
        ):
            raise ValueError("duration_seconds must be finite and nonnegative")
        if type(self.timings) is not TranscriptionTimings:
            raise TypeError("timings must be a TranscriptionTimings")
        if type(self.warnings) is not tuple or not all(
            type(warning) is TranscriptionWarningCode for warning in self.warnings
        ):
            raise TypeError(
                "warnings must be a tuple of TranscriptionWarningCode values"
            )


@runtime_checkable
class TranscriptionAdapter(Protocol):
    """Dependency-free interface implemented by exact STT provider adapters."""

    def provider(self) -> ProviderMetadata:
        """Return the adapter's exact provider metadata."""

        ...

    def describe(self) -> tuple[ModelMetadata, ...]:
        """Return the exact declared model subset served by this adapter."""

        ...

    def probe(self, model_id: str) -> RuntimeObservation:
        """Probe one exact model without loading unrelated runtimes."""

        ...

    def transcribe(
        self,
        request: ResolvedTranscriptionRequest,
    ) -> ProviderTranscriptionOutput:
        """Transcribe one fully resolved request."""

        ...

    def close(self) -> None:
        """Release provider-owned resources best-effort."""

        ...


@dataclass(frozen=True, slots=True, init=False)
class ProviderRegistry:
    """An immutable exact-ID registry built from one catalog snapshot."""

    declarations: CatalogDeclarations
    _providers: Mapping[str, ProviderMetadata] = field(repr=False)
    _models: Mapping[tuple[str, str], ModelMetadata] = field(repr=False)
    _adapters: Mapping[str, TranscriptionAdapter] = field(repr=False)
    _model_adapters: Mapping[tuple[str, str], TranscriptionAdapter] = field(repr=False)

    @classmethod
    def sealed(
        cls,
        declarations: CatalogDeclarations,
        adapters: tuple[TranscriptionAdapter, ...] = (),
    ) -> ProviderRegistry:
        """Validate declarations and adapters, then build immutable lookup maps."""

        if type(declarations) is not CatalogDeclarations:
            raise TypeError("declarations must be a CatalogDeclarations")
        if type(adapters) is not tuple:
            raise TypeError("adapters must be a tuple")

        provider_map: dict[str, ProviderMetadata] = {}
        for provider in declarations.providers:
            if provider.provider_id in provider_map:
                raise DuplicateDeclarationError(
                    f"duplicate provider declaration: {provider.provider_id}"
                )
            provider_map[provider.provider_id] = provider

        model_map: dict[tuple[str, str], ModelMetadata] = {}
        for model in declarations.models:
            if model.provider_id not in provider_map:
                raise CatalogDeclarationError(
                    f"model references undeclared provider: {model.provider_id}"
                )
            model_key = (model.provider_id, model.model_id)
            if model_key in model_map:
                raise DuplicateDeclarationError(
                    f"duplicate model declaration: {model.provider_id}/{model.model_id}"
                )
            model_map[model_key] = model

        adapter_map: dict[str, TranscriptionAdapter] = {}
        model_adapter_map: dict[tuple[str, str], TranscriptionAdapter] = {}
        for adapter in adapters:
            cls._add_adapter(
                adapter,
                providers=provider_map,
                models=model_map,
                adapters=adapter_map,
                model_adapters=model_adapter_map,
            )

        registry = object.__new__(cls)
        object.__setattr__(registry, "declarations", declarations)
        object.__setattr__(registry, "_providers", MappingProxyType(provider_map))
        object.__setattr__(registry, "_models", MappingProxyType(model_map))
        object.__setattr__(registry, "_adapters", MappingProxyType(adapter_map))
        object.__setattr__(
            registry,
            "_model_adapters",
            MappingProxyType(model_adapter_map),
        )
        return registry

    @staticmethod
    def _add_adapter(
        adapter: TranscriptionAdapter,
        *,
        providers: Mapping[str, ProviderMetadata],
        models: Mapping[tuple[str, str], ModelMetadata],
        adapters: dict[str, TranscriptionAdapter],
        model_adapters: dict[tuple[str, str], TranscriptionAdapter],
    ) -> None:
        for method_name in ("provider", "describe", "probe", "transcribe", "close"):
            if not callable(getattr(adapter, method_name, None)):
                raise AdapterRegistrationError(
                    f"adapter must implement callable {method_name}()"
                )

        provider = adapter.provider()
        if type(provider) is not ProviderMetadata:
            raise AdapterRegistrationError(
                "adapter provider() must return ProviderMetadata"
            )
        declared_provider = providers.get(provider.provider_id)
        if declared_provider is None:
            raise AdapterRegistrationError(
                f"adapter references undeclared provider: {provider.provider_id}"
            )
        if declared_provider != provider:
            raise AdapterRegistrationError(
                f"adapter provider metadata mismatch: {provider.provider_id}"
            )
        if provider.provider_id in adapters:
            raise DuplicateAdapterError(
                f"duplicate adapter for provider: {provider.provider_id}"
            )

        described_models = adapter.describe()
        if type(described_models) is not tuple or not described_models:
            raise AdapterRegistrationError(
                "adapter describe() must return a non-empty model tuple"
            )
        seen_model_ids: set[str] = set()
        for model in described_models:
            if type(model) is not ModelMetadata:
                raise AdapterRegistrationError(
                    "adapter describe() must return only ModelMetadata values"
                )
            if model.provider_id != provider.provider_id:
                raise AdapterRegistrationError(
                    "adapter model provider metadata mismatch: "
                    f"{model.provider_id}/{model.model_id}"
                )
            if model.model_id in seen_model_ids:
                raise DuplicateAdapterError(
                    "duplicate adapter model description: "
                    f"{model.provider_id}/{model.model_id}"
                )
            seen_model_ids.add(model.model_id)
            declared_model = models.get((model.provider_id, model.model_id))
            if declared_model is None:
                raise AdapterRegistrationError(
                    "adapter references undeclared model: "
                    f"{model.provider_id}/{model.model_id}"
                )
            if declared_model != model:
                raise AdapterRegistrationError(
                    "adapter model metadata mismatch: "
                    f"{model.provider_id}/{model.model_id}"
                )

        adapters[provider.provider_id] = adapter
        for model in described_models:
            model_adapters[(model.provider_id, model.model_id)] = adapter

    @property
    def providers(self) -> Mapping[str, ProviderMetadata]:
        """Return the read-only exact provider map."""

        return self._providers

    @property
    def models(self) -> Mapping[tuple[str, str], ModelMetadata]:
        """Return the read-only exact provider/model map."""

        return self._models

    @property
    def adapters(self) -> Mapping[str, TranscriptionAdapter]:
        """Return the read-only exact provider/adapter map."""

        return self._adapters

    def provider(self, provider_id: str) -> ProviderMetadata | None:
        """Return an exact provider declaration, with no alias matching."""

        return self._providers.get(provider_id)

    def model(self, provider_id: str, model_id: str) -> ModelMetadata | None:
        """Return an exact model declaration, with no alias matching."""

        return self._models.get((provider_id, model_id))

    def adapter(self, provider_id: str) -> TranscriptionAdapter | None:
        """Return the exact provider adapter when one was supplied."""

        return self._adapters.get(provider_id)

    def adapter_for_model(
        self,
        provider_id: str,
        model_id: str,
    ) -> TranscriptionAdapter | None:
        """Return the sealed adapter for one exact provider/model identity."""

        return self._model_adapters.get((provider_id, model_id))

    def validate_observation(
        self,
        selected: ModelMetadata,
        observation: RuntimeObservation,
    ) -> RuntimeObservation:
        """Validate runtime identity and the declared capability lattice."""

        if type(selected) is not ModelMetadata:
            raise TypeError("selected must be a ModelMetadata")
        if type(observation) is not RuntimeObservation:
            raise TypeError("observation must be a RuntimeObservation")

        declared = self.model(selected.provider_id, selected.model_id)
        if declared is None or declared != selected:
            raise RuntimeCapabilityError(
                "selected metadata must exactly match a catalog declaration"
            )
        if (
            observation.provider_id != selected.provider_id
            or observation.model_id != selected.model_id
        ):
            raise RuntimeCapabilityError(
                "runtime observation identity must exactly match selected metadata"
            )
        if not observation.available:
            return observation

        runtime = observation.capabilities
        if runtime is None:
            raise RuntimeCapabilityError(
                "available runtime observation must include capabilities"
            )
        declared_capabilities = selected.capabilities
        for field_name in _SEMANTIC_CAPABILITY_FIELDS:
            if getattr(runtime, field_name) != getattr(
                declared_capabilities,
                field_name,
            ):
                raise RuntimeCapabilityError(
                    f"runtime semantic capability mismatch: {field_name}"
                )
        if not runtime.execution_devices <= declared_capabilities.execution_devices:
            raise RuntimeCapabilityError(
                "runtime capability escalation: execution_devices"
            )
        if not runtime.precisions <= declared_capabilities.precisions:
            raise RuntimeCapabilityError("runtime capability escalation: precisions")
        return observation
