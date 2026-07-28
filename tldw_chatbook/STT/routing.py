"""Authoritative built-in speech-to-text metadata and deterministic routing.

This module contains declarations and policy only. It intentionally does not
import native runtimes, configuration, artifact acquisition, persistence,
HTTP, UI, or the retained transcription implementation.

faster-whisper's broad explicit-language support is an immutable reviewed set
of exact provider language codes. No wildcard, prefix, or native-runtime lookup
participates in routing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .contracts import (
    TRANSCRIPTION_FAILURE_CONTRACT,
    CancellationGranularity,
    ExecutionDevice,
    InputKind,
    LanguageInputMode,
    TimestampGranularity,
    TranscriptionFailureCode,
    TranscriptionRequest,
    TranscriptionTask,
    TranscriptionWarningCode,
)
from .registry import (
    CapabilitySet,
    CatalogDeclarations,
    ModelMetadata,
    ProviderMetadata,
    ProviderRegistry,
    TranscriptionAdapter,
)


_LANGUAGE_PATTERN = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]{1,8})*")
_FASTER_WHISPER_EXPLICIT_LANGUAGES = frozenset(
    """
    af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi
    fo fr gl gu ha haw he hi hr ht hu hy id is it ja jw ka kk km kn ko la lb
    ln lo lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru
    sa sd si sk sl sn so sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi
    yi yo zh yue
    """.split()
)


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    """Immutable semantic-default identities and validated v3 language set."""

    validated_v3_languages: frozenset[str]
    default_provider_id: str = field(default="default", init=False)
    parakeet_provider_id: str = field(default="parakeet-onnx", init=False)
    parakeet_v2_model_id: str = field(
        default="nemo-parakeet-tdt-0.6b-v2",
        init=False,
    )
    parakeet_v3_model_id: str = field(
        default="nemo-parakeet-tdt-0.6b-v3",
        init=False,
    )
    faster_whisper_provider_id: str = field(
        default="faster-whisper",
        init=False,
    )
    faster_whisper_model_id: str = field(default="base", init=False)

    def __post_init__(self) -> None:
        if type(self.validated_v3_languages) is not frozenset:
            raise TypeError("validated_v3_languages must be a frozenset")
        if not self.validated_v3_languages:
            raise ValueError("validated_v3_languages must not be empty")
        for language in self.validated_v3_languages:
            if type(language) is not str:
                raise TypeError("validated_v3_languages must contain only strings")
            if (
                not language
                or language in {"auto", "en"}
                or not _LANGUAGE_PATTERN.fullmatch(language)
            ):
                raise ValueError(
                    "validated_v3_languages must contain canonical lower-case "
                    "non-English explicit language tags"
                )


@dataclass(frozen=True, slots=True)
class ResolvedTranscriptionRequest:
    """A request resolved to one exact provider/model without executing it."""

    request: TranscriptionRequest
    provider_id: str
    model_id: str
    requested_language: str
    effective_language: str
    precision: str
    warning_codes: tuple[TranscriptionWarningCode, ...] = ()

    def __post_init__(self) -> None:
        if type(self.request) is not TranscriptionRequest:
            raise TypeError("request must be a TranscriptionRequest")
        for field_name, value in (
            ("provider_id", self.provider_id),
            ("model_id", self.model_id),
            ("precision", self.precision),
        ):
            if type(value) is not str:
                raise TypeError(f"{field_name} must be a string")
            if not value or value != value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string without "
                    "surrounding whitespace"
                )
        for field_name, value in (
            ("requested_language", self.requested_language),
            ("effective_language", self.effective_language),
        ):
            if type(value) is not str:
                raise TypeError(f"{field_name} must be a string")
            if value != "auto" and not _LANGUAGE_PATTERN.fullmatch(value):
                raise ValueError(
                    f"{field_name} must be 'auto' or a canonical lower-case "
                    "language tag"
                )
        if type(self.warning_codes) is not tuple or not all(
            type(warning) is TranscriptionWarningCode for warning in self.warning_codes
        ):
            raise TypeError(
                "warning_codes must be a tuple of TranscriptionWarningCode values"
            )


class RoutingResolutionError(Exception):
    """A stable typed routing failure that contains no free-form explanation."""

    __slots__ = ("code", "provider_id", "model_id")

    def __init__(
        self,
        code: TranscriptionFailureCode,
        *,
        provider_id: str,
        model_id: str | None,
    ) -> None:
        if type(code) is not TranscriptionFailureCode:
            raise TypeError("code must be a TranscriptionFailureCode")
        self.code = code
        self.provider_id = provider_id
        self.model_id = model_id
        super().__init__(TRANSCRIPTION_FAILURE_CONTRACT[code][0])


def build_builtin_declarations(policy: RoutingPolicy) -> CatalogDeclarations:
    """Build the authoritative immutable built-in provider/model declarations."""

    if type(policy) is not RoutingPolicy:
        raise TypeError("policy must be a RoutingPolicy")

    providers = (
        ProviderMetadata(
            provider_id=policy.parakeet_provider_id,
            display_name="Parakeet ONNX",
            local_processing=True,
        ),
        ProviderMetadata(
            provider_id=policy.faster_whisper_provider_id,
            display_name="faster-whisper",
            local_processing=True,
        ),
    )
    parakeet_common = {
        "tasks": frozenset({TranscriptionTask.TRANSCRIBE}),
        "inputs": frozenset({InputKind.FILE, InputKind.BUFFER}),
        "timestamps": frozenset({TimestampGranularity.NONE}),
        "true_streaming": False,
        "batch": True,
        "cancellation": CancellationGranularity.SEGMENT_BOUNDARY,
        "vad": True,
        "diarization": False,
        "punctuation": True,
        "capitalization": True,
        "execution_devices": frozenset({ExecutionDevice.CPU}),
        "precisions": frozenset({"int8", "f32"}),
    }
    models = (
        ModelMetadata(
            provider_id=policy.parakeet_provider_id,
            model_id=policy.parakeet_v2_model_id,
            display_name="NVIDIA Parakeet TDT 0.6B v2",
            capabilities=CapabilitySet(
                languages=frozenset({"en"}),
                automatic_language=False,
                language_input_mode=LanguageInputMode.ENFORCED,
                **parakeet_common,  # type: ignore[arg-type]
            ),
            default_precision="int8",
            semantic_default_eligible=True,
            enforces_language_hint=True,
        ),
        ModelMetadata(
            provider_id=policy.parakeet_provider_id,
            model_id=policy.parakeet_v3_model_id,
            display_name="NVIDIA Parakeet TDT 0.6B v3",
            capabilities=CapabilitySet(
                languages=policy.validated_v3_languages,
                automatic_language=True,
                language_input_mode=LanguageInputMode.ROUTING_ASSERTION,
                **parakeet_common,  # type: ignore[arg-type]
            ),
            default_precision="int8",
            semantic_default_eligible=True,
            enforces_language_hint=False,
        ),
        ModelMetadata(
            provider_id=policy.faster_whisper_provider_id,
            model_id=policy.faster_whisper_model_id,
            display_name="faster-whisper base",
            capabilities=CapabilitySet(
                languages=_FASTER_WHISPER_EXPLICIT_LANGUAGES,
                automatic_language=True,
                tasks=frozenset(
                    {
                        TranscriptionTask.TRANSCRIBE,
                        TranscriptionTask.TRANSLATE,
                    }
                ),
                inputs=frozenset({InputKind.FILE, InputKind.BUFFER}),
                timestamps=frozenset(
                    {
                        TimestampGranularity.NONE,
                        TimestampGranularity.SEGMENT,
                        TimestampGranularity.WORD,
                    }
                ),
                true_streaming=False,
                batch=True,
                cancellation=CancellationGranularity.SEGMENT_BOUNDARY,
                vad=True,
                diarization=False,
                punctuation=True,
                capitalization=True,
                language_input_mode=LanguageInputMode.AUTOMATIC,
                execution_devices=frozenset(
                    {ExecutionDevice.CPU, ExecutionDevice.CUDA}
                ),
                precisions=frozenset({"int8", "float16", "float32"}),
            ),
            default_precision="int8",
            semantic_default_eligible=True,
            enforces_language_hint=True,
        ),
    )
    return CatalogDeclarations(providers=providers, models=models)


def build_builtin_registry(
    policy: RoutingPolicy,
    *,
    adapters: tuple[TranscriptionAdapter, ...] = (),
    extra_declarations: CatalogDeclarations | None = None,
) -> ProviderRegistry:
    """Seal built-ins plus optional exact declarations and adapters.

    Provider and model identities are concatenated without replacement.
    ``ProviderRegistry.sealed`` therefore rejects every exact collision.
    """

    builtins = build_builtin_declarations(policy)
    if (
        extra_declarations is not None
        and type(extra_declarations) is not CatalogDeclarations
    ):
        raise TypeError("extra_declarations must be a CatalogDeclarations")
    if extra_declarations is None:
        declarations = builtins
    else:
        declarations = CatalogDeclarations(
            providers=builtins.providers + extra_declarations.providers,
            models=builtins.models + extra_declarations.models,
        )
    return ProviderRegistry.sealed(declarations, adapters=adapters)


@dataclass(frozen=True, slots=True)
class TranscriptionRouter:
    """Resolve semantic defaults or validate one exact provider/model choice."""

    policy: RoutingPolicy

    def __post_init__(self) -> None:
        if type(self.policy) is not RoutingPolicy:
            raise TypeError("policy must be a RoutingPolicy")

    def resolve(
        self,
        request: TranscriptionRequest,
        registry: ProviderRegistry,
    ) -> ResolvedTranscriptionRequest:
        """Resolve one request without probing, executing, or retrying."""

        if type(request) is not TranscriptionRequest:
            raise TypeError("request must be a TranscriptionRequest")
        if type(registry) is not ProviderRegistry:
            raise TypeError("registry must be a ProviderRegistry")

        requested_language = request.language or "en"
        semantic_default = request.provider_id == self.policy.default_provider_id
        model_id: str | None
        if semantic_default:
            provider_id, model_id = self._semantic_target(
                requested_language,
                request.task,
            )
        else:
            provider_id = request.provider_id
            model_id = request.model_id

        if registry.provider(provider_id) is None:
            raise RoutingResolutionError(
                TranscriptionFailureCode.PROVIDER_UNAVAILABLE,
                provider_id=provider_id,
                model_id=model_id,
            )
        if model_id is None:
            raise RoutingResolutionError(
                TranscriptionFailureCode.UNSUPPORTED_CAPABILITY,
                provider_id=provider_id,
                model_id=None,
            )
        model = registry.model(provider_id, model_id)
        if model is None:
            raise RoutingResolutionError(
                TranscriptionFailureCode.UNSUPPORTED_CAPABILITY,
                provider_id=provider_id,
                model_id=model_id,
            )
        if semantic_default and not model.semantic_default_eligible:
            self._fail_unsupported_capability(model)

        self._validate_selected_model(
            request,
            requested_language,
            model,
        )
        effective_language, warning_codes = self._language_resolution(
            requested_language,
            model,
        )
        return ResolvedTranscriptionRequest(
            request=request,
            provider_id=provider_id,
            model_id=model_id,
            requested_language=requested_language,
            effective_language=effective_language,
            precision=request.precision or model.default_precision,
            warning_codes=warning_codes,
        )

    def _semantic_target(
        self,
        requested_language: str,
        task: TranscriptionTask,
    ) -> tuple[str, str]:
        if task is TranscriptionTask.TRANSLATE:
            return (
                self.policy.faster_whisper_provider_id,
                self.policy.faster_whisper_model_id,
            )
        if requested_language == "en":
            return (
                self.policy.parakeet_provider_id,
                self.policy.parakeet_v2_model_id,
            )
        if requested_language in self.policy.validated_v3_languages:
            return (
                self.policy.parakeet_provider_id,
                self.policy.parakeet_v3_model_id,
            )
        return (
            self.policy.faster_whisper_provider_id,
            self.policy.faster_whisper_model_id,
        )

    def _validate_selected_model(
        self,
        request: TranscriptionRequest,
        requested_language: str,
        model: ModelMetadata,
    ) -> None:
        capabilities = model.capabilities
        if (
            model.provider_id == self.policy.parakeet_provider_id
            and model.model_id == self.policy.parakeet_v3_model_id
            and (
                capabilities.languages != self.policy.validated_v3_languages
                or not capabilities.automatic_language
                or capabilities.language_input_mode
                is not LanguageInputMode.ROUTING_ASSERTION
                or model.enforces_language_hint
            )
        ):
            self._fail_unsupported_capability(model)

        if request.task not in capabilities.tasks:
            self._fail_unsupported_capability(model)

        if (
            request.precision is not None
            and request.precision not in capabilities.precisions
        ):
            self._fail_unsupported_capability(model)
        if (
            request.device is not ExecutionDevice.AUTO
            and request.device not in capabilities.execution_devices
        ):
            self._fail_unsupported_capability(model)

        if requested_language == "auto":
            if not capabilities.automatic_language:
                self._fail_unsupported_language(model)
            return
        if not self._supports_explicit_language(capabilities, requested_language):
            self._fail_unsupported_language(model)

    @staticmethod
    def _supports_explicit_language(
        capabilities: CapabilitySet,
        language: str,
    ) -> bool:
        return language in capabilities.languages

    @staticmethod
    def _language_resolution(
        requested_language: str,
        model: ModelMetadata,
    ) -> tuple[str, tuple[TranscriptionWarningCode, ...]]:
        if requested_language == "auto":
            return "auto", ()
        if (
            model.capabilities.language_input_mode
            is LanguageInputMode.ROUTING_ASSERTION
        ):
            return (
                "auto",
                (TranscriptionWarningCode.REQUESTED_LANGUAGE_NOT_ENFORCED,),
            )
        return requested_language, ()

    @staticmethod
    def _fail_unsupported_language(model: ModelMetadata) -> None:
        raise RoutingResolutionError(
            TranscriptionFailureCode.UNSUPPORTED_LANGUAGE,
            provider_id=model.provider_id,
            model_id=model.model_id,
        )

    @staticmethod
    def _fail_unsupported_capability(model: ModelMetadata) -> None:
        raise RoutingResolutionError(
            TranscriptionFailureCode.UNSUPPORTED_CAPABILITY,
            provider_id=model.provider_id,
            model_id=model.model_id,
        )


__all__ = [
    "ResolvedTranscriptionRequest",
    "RoutingPolicy",
    "RoutingResolutionError",
    "TranscriptionRouter",
    "build_builtin_declarations",
    "build_builtin_registry",
]
