from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import cast

import pytest

from tldw_chatbook.STT.contracts import (
    TRANSCRIPTION_FAILURE_CONTRACT,
    BufferAudioSource,
    CancellationGranularity,
    ExecutionDevice,
    InputKind,
    LanguageInputMode,
    ResolvedTranscriptionRequest,
    TimestampGranularity,
    TranscriptionFailureCode,
    TranscriptionRequest,
    TranscriptionTask,
    TranscriptionWarningCode,
)
from tldw_chatbook.STT.registry import (
    CapabilitySet,
    CatalogDeclarations,
    DuplicateDeclarationError,
    ModelMetadata,
    ProviderMetadata,
    ProviderRegistry,
)
from tldw_chatbook.STT.routing import (
    RoutingPolicy,
    RoutingResolutionError,
    TranscriptionRouter,
    build_builtin_declarations,
    build_builtin_registry,
)


VALIDATED_V3_LANGUAGES = frozenset({"es", "fr"})
EXPECTED_FASTER_WHISPER_BASE_LANGUAGES = frozenset(
    """
    af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi
    fo fr gl gu ha haw he hi hr ht hu hy id is it ja jw ka kk km kn ko la lb
    ln lo lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru
    sa sd si sk sl sn so sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi
    yi yo zh
    """.split()
)


def _request(**overrides: object) -> TranscriptionRequest:
    values: dict[str, object] = {
        "attempt_id": "attempt-1",
        "source": BufferAudioSource(b"\x00\x00", 16_000),
        "timestamps": TimestampGranularity.NONE,
    }
    values.update(overrides)
    return TranscriptionRequest(**values)  # type: ignore[arg-type]


def _policy() -> RoutingPolicy:
    return RoutingPolicy(validated_v3_languages=VALIDATED_V3_LANGUAGES)


def _registry(policy: RoutingPolicy | None = None) -> ProviderRegistry:
    return build_builtin_registry(policy or _policy())


def _replace_builtin_model(
    policy: RoutingPolicy,
    provider_id: str,
    model_id: str,
    replacement: ModelMetadata,
) -> ProviderRegistry:
    declarations = build_builtin_declarations(policy)
    models = tuple(
        replacement
        if (model.provider_id, model.model_id) == (provider_id, model_id)
        else model
        for model in declarations.models
    )
    return ProviderRegistry.sealed(
        CatalogDeclarations(providers=declarations.providers, models=models)
    )


@pytest.mark.parametrize("language", [None, ""])
def test_semantic_default_normalizes_omitted_or_empty_language_to_english(
    language: str | None,
) -> None:
    policy = _policy()
    request = _request(language=language)

    resolved = TranscriptionRouter(policy).resolve(request, _registry(policy))

    assert resolved.request is request
    assert resolved.provider_id == policy.parakeet_provider_id
    assert resolved.model_id == policy.parakeet_v2_model_id
    assert resolved.requested_language == "en"
    assert resolved.effective_language == "en"
    assert resolved.warning_codes == ()
    assert resolved.precision == "int8"


def test_semantic_default_routes_explicit_english_to_parakeet_v2() -> None:
    policy = _policy()

    resolved = TranscriptionRouter(policy).resolve(
        _request(language="en"),
        _registry(policy),
    )

    assert (resolved.provider_id, resolved.model_id) == (
        "parakeet-onnx",
        "nemo-parakeet-tdt-0.6b-v2",
    )
    assert (resolved.requested_language, resolved.effective_language) == ("en", "en")
    assert resolved.warning_codes == ()


def test_semantic_default_routes_validated_non_english_to_parakeet_v3() -> None:
    policy = _policy()

    resolved = TranscriptionRouter(policy).resolve(
        _request(language="es"),
        _registry(policy),
    )

    assert (resolved.provider_id, resolved.model_id) == (
        "parakeet-onnx",
        "nemo-parakeet-tdt-0.6b-v3",
    )
    assert resolved.requested_language == "es"
    assert resolved.effective_language == "auto"
    assert resolved.warning_codes == (
        TranscriptionWarningCode.REQUESTED_LANGUAGE_NOT_ENFORCED,
    )


@pytest.mark.parametrize(
    ("language", "task"),
    [
        ("auto", TranscriptionTask.TRANSCRIBE),
        ("ja", TranscriptionTask.TRANSCRIBE),
        ("en", TranscriptionTask.TRANSLATE),
        ("es", TranscriptionTask.TRANSLATE),
    ],
)
def test_semantic_default_routes_auto_unsupported_languages_and_translation_to_faster_whisper(
    language: str,
    task: TranscriptionTask,
) -> None:
    policy = _policy()

    resolved = TranscriptionRouter(policy).resolve(
        _request(language=language, task=task),
        _registry(policy),
    )

    assert (resolved.provider_id, resolved.model_id) == (
        "faster-whisper",
        "base",
    )
    assert resolved.requested_language == language
    assert resolved.effective_language == language
    assert resolved.warning_codes == ()


def test_english_routes_to_v2_even_if_v3_metadata_accidentally_contains_english() -> (
    None
):
    policy = _policy()
    declarations = build_builtin_declarations(policy)
    v3 = next(
        model
        for model in declarations.models
        if model.model_id == policy.parakeet_v3_model_id
    )
    registry = _replace_builtin_model(
        policy,
        policy.parakeet_provider_id,
        policy.parakeet_v3_model_id,
        replace(
            v3,
            capabilities=replace(
                v3.capabilities,
                languages=frozenset({"en", *VALIDATED_V3_LANGUAGES}),
            ),
        ),
    )

    resolved = TranscriptionRouter(policy).resolve(
        _request(language="en"),
        registry,
    )

    assert resolved.model_id == policy.parakeet_v2_model_id


def test_exact_compatible_provider_and_model_are_preserved() -> None:
    policy = _policy()
    request = _request(
        provider_id="faster-whisper",
        model_id="base",
        language="ja",
        timestamps=TimestampGranularity.WORD,
        precision="float32",
        device=ExecutionDevice.CUDA,
    )

    resolved = TranscriptionRouter(policy).resolve(request, _registry(policy))

    assert resolved.request is request
    assert (resolved.provider_id, resolved.model_id) == (
        request.provider_id,
        request.model_id,
    )
    assert (resolved.requested_language, resolved.effective_language) == ("ja", "ja")
    assert resolved.precision == "float32"
    assert resolved.warning_codes == ()


def test_exact_faster_whisper_base_rejects_cpu_float16_pair() -> None:
    policy = _policy()

    with pytest.raises(RoutingResolutionError) as caught:
        TranscriptionRouter(policy).resolve(
            _request(
                provider_id=policy.faster_whisper_provider_id,
                model_id=policy.faster_whisper_model_id,
                language="en",
                precision="float16",
                device=ExecutionDevice.CPU,
            ),
            _registry(policy),
        )

    assert caught.value.code is TranscriptionFailureCode.UNSUPPORTED_CAPABILITY
    assert caught.value.provider_id == policy.faster_whisper_provider_id
    assert caught.value.model_id == policy.faster_whisper_model_id


def test_exact_parakeet_v3_preserves_requested_language_as_routing_assertion() -> None:
    policy = _policy()

    resolved = TranscriptionRouter(policy).resolve(
        _request(
            provider_id=policy.parakeet_provider_id,
            model_id=policy.parakeet_v3_model_id,
            language="fr",
        ),
        _registry(policy),
    )

    assert resolved.provider_id == policy.parakeet_provider_id
    assert resolved.model_id == policy.parakeet_v3_model_id
    assert resolved.requested_language == "fr"
    assert resolved.effective_language == "auto"
    assert resolved.warning_codes == (
        TranscriptionWarningCode.REQUESTED_LANGUAGE_NOT_ENFORCED,
    )


@pytest.mark.parametrize(
    ("model_id", "language"),
    [
        ("nemo-parakeet-tdt-0.6b-v2", "fr"),
        ("nemo-parakeet-tdt-0.6b-v3", "ja"),
        ("nemo-parakeet-tdt-0.6b-v2", "auto"),
    ],
)
def test_exact_parakeet_unsupported_language_fails_without_engine_switch(
    model_id: str,
    language: str,
) -> None:
    policy = _policy()

    with pytest.raises(RoutingResolutionError) as caught:
        TranscriptionRouter(policy).resolve(
            _request(
                provider_id=policy.parakeet_provider_id,
                model_id=model_id,
                language=language,
            ),
            _registry(policy),
        )

    assert caught.value.code is TranscriptionFailureCode.UNSUPPORTED_LANGUAGE
    assert caught.value.provider_id == policy.parakeet_provider_id
    assert caught.value.model_id == model_id
    assert str(caught.value) == TRANSCRIPTION_FAILURE_CONTRACT[caught.value.code][0]


@pytest.mark.parametrize(
    "request_overrides",
    [
        {"task": TranscriptionTask.TRANSLATE},
        {"precision": "float16"},
        {"device": ExecutionDevice.CUDA},
    ],
)
def test_exact_parakeet_unsupported_capability_fails_without_engine_switch(
    request_overrides: dict[str, object],
) -> None:
    policy = _policy()

    with pytest.raises(RoutingResolutionError) as caught:
        TranscriptionRouter(policy).resolve(
            _request(
                provider_id=policy.parakeet_provider_id,
                model_id=policy.parakeet_v2_model_id,
                language="en",
                **request_overrides,
            ),
            _registry(policy),
        )

    assert caught.value.code is TranscriptionFailureCode.UNSUPPORTED_CAPABILITY
    assert caught.value.provider_id == policy.parakeet_provider_id
    assert caught.value.model_id == policy.parakeet_v2_model_id


def test_exact_model_defers_timestamp_and_input_pipeline_compatibility() -> None:
    policy = _policy()
    declarations = build_builtin_declarations(policy)
    v2 = next(
        model
        for model in declarations.models
        if model.model_id == policy.parakeet_v2_model_id
    )
    registry = _replace_builtin_model(
        policy,
        policy.parakeet_provider_id,
        policy.parakeet_v2_model_id,
        replace(
            v2,
            capabilities=replace(
                v2.capabilities,
                inputs=frozenset({InputKind.FILE}),
            ),
        ),
    )

    resolved = TranscriptionRouter(policy).resolve(
        _request(
            provider_id=policy.parakeet_provider_id,
            model_id=policy.parakeet_v2_model_id,
            language="en",
            timestamps=TimestampGranularity.SEGMENT,
        ),
        registry,
    )

    assert resolved.provider_id == policy.parakeet_provider_id
    assert resolved.model_id == policy.parakeet_v2_model_id


@pytest.mark.parametrize(
    "declared_languages",
    [frozenset(), frozenset({"en"})],
)
def test_automatic_only_exact_model_rejects_explicit_language_without_dropping_it(
    declared_languages: frozenset[str],
) -> None:
    policy = _policy()
    provider = ProviderMetadata(
        provider_id="transcribe-cpp",
        display_name="transcribe.cpp",
        local_processing=True,
    )
    model = ModelMetadata(
        provider_id=provider.provider_id,
        model_id="qwen3-asr-0.6b-q8_0",
        display_name="Qwen3-ASR 0.6B Q8_0",
        capabilities=CapabilitySet(
            languages=declared_languages,
            automatic_language=True,
            tasks=frozenset({TranscriptionTask.TRANSCRIBE}),
            inputs=frozenset({InputKind.FILE, InputKind.BUFFER}),
            timestamps=frozenset({TimestampGranularity.NONE}),
            true_streaming=False,
            batch=True,
            cancellation=CancellationGranularity.SEGMENT_BOUNDARY,
            vad=True,
            diarization=False,
            punctuation=True,
            capitalization=True,
            language_input_mode=LanguageInputMode.AUTOMATIC_ONLY,
            execution_devices=frozenset({ExecutionDevice.CPU}),
            precisions=frozenset({"q8_0"}),
        ),
        default_precision="q8_0",
        semantic_default_eligible=False,
        enforces_language_hint=False,
    )
    registry = build_builtin_registry(
        policy,
        extra_declarations=CatalogDeclarations(
            providers=(provider,),
            models=(model,),
        ),
    )

    with pytest.raises(RoutingResolutionError) as caught:
        TranscriptionRouter(policy).resolve(
            _request(
                provider_id=provider.provider_id,
                model_id=model.model_id,
                language="en",
            ),
            registry,
        )

    assert caught.value.code is TranscriptionFailureCode.UNSUPPORTED_LANGUAGE
    assert caught.value.provider_id == provider.provider_id
    assert caught.value.model_id == model.model_id

    resolved = TranscriptionRouter(policy).resolve(
        _request(
            provider_id=provider.provider_id,
            model_id=model.model_id,
            language="auto",
        ),
        registry,
    )
    assert resolved.requested_language == "auto"
    assert resolved.effective_language == "auto"


@pytest.mark.parametrize(
    ("provider_id", "model_id", "expected_code"),
    [
        (
            "parakeet",
            "nemo-parakeet-tdt-0.6b-v2",
            TranscriptionFailureCode.PROVIDER_UNAVAILABLE,
        ),
        (
            "PARAKEET-ONNX",
            "nemo-parakeet-tdt-0.6b-v2",
            TranscriptionFailureCode.PROVIDER_UNAVAILABLE,
        ),
        (
            "parakeet-onnx",
            "nemo-parakeet",
            TranscriptionFailureCode.UNSUPPORTED_CAPABILITY,
        ),
        (
            "faster-whisper",
            "Base",
            TranscriptionFailureCode.UNSUPPORTED_CAPABILITY,
        ),
    ],
)
def test_unknown_exact_identity_fails_without_alias_prefix_or_case_matching(
    provider_id: str,
    model_id: str,
    expected_code: TranscriptionFailureCode,
) -> None:
    policy = _policy()

    with pytest.raises(RoutingResolutionError) as caught:
        TranscriptionRouter(policy).resolve(
            _request(
                provider_id=provider_id,
                model_id=model_id,
                language="en",
            ),
            _registry(policy),
        )

    assert caught.value.code is expected_code
    assert caught.value.provider_id == provider_id
    assert caught.value.model_id == model_id
    assert str(caught.value) == TRANSCRIPTION_FAILURE_CONTRACT[expected_code][0]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("validated_v3_languages", {"es"}),
        ("validated_v3_languages", frozenset()),
        ("validated_v3_languages", frozenset({""})),
        ("validated_v3_languages", frozenset({"auto"})),
        ("validated_v3_languages", frozenset({"en"})),
        ("validated_v3_languages", frozenset({"ES"})),
        ("validated_v3_languages", frozenset({"es_419"})),
    ],
)
def test_routing_policy_rejects_mutable_empty_or_noncanonical_v3_language_sets(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RoutingPolicy(**{field_name: value})  # type: ignore[arg-type]


def test_routing_policy_and_resolved_request_are_frozen_and_slotted() -> None:
    policy = _policy()
    resolved = TranscriptionRouter(policy).resolve(
        _request(language="en"),
        _registry(policy),
    )

    assert type(resolved) is ResolvedTranscriptionRequest
    assert not hasattr(policy, "__dict__")
    assert not hasattr(resolved, "__dict__")
    with pytest.raises(FrozenInstanceError):
        policy.validated_v3_languages = frozenset({"de"})  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        resolved.model_id = "other"  # type: ignore[misc]


def test_builtin_metadata_matches_the_authoritative_capability_matrix() -> None:
    policy = _policy()
    declarations = build_builtin_declarations(policy)
    providers = {provider.provider_id: provider for provider in declarations.providers}
    models = {
        (model.provider_id, model.model_id): model for model in declarations.models
    }

    assert providers == {
        "parakeet-onnx": ProviderMetadata(
            provider_id="parakeet-onnx",
            display_name="Parakeet ONNX",
            local_processing=True,
        ),
        "faster-whisper": ProviderMetadata(
            provider_id="faster-whisper",
            display_name="faster-whisper",
            local_processing=True,
        ),
    }
    assert models == {
        ("parakeet-onnx", "nemo-parakeet-tdt-0.6b-v2"): ModelMetadata(
            provider_id="parakeet-onnx",
            model_id="nemo-parakeet-tdt-0.6b-v2",
            display_name="NVIDIA Parakeet TDT 0.6B v2",
            capabilities=CapabilitySet(
                languages=frozenset({"en"}),
                automatic_language=False,
                tasks=frozenset({TranscriptionTask.TRANSCRIBE}),
                inputs=frozenset({InputKind.FILE, InputKind.BUFFER}),
                timestamps=frozenset({TimestampGranularity.NONE}),
                true_streaming=False,
                batch=True,
                cancellation=CancellationGranularity.SEGMENT_BOUNDARY,
                vad=True,
                diarization=False,
                punctuation=True,
                capitalization=True,
                language_input_mode=LanguageInputMode.ENFORCED,
                execution_devices=frozenset({ExecutionDevice.CPU}),
                precisions=frozenset({"int8", "f32"}),
            ),
            default_precision="int8",
            semantic_default_eligible=True,
            enforces_language_hint=True,
        ),
        ("parakeet-onnx", "nemo-parakeet-tdt-0.6b-v3"): ModelMetadata(
            provider_id="parakeet-onnx",
            model_id="nemo-parakeet-tdt-0.6b-v3",
            display_name="NVIDIA Parakeet TDT 0.6B v3",
            capabilities=CapabilitySet(
                languages=VALIDATED_V3_LANGUAGES,
                automatic_language=True,
                tasks=frozenset({TranscriptionTask.TRANSCRIBE}),
                inputs=frozenset({InputKind.FILE, InputKind.BUFFER}),
                timestamps=frozenset({TimestampGranularity.NONE}),
                true_streaming=False,
                batch=True,
                cancellation=CancellationGranularity.SEGMENT_BOUNDARY,
                vad=True,
                diarization=False,
                punctuation=True,
                capitalization=True,
                language_input_mode=LanguageInputMode.ROUTING_ASSERTION,
                execution_devices=frozenset({ExecutionDevice.CPU}),
                precisions=frozenset({"int8", "f32"}),
            ),
            default_precision="int8",
            semantic_default_eligible=True,
            enforces_language_hint=False,
        ),
        ("faster-whisper", "base"): ModelMetadata(
            provider_id="faster-whisper",
            model_id="base",
            display_name="faster-whisper base",
            capabilities=CapabilitySet(
                languages=EXPECTED_FASTER_WHISPER_BASE_LANGUAGES,
                automatic_language=True,
                tasks=frozenset(
                    {TranscriptionTask.TRANSCRIBE, TranscriptionTask.TRANSLATE}
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
                # The current independent device/precision sets cannot safely
                # express CUDA-only float16, so base advertises the pairs that
                # are valid across its declared CPU/CUDA devices.
                precisions=frozenset({"int8", "float32"}),
            ),
            default_precision="int8",
            semantic_default_eligible=True,
            enforces_language_hint=True,
        ),
    }


def test_faster_whisper_uses_finite_broad_explicit_language_support() -> None:
    policy = _policy()
    registry = _registry(policy)
    model = registry.model(
        policy.faster_whisper_provider_id,
        policy.faster_whisper_model_id,
    )

    assert model is not None
    assert model.capabilities.languages == EXPECTED_FASTER_WHISPER_BASE_LANGUAGES
    assert model.capabilities.language_input_mode is LanguageInputMode.AUTOMATIC
    assert model.capabilities.automatic_language

    resolved = TranscriptionRouter(policy).resolve(
        _request(
            provider_id=policy.faster_whisper_provider_id,
            model_id=policy.faster_whisper_model_id,
            language="ja",
        ),
        registry,
    )
    assert resolved.requested_language == "ja"
    assert resolved.effective_language == "ja"

    with pytest.raises(RoutingResolutionError) as caught:
        TranscriptionRouter(policy).resolve(
            _request(
                provider_id=policy.faster_whisper_provider_id,
                model_id=policy.faster_whisper_model_id,
                language="zh-hant",
            ),
            registry,
        )
    assert caught.value.code is TranscriptionFailureCode.UNSUPPORTED_LANGUAGE


def test_semantic_default_yue_fails_closed_for_faster_whisper_base() -> None:
    policy = _policy()

    with pytest.raises(RoutingResolutionError) as caught:
        TranscriptionRouter(policy).resolve(
            _request(language="yue"),
            _registry(policy),
        )

    assert caught.value.code is TranscriptionFailureCode.UNSUPPORTED_LANGUAGE
    assert caught.value.provider_id == policy.faster_whisper_provider_id
    assert caught.value.model_id == policy.faster_whisper_model_id
    assert not hasattr(caught.value, "effective_language")


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("language_input_mode", LanguageInputMode.ENFORCED),
        ("automatic_language", False),
        ("enforces_language_hint", True),
    ],
)
def test_parakeet_v3_route_fails_closed_when_language_metadata_is_unsafe(
    field_name: str,
    replacement: object,
) -> None:
    policy = _policy()
    declarations = build_builtin_declarations(policy)
    v3 = next(
        model
        for model in declarations.models
        if model.model_id == policy.parakeet_v3_model_id
    )
    if field_name == "enforces_language_hint":
        unsafe_v3 = replace(
            v3,
            enforces_language_hint=cast(bool, replacement),
        )
    elif field_name == "automatic_language":
        unsafe_v3 = replace(
            v3,
            capabilities=replace(
                v3.capabilities,
                automatic_language=cast(bool, replacement),
            ),
        )
    else:
        unsafe_v3 = replace(
            v3,
            capabilities=replace(
                v3.capabilities,
                language_input_mode=cast(LanguageInputMode, replacement),
            ),
        )
    registry = _replace_builtin_model(
        policy,
        policy.parakeet_provider_id,
        policy.parakeet_v3_model_id,
        unsafe_v3,
    )

    with pytest.raises(RoutingResolutionError) as caught:
        TranscriptionRouter(policy).resolve(
            _request(language="es"),
            registry,
        )

    assert caught.value.code is TranscriptionFailureCode.UNSUPPORTED_CAPABILITY


def test_builtin_declarations_and_sealed_registry_snapshots_cannot_mutate() -> None:
    policy = _policy()
    declarations = build_builtin_declarations(policy)
    registry = build_builtin_registry(policy)
    declaration_model = declarations.models[0]
    mutable_view = cast(
        dict[tuple[str, str], ModelMetadata],
        registry.models,
    )

    with pytest.raises(TypeError):
        mutable_view[(policy.parakeet_provider_id, policy.parakeet_v2_model_id)] = (
            declaration_model
        )
    with pytest.raises(FrozenInstanceError):
        declarations.models = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        declaration_model.default_precision = "f32"  # type: ignore[misc]


def test_registry_builder_merges_exact_extra_declarations_and_rejects_collisions() -> (
    None
):
    policy = _policy()
    provider = ProviderMetadata(
        provider_id="exact-extra",
        display_name="Exact Extra",
        local_processing=True,
    )
    v2 = build_builtin_declarations(policy).models[0]
    model = replace(
        v2,
        provider_id=provider.provider_id,
        model_id="exact-model",
        semantic_default_eligible=False,
    )
    registry = build_builtin_registry(
        policy,
        extra_declarations=CatalogDeclarations(
            providers=(provider,),
            models=(model,),
        ),
    )

    assert registry.provider(provider.provider_id) is provider
    assert registry.model(provider.provider_id, model.model_id) is model

    colliding_provider = replace(provider, provider_id=policy.parakeet_provider_id)
    with pytest.raises(DuplicateDeclarationError):
        build_builtin_registry(
            policy,
            extra_declarations=CatalogDeclarations(
                providers=(colliding_provider,),
                models=(
                    replace(
                        model,
                        provider_id=policy.parakeet_provider_id,
                    ),
                ),
            ),
        )


def test_noncanonical_language_is_rejected_by_request_before_routing() -> None:
    with pytest.raises(ValueError, match="canonical lower-case language tag"):
        _request(language="EN")
