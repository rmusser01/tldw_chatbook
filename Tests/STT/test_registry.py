from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from types import MappingProxyType
from typing import get_type_hints

import pytest

from tldw_chatbook.STT.contracts import (
    BufferAudioSource,
    CancellationGranularity,
    ExecutionDevice,
    InputKind,
    LanguageInputMode,
    ProducedCapabilities,
    ResolvedTranscriptionRequest,
    TimestampGranularity,
    TranscriptionRequest,
    TranscriptionSegment,
    TranscriptionTask,
    TranscriptionTimings,
)
from tldw_chatbook.STT.registry import (
    AdapterRegistrationError,
    CapabilitySet,
    CatalogDeclarationError,
    CatalogDeclarations,
    DuplicateAdapterError,
    DuplicateDeclarationError,
    ModelMetadata,
    ProviderMetadata,
    ProviderRegistry,
    ProviderTranscriptionOutput,
    RuntimeCapabilityError,
    RuntimeObservation,
    TranscriptionAdapter,
)


def _capabilities(**overrides: object) -> CapabilitySet:
    values: dict[str, object] = {
        "languages": frozenset({"en", "fr"}),
        "automatic_language": True,
        "tasks": frozenset({TranscriptionTask.TRANSCRIBE, TranscriptionTask.TRANSLATE}),
        "inputs": frozenset({InputKind.FILE, InputKind.BUFFER}),
        "timestamps": frozenset(
            {
                TimestampGranularity.NONE,
                TimestampGranularity.SEGMENT,
                TimestampGranularity.WORD,
            }
        ),
        "true_streaming": True,
        "batch": True,
        "cancellation": CancellationGranularity.ACTIVE,
        "vad": True,
        "diarization": True,
        "punctuation": True,
        "capitalization": True,
        "language_input_mode": LanguageInputMode.AUTOMATIC,
        "execution_devices": frozenset({ExecutionDevice.CPU, ExecutionDevice.CUDA}),
        "precisions": frozenset({"int8", "fp32"}),
    }
    values.update(overrides)
    return CapabilitySet(**values)  # type: ignore[arg-type]


def _provider(**overrides: object) -> ProviderMetadata:
    values: dict[str, object] = {
        "provider_id": "test-provider",
        "display_name": "Test Provider",
        "local_processing": True,
    }
    values.update(overrides)
    return ProviderMetadata(**values)  # type: ignore[arg-type]


def _model(**overrides: object) -> ModelMetadata:
    values: dict[str, object] = {
        "provider_id": "test-provider",
        "model_id": "model-a",
        "display_name": "Model A",
        "capabilities": _capabilities(),
        "default_precision": "int8",
        "semantic_default_eligible": False,
        "enforces_language_hint": True,
    }
    values.update(overrides)
    return ModelMetadata(**values)  # type: ignore[arg-type]


def _declarations(
    *,
    providers: tuple[ProviderMetadata, ...] | None = None,
    models: tuple[ModelMetadata, ...] | None = None,
) -> CatalogDeclarations:
    return CatalogDeclarations(
        providers=providers if providers is not None else (_provider(),),
        models=models if models is not None else (_model(),),
    )


class _Adapter:
    def __init__(
        self,
        provider: ProviderMetadata,
        models: tuple[ModelMetadata, ...],
    ) -> None:
        self._provider = provider
        self._models = models

    def provider(self) -> ProviderMetadata:
        return self._provider

    def describe(self) -> tuple[ModelMetadata, ...]:
        return self._models

    def probe(self, model_id: str) -> RuntimeObservation:
        model = next(model for model in self._models if model.model_id == model_id)
        return RuntimeObservation(
            provider_id=model.provider_id,
            model_id=model.model_id,
            available=True,
            capabilities=model.capabilities,
        )

    def transcribe(
        self,
        request: ResolvedTranscriptionRequest,
    ) -> ProviderTranscriptionOutput:
        del request
        return _provider_output()

    def close(self) -> None:
        return None


def _provider_output(**overrides: object) -> ProviderTranscriptionOutput:
    values: dict[str, object] = {
        "text": "hello",
        "segments": (TranscriptionSegment(0.0, 1.0, "hello"),),
        "effective_language": "en",
        "detected_language": None,
        "effective_device": ExecutionDevice.CPU,
        "produced_capabilities": ProducedCapabilities(
            timestamps=TimestampGranularity.SEGMENT,
            punctuation=True,
            capitalization=True,
            vad=False,
            diarization=False,
        ),
        "duration_seconds": 1.0,
        "timings": TranscriptionTimings(total_seconds=0.5),
        "warnings": (),
    }
    values.update(overrides)
    return ProviderTranscriptionOutput(**values)  # type: ignore[arg-type]


def test_capability_set_has_exact_frozen_slotted_shape() -> None:
    capabilities = _capabilities()

    assert tuple(field.name for field in fields(capabilities)) == (
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
        "execution_devices",
        "precisions",
    )
    assert not hasattr(capabilities, "__dict__")
    assert all(
        isinstance(value, frozenset)
        for value in (
            capabilities.languages,
            capabilities.tasks,
            capabilities.inputs,
            capabilities.timestamps,
            capabilities.execution_devices,
            capabilities.precisions,
        )
    )
    with pytest.raises(FrozenInstanceError):
        capabilities.batch = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("languages", {"en"}),
        ("languages", frozenset({"EN"})),
        ("languages", frozenset({"auto"})),
        ("automatic_language", 1),
        ("tasks", {TranscriptionTask.TRANSCRIBE}),
        ("tasks", frozenset()),
        ("tasks", frozenset({"transcribe"})),
        ("inputs", {InputKind.FILE}),
        ("inputs", frozenset()),
        ("inputs", frozenset({"file"})),
        ("timestamps", {TimestampGranularity.NONE}),
        ("timestamps", frozenset()),
        ("timestamps", frozenset({"none"})),
        ("true_streaming", 0),
        ("batch", 1),
        ("cancellation", "active"),
        ("vad", 1),
        ("diarization", 0),
        ("punctuation", 1),
        ("capitalization", 0),
        ("language_input_mode", "automatic"),
        ("execution_devices", {ExecutionDevice.CPU}),
        ("execution_devices", frozenset()),
        ("execution_devices", frozenset({ExecutionDevice.AUTO})),
        ("execution_devices", frozenset({"cpu"})),
        ("precisions", {"int8"}),
        ("precisions", frozenset()),
        ("precisions", frozenset({""})),
    ],
)
def test_capability_set_rejects_malformed_fields(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _capabilities(**{field_name: value})


@pytest.mark.parametrize(
    "metadata",
    [
        _provider(),
        _model(),
        _declarations(),
        RuntimeObservation(
            provider_id="test-provider",
            model_id="model-a",
            available=True,
            capabilities=_capabilities(),
        ),
    ],
)
def test_registry_metadata_values_are_frozen_and_slotted(metadata: object) -> None:
    assert not hasattr(metadata, "__dict__")
    field_name = fields(metadata)[0].name  # type: ignore[arg-type]
    with pytest.raises(FrozenInstanceError):
        setattr(metadata, field_name, object())


@pytest.mark.parametrize(
    ("factory", "overrides"),
    [
        (_provider, {"provider_id": ""}),
        (_provider, {"provider_id": " test-provider"}),
        (_provider, {"display_name": " "}),
        (_provider, {"local_processing": 1}),
        (_model, {"provider_id": ""}),
        (_model, {"model_id": "model-a "}),
        (_model, {"display_name": ""}),
        (_model, {"capabilities": object()}),
        (_model, {"default_precision": "fp16"}),
        (_model, {"semantic_default_eligible": 1}),
        (_model, {"enforces_language_hint": 1}),
    ],
)
def test_provider_and_model_metadata_reject_malformed_values(
    factory: object,
    overrides: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory(**overrides)  # type: ignore[operator]


@pytest.mark.parametrize(
    ("providers", "models"),
    [
        ([], (_model(),)),
        ((_provider(),), [_model()]),
        ((), ()),
        ((_provider(),), ()),
        ((object(),), (_model(),)),
        ((_provider(),), (object(),)),
    ],
)
def test_catalog_declarations_reject_empty_or_malformed_values(
    providers: object,
    models: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        CatalogDeclarations(providers=providers, models=models)  # type: ignore[arg-type]


def test_provider_transcription_output_is_dependency_free_and_immutable() -> None:
    output = _provider_output()

    assert output.text == "hello"
    assert type(output.segments) is tuple
    assert type(output.warnings) is tuple
    assert not hasattr(output, "__dict__")
    with pytest.raises(FrozenInstanceError):
        output.text = "changed"  # type: ignore[misc]


def test_sealing_rejects_duplicate_provider_ids() -> None:
    declarations = _declarations(
        providers=(_provider(), _provider(display_name="Duplicate")),
    )

    with pytest.raises(DuplicateDeclarationError, match="test-provider"):
        ProviderRegistry.sealed(declarations)


def test_sealing_rejects_duplicate_exact_model_ids() -> None:
    declarations = _declarations(
        models=(_model(), _model(display_name="Duplicate")),
    )

    with pytest.raises(DuplicateDeclarationError, match="model-a"):
        ProviderRegistry.sealed(declarations)


def test_sealing_allows_same_model_id_for_different_providers() -> None:
    second_provider = _provider(provider_id="other-provider")
    second_model = _model(provider_id="other-provider")

    registry = ProviderRegistry.sealed(
        _declarations(
            providers=(_provider(), second_provider),
            models=(_model(), second_model),
        )
    )

    assert registry.model("test-provider", "model-a") == _model()
    assert registry.model("other-provider", "model-a") == second_model


def test_sealing_rejects_model_referencing_undeclared_provider() -> None:
    declarations = _declarations(
        models=(_model(provider_id="missing-provider"),),
    )

    with pytest.raises(CatalogDeclarationError, match="missing-provider"):
        ProviderRegistry.sealed(declarations)


def test_catalog_only_declarations_are_resolvable_without_adapter() -> None:
    declarations = _declarations()

    registry = ProviderRegistry.sealed(declarations)

    assert registry.provider("test-provider") == _provider()
    assert registry.model("test-provider", "model-a") == _model()
    assert registry.adapter("test-provider") is None


def test_exact_lookups_do_not_match_aliases_prefixes_or_wildcards() -> None:
    registry = ProviderRegistry.sealed(_declarations())

    assert registry.provider("test") is None
    assert registry.provider("test-*") is None
    assert registry.provider("TEST-PROVIDER") is None
    assert registry.model("test-provider", "model") is None
    assert registry.model("test-provider", "model-*") is None
    assert registry.model("test", "model-a") is None
    assert registry.adapter("test-*") is None


def test_exposed_registry_collections_and_registry_are_immutable() -> None:
    registry = ProviderRegistry.sealed(_declarations())

    assert registry.declarations == _declarations()
    assert isinstance(registry.providers, MappingProxyType)
    assert isinstance(registry.models, MappingProxyType)
    assert isinstance(registry.adapters, MappingProxyType)
    with pytest.raises(TypeError):
        registry.providers["other"] = _provider()  # type: ignore[index]
    with pytest.raises(TypeError):
        registry.models[("test-provider", "other")] = _model()  # type: ignore[index]
    with pytest.raises(TypeError):
        registry.adapters["test-provider"] = object()  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        registry.declarations = _declarations()  # type: ignore[misc]


def test_registry_can_only_be_built_by_sealing_declarations() -> None:
    with pytest.raises(TypeError):
        ProviderRegistry(  # type: ignore[call-arg]
            declarations=_declarations(),
            _providers={},
            _models={},
            _adapters={},
        )


def test_sealing_rejects_duplicate_adapters_for_exact_provider() -> None:
    declarations = _declarations()
    first = _Adapter(_provider(), (_model(),))
    second = _Adapter(_provider(), (_model(),))

    with pytest.raises(DuplicateAdapterError, match="test-provider"):
        ProviderRegistry.sealed(declarations, adapters=(first, second))


def test_sealing_rejects_adapter_for_undeclared_provider() -> None:
    adapter = _Adapter(
        _provider(provider_id="missing-provider"),
        (_model(provider_id="missing-provider"),),
    )

    with pytest.raises(AdapterRegistrationError, match="missing-provider"):
        ProviderRegistry.sealed(_declarations(), adapters=(adapter,))


def test_sealing_rejects_adapter_provider_metadata_mismatch() -> None:
    adapter = _Adapter(
        _provider(display_name="Different Name"),
        (_model(),),
    )

    with pytest.raises(AdapterRegistrationError, match="metadata"):
        ProviderRegistry.sealed(_declarations(), adapters=(adapter,))


def test_sealing_rejects_adapter_for_undeclared_model() -> None:
    adapter = _Adapter(
        _provider(),
        (_model(model_id="missing-model"),),
    )

    with pytest.raises(AdapterRegistrationError, match="missing-model"):
        ProviderRegistry.sealed(_declarations(), adapters=(adapter,))


def test_sealing_rejects_adapter_model_metadata_mismatch() -> None:
    adapter = _Adapter(
        _provider(),
        (_model(default_precision="fp32"),),
    )

    with pytest.raises(AdapterRegistrationError, match="metadata"):
        ProviderRegistry.sealed(_declarations(), adapters=(adapter,))


def test_adapter_may_serve_an_exact_declared_model_subset() -> None:
    model_b = _model(model_id="model-b", display_name="Model B")
    adapter = _Adapter(_provider(), (_model(),))

    registry = ProviderRegistry.sealed(
        _declarations(models=(_model(), model_b)),
        adapters=(adapter,),
    )

    assert registry.adapter("test-provider") is adapter
    assert registry.adapter_for_model("test-provider", "model-a") is adapter
    assert registry.adapter_for_model("test-provider", "model-b") is None
    assert registry.model("test-provider", "model-b") == model_b


def test_model_adapter_lookup_uses_the_immutable_sealed_snapshot() -> None:
    model_a = _model()
    model_b = _model(model_id="model-b", display_name="Model B")
    adapter = _Adapter(_provider(), (model_a,))
    registry = ProviderRegistry.sealed(
        _declarations(models=(model_a, model_b)),
        adapters=(adapter,),
    )

    adapter._models = (model_b,)

    assert registry.adapter_for_model("test-provider", "model-a") is adapter
    assert registry.adapter_for_model("test-provider", "model-b") is None
    assert registry.adapter_for_model("test-provider", "model-*") is None
    assert registry.adapter_for_model("test-*", "model-a") is None


def test_sealing_rejects_duplicate_model_descriptions_within_adapter() -> None:
    adapter = _Adapter(_provider(), (_model(), _model()))

    with pytest.raises(DuplicateAdapterError, match="model-a"):
        ProviderRegistry.sealed(_declarations(), adapters=(adapter,))


@pytest.mark.parametrize(
    ("available", "capabilities"),
    [
        (True, None),
        (False, _capabilities()),
        (1, _capabilities()),
        (True, object()),
    ],
)
def test_runtime_observation_rejects_invalid_availability_combinations(
    available: object,
    capabilities: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RuntimeObservation(
            provider_id="test-provider",
            model_id="model-a",
            available=available,  # type: ignore[arg-type]
            capabilities=capabilities,  # type: ignore[arg-type]
        )


def test_unavailable_observation_without_capabilities_is_valid() -> None:
    registry = ProviderRegistry.sealed(_declarations())
    model = _model()
    observation = RuntimeObservation(
        provider_id="test-provider",
        model_id="model-a",
        available=False,
        capabilities=None,
        detail_code="runtime.not-installed",
    )

    assert registry.validate_observation(model, observation) is observation


@pytest.mark.parametrize(
    ("provider_id", "model_id"),
    [
        ("other-provider", "model-a"),
        ("test-provider", "other-model"),
        ("TEST-PROVIDER", "model-a"),
    ],
)
def test_runtime_observation_identity_must_match_selected_metadata_exactly(
    provider_id: str,
    model_id: str,
) -> None:
    registry = ProviderRegistry.sealed(_declarations())
    observation = RuntimeObservation(
        provider_id=provider_id,
        model_id=model_id,
        available=True,
        capabilities=_capabilities(),
    )

    with pytest.raises(RuntimeCapabilityError, match="identity"):
        registry.validate_observation(_model(), observation)


def test_selected_metadata_must_be_the_exact_catalog_declaration() -> None:
    registry = ProviderRegistry.sealed(_declarations())
    selected = _model(display_name="Altered")
    observation = RuntimeObservation(
        provider_id="test-provider",
        model_id="model-a",
        available=True,
        capabilities=_capabilities(),
    )

    with pytest.raises(RuntimeCapabilityError, match="selected metadata"):
        registry.validate_observation(selected, observation)


@pytest.mark.parametrize(
    ("field_name", "runtime_value"),
    [
        ("languages", frozenset({"en"})),
        ("automatic_language", False),
        ("tasks", frozenset({TranscriptionTask.TRANSCRIBE})),
        ("inputs", frozenset({InputKind.FILE})),
        (
            "timestamps",
            frozenset(
                {
                    TimestampGranularity.NONE,
                    TimestampGranularity.SEGMENT,
                }
            ),
        ),
        ("true_streaming", False),
        ("batch", False),
        ("cancellation", CancellationGranularity.SEGMENT_BOUNDARY),
        ("vad", False),
        ("diarization", False),
        ("punctuation", False),
        ("capitalization", False),
        ("language_input_mode", LanguageInputMode.ENFORCED),
    ],
)
def test_runtime_observation_forbids_loss_of_every_semantic_field(
    field_name: str,
    runtime_value: object,
) -> None:
    registry = ProviderRegistry.sealed(_declarations())
    runtime_capabilities = replace(
        _capabilities(),
        **{field_name: runtime_value},  # type: ignore[arg-type]
    )
    observation = RuntimeObservation(
        provider_id="test-provider",
        model_id="model-a",
        available=True,
        capabilities=runtime_capabilities,
    )

    with pytest.raises(RuntimeCapabilityError, match=field_name):
        registry.validate_observation(_model(), observation)


def test_runtime_observation_forbids_semantic_escalation() -> None:
    declared = _capabilities(diarization=False)
    model = _model(capabilities=declared)
    registry = ProviderRegistry.sealed(_declarations(models=(model,)))
    observation = RuntimeObservation(
        provider_id="test-provider",
        model_id="model-a",
        available=True,
        capabilities=replace(declared, diarization=True),
    )

    with pytest.raises(RuntimeCapabilityError, match="diarization"):
        registry.validate_observation(model, observation)


def test_runtime_observation_may_narrow_devices_and_precisions() -> None:
    registry = ProviderRegistry.sealed(_declarations())
    observation = RuntimeObservation(
        provider_id="test-provider",
        model_id="model-a",
        available=True,
        capabilities=replace(
            _capabilities(),
            execution_devices=frozenset({ExecutionDevice.CPU}),
            precisions=frozenset({"int8"}),
        ),
    )

    assert registry.validate_observation(_model(), observation) is observation


@pytest.mark.parametrize(
    ("field_name", "runtime_value"),
    [
        (
            "execution_devices",
            frozenset({ExecutionDevice.CPU, ExecutionDevice.METAL}),
        ),
        ("precisions", frozenset({"int8", "fp16"})),
    ],
)
def test_runtime_observation_forbids_device_or_precision_escalation(
    field_name: str,
    runtime_value: object,
) -> None:
    registry = ProviderRegistry.sealed(_declarations())
    observation = RuntimeObservation(
        provider_id="test-provider",
        model_id="model-a",
        available=True,
        capabilities=replace(
            _capabilities(),
            **{field_name: runtime_value},  # type: ignore[arg-type]
        ),
    )

    with pytest.raises(RuntimeCapabilityError, match=field_name):
        registry.validate_observation(_model(), observation)


def test_adapter_metadata_equality_protects_default_and_language_enforcement() -> None:
    declarations = _declarations()
    mismatches = (
        _model(default_precision="fp32"),
        _model(enforces_language_hint=False),
    )

    for mismatched_model in mismatches:
        with pytest.raises(AdapterRegistrationError, match="metadata"):
            ProviderRegistry.sealed(
                declarations,
                adapters=(_Adapter(_provider(), (mismatched_model,)),),
            )


def test_protocol_request_example_remains_a_contract_value() -> None:
    request = TranscriptionRequest(
        attempt_id="attempt-1",
        source=BufferAudioSource(b"\x00\x00", 16_000),
    )

    assert request.provider_id == "default"


def test_adapter_transcribe_annotation_resolves_at_runtime() -> None:
    hints = get_type_hints(TranscriptionAdapter.transcribe)

    assert ResolvedTranscriptionRequest.__module__ == "tldw_chatbook.STT.contracts"
    assert hints["request"] is ResolvedTranscriptionRequest
    assert hints["return"] is ProviderTranscriptionOutput
