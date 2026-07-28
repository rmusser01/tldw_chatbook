from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from threading import Event, Thread
from typing import Callable

import pytest

from tldw_chatbook.STT.contracts import (
    BufferAudioSource,
    CancellationGranularity,
    DeviceFailureOrigin,
    DeviceRetryPolicy,
    ExecutionDevice,
    FileAudioSource,
    InputKind,
    LanguageInputMode,
    PipelineCapabilities,
    PrivacyRequirements,
    ProducedCapabilities,
    ResolvedTranscriptionRequest,
    TimestampGranularity,
    TranscriptionAction,
    TranscriptionFailure,
    TranscriptionFailureCode,
    TranscriptionPhase,
    TranscriptionProgress,
    TranscriptionRequest,
    TranscriptionSegment,
    TranscriptionTask,
    TranscriptionTimings,
    TranscriptionWarningCode,
)
from tldw_chatbook.STT.coordinator import (
    TranscriptionCoordinator,
    TranscriptionCoordinatorError,
    TranscriptionFailureDecision,
    device_retry_policy_for_failure,
)
from tldw_chatbook.STT.registry import (
    CapabilitySet,
    CatalogDeclarations,
    ModelMetadata,
    ProviderMetadata,
    ProviderRegistry,
    ProviderTranscriptionOutput,
    RuntimeObservation,
)
from tldw_chatbook.STT.routing import (
    RoutingPolicy,
    TranscriptionRouter,
    build_builtin_declarations,
    build_builtin_registry,
)


POLICY = RoutingPolicy(validated_v3_languages=frozenset({"es", "fr"}))


def _capabilities(**overrides: object) -> CapabilitySet:
    values: dict[str, object] = {
        "languages": frozenset({"en", "es"}),
        "automatic_language": True,
        "tasks": frozenset({TranscriptionTask.TRANSCRIBE}),
        "inputs": frozenset({InputKind.FILE, InputKind.BUFFER}),
        "timestamps": frozenset(
            {
                TimestampGranularity.NONE,
                TimestampGranularity.SEGMENT,
                TimestampGranularity.WORD,
            }
        ),
        "true_streaming": False,
        "batch": True,
        "cancellation": CancellationGranularity.SEGMENT_BOUNDARY,
        "vad": True,
        "diarization": True,
        "punctuation": True,
        "capitalization": True,
        "language_input_mode": LanguageInputMode.AUTOMATIC,
        "execution_devices": frozenset({ExecutionDevice.CPU, ExecutionDevice.CUDA}),
        "precisions": frozenset({"int8", "float32"}),
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


def _output(**overrides: object) -> ProviderTranscriptionOutput:
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


def _request(**overrides: object) -> TranscriptionRequest:
    values: dict[str, object] = {
        "attempt_id": "attempt-1",
        "batch_id": "batch-1",
        "job_id": "job-1",
        "retry_of_attempt_id": "attempt-0",
        "retry_of_job_id": "job-0",
        "source": BufferAudioSource(b"\x00\x00", 16_000),
        "provider_id": "test-provider",
        "model_id": "model-a",
        "language": "en",
        "timestamps": TimestampGranularity.SEGMENT,
    }
    values.update(overrides)
    return TranscriptionRequest(**values)  # type: ignore[arg-type]


class _Token:
    def __init__(self, cancelled: Callable[[int], bool]) -> None:
        self.calls = 0
        self._cancelled = cancelled

    def is_cancelled(self) -> bool:
        self.calls += 1
        return self._cancelled(self.calls)


class _Adapter:
    def __init__(
        self,
        provider: ProviderMetadata,
        models: tuple[ModelMetadata, ...],
        *,
        observation: (
            RuntimeObservation
            | BaseException
            | Callable[[str], RuntimeObservation]
            | None
        ) = None,
        output: (
            ProviderTranscriptionOutput
            | BaseException
            | Callable[[ResolvedTranscriptionRequest], ProviderTranscriptionOutput]
            | None
        ) = None,
    ) -> None:
        self._provider = provider
        self._models = models
        self._observation = observation
        self._output = output
        self.probe_calls: list[str] = []
        self.transcribe_calls: list[ResolvedTranscriptionRequest] = []

    def provider(self) -> ProviderMetadata:
        return self._provider

    def describe(self) -> tuple[ModelMetadata, ...]:
        return self._models

    def probe(self, model_id: str) -> RuntimeObservation:
        self.probe_calls.append(model_id)
        if isinstance(self._observation, BaseException):
            raise self._observation
        if callable(self._observation):
            return self._observation(model_id)
        if self._observation is not None:
            return self._observation
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
        self.transcribe_calls.append(request)
        if isinstance(self._output, BaseException):
            raise self._output
        if callable(self._output):
            return self._output(request)
        return self._output or _output(
            effective_language=request.effective_language,
            effective_device=(
                request.request.device
                if request.request.device is not ExecutionDevice.AUTO
                else ExecutionDevice.CPU
            ),
            produced_capabilities=ProducedCapabilities(
                timestamps=request.request.timestamps,
                punctuation=True,
                capitalization=True,
                vad=request.request.vad,
                diarization=request.request.diarization,
            ),
        )

    def close(self) -> None:
        return None


def _coordinator(
    *,
    provider: ProviderMetadata | None = None,
    model: ModelMetadata | None = None,
    adapter: _Adapter | None = None,
    pipeline: PipelineCapabilities | None = None,
) -> tuple[TranscriptionCoordinator, _Adapter]:
    selected_provider = provider or _provider()
    selected_model = model or _model(provider_id=selected_provider.provider_id)
    selected_adapter = adapter or _Adapter(
        selected_provider,
        (selected_model,),
    )
    registry = ProviderRegistry.sealed(
        CatalogDeclarations(
            providers=(selected_provider,),
            models=(selected_model,),
        ),
        adapters=(selected_adapter,),
    )
    return (
        TranscriptionCoordinator(
            registry,
            TranscriptionRouter(POLICY),
            pipeline or PipelineCapabilities(),
        ),
        selected_adapter,
    )


def _assert_failure(
    caught: pytest.ExceptionInfo[TranscriptionCoordinatorError],
    code: TranscriptionFailureCode,
    *,
    secret: str | None = None,
) -> None:
    error = caught.value
    assert error.failure.code is code
    assert str(error) == error.failure.message
    if secret is not None:
        assert secret not in str(error)
        assert secret not in repr(error)
        assert secret not in repr(error.failure)


def _failure(
    code: TranscriptionFailureCode,
    *,
    provider_id: str = "test-provider",
    model_id: str = "model-a",
    requested_device: ExecutionDevice = ExecutionDevice.AUTO,
    effective_device: ExecutionDevice | None = None,
) -> TranscriptionFailure:
    return TranscriptionFailure(
        code=code,
        attempt_id="attempt-1",
        batch_id="batch-1",
        job_id="job-1",
        phase=TranscriptionPhase.TRANSCRIBING,
        provider_id=provider_id,
        model_id=model_id,
        artifact_root=None,
        precision="int8",
        requested_device=requested_device,
        effective_device=effective_device,
    )


def test_resolve_performs_routing_and_declared_composition_without_probing() -> None:
    coordinator, adapter = _coordinator()
    request = _request(vad=True, diarization=True)

    resolved = coordinator.resolve(request)

    assert resolved.request is request
    assert (resolved.provider_id, resolved.model_id) == (
        "test-provider",
        "model-a",
    )
    assert adapter.probe_calls == []
    assert adapter.transcribe_calls == []


def test_resolve_rejects_remote_processing_before_probe() -> None:
    provider = _provider(local_processing=False)
    coordinator, adapter = _coordinator(provider=provider)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.resolve(
            _request(
                privacy=PrivacyRequirements(allow_remote_processing=False),
            )
        )

    _assert_failure(caught, TranscriptionFailureCode.UNSUPPORTED_CAPABILITY)
    assert adapter.probe_calls == []


def test_resolve_rejects_forbidden_buffer_disk_staging_before_probe() -> None:
    coordinator, adapter = _coordinator(
        pipeline=PipelineCapabilities(requires_disk_staging_for_buffer=True)
    )

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.resolve(
            _request(
                privacy=PrivacyRequirements(allow_disk_staging=False),
            )
        )

    _assert_failure(caught, TranscriptionFailureCode.UNSUPPORTED_CAPABILITY)
    assert adapter.probe_calls == []


@pytest.mark.parametrize(
    ("source", "declared_input"),
    [
        (
            BufferAudioSource(b"\x00\x00", 16_000),
            InputKind.FILE,
        ),
        (
            FileAudioSource(Path("audio.wav")),
            InputKind.BUFFER,
        ),
    ],
)
def test_resolve_requires_the_exact_declared_source_kind(
    source: object,
    declared_input: InputKind,
) -> None:
    model = _model(capabilities=_capabilities(inputs=frozenset({declared_input})))
    coordinator, adapter = _coordinator(model=model)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.resolve(_request(source=source))

    _assert_failure(caught, TranscriptionFailureCode.UNSUPPORTED_CAPABILITY)
    assert adapter.probe_calls == []


@pytest.mark.parametrize(
    ("requested", "model_timestamps", "pipeline_timestamps"),
    [
        (
            TimestampGranularity.NONE,
            frozenset({TimestampGranularity.SEGMENT}),
            frozenset(),
        ),
        (
            TimestampGranularity.SEGMENT,
            frozenset({TimestampGranularity.SEGMENT}),
            frozenset(),
        ),
        (
            TimestampGranularity.WORD,
            frozenset({TimestampGranularity.NONE}),
            frozenset({TimestampGranularity.WORD}),
        ),
    ],
)
def test_resolve_accepts_none_or_composed_timestamp_support(
    requested: TimestampGranularity,
    model_timestamps: frozenset[TimestampGranularity],
    pipeline_timestamps: frozenset[TimestampGranularity],
) -> None:
    model = _model(capabilities=_capabilities(timestamps=model_timestamps))
    coordinator, adapter = _coordinator(
        model=model,
        pipeline=PipelineCapabilities(timestamps=pipeline_timestamps),
    )

    resolved = coordinator.resolve(_request(timestamps=requested))

    assert resolved.model_id == model.model_id
    assert adapter.probe_calls == []


def test_resolve_rejects_unsupported_composed_timestamp_before_probe() -> None:
    model = _model(
        capabilities=_capabilities(timestamps=frozenset({TimestampGranularity.NONE}))
    )
    coordinator, adapter = _coordinator(model=model)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.resolve(_request(timestamps=TimestampGranularity.WORD))

    _assert_failure(caught, TranscriptionFailureCode.UNSUPPORTED_CAPABILITY)
    assert adapter.probe_calls == []


@pytest.mark.parametrize(
    ("capability", "supplied_by_pipeline"), [("vad", True), ("diarization", True)]
)
def test_resolve_accepts_vad_and_diarization_from_the_composed_pipeline(
    capability: str,
    supplied_by_pipeline: bool,
) -> None:
    model = _model(capabilities=_capabilities(**{capability: False}))
    coordinator, _ = _coordinator(
        model=model,
        pipeline=PipelineCapabilities(
            vad=supplied_by_pipeline if capability == "vad" else False,
            diarization=(
                supplied_by_pipeline if capability == "diarization" else False
            ),
        ),
    )

    resolved = coordinator.resolve(_request(**{capability: True}))

    assert resolved.model_id == "model-a"


@pytest.mark.parametrize("capability", ["vad", "diarization"])
def test_resolve_rejects_missing_composed_boolean_capability(
    capability: str,
) -> None:
    model = _model(capabilities=_capabilities(**{capability: False}))
    coordinator, adapter = _coordinator(model=model)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.resolve(_request(**{capability: True}))

    _assert_failure(caught, TranscriptionFailureCode.UNSUPPORTED_CAPABILITY)
    assert adapter.probe_calls == []


def test_preflight_probes_only_the_exact_model_adapter() -> None:
    provider = _provider()
    model_a = _model()
    model_b = _model(model_id="model-b", display_name="Model B")
    adapter = _Adapter(provider, (model_b,))
    registry = ProviderRegistry.sealed(
        CatalogDeclarations(
            providers=(provider,),
            models=(model_a, model_b),
        ),
        adapters=(adapter,),
    )
    coordinator = TranscriptionCoordinator(
        registry,
        TranscriptionRouter(POLICY),
        PipelineCapabilities(),
    )
    resolved = TranscriptionRouter(POLICY).resolve(_request(), registry)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.preflight(resolved)

    _assert_failure(caught, TranscriptionFailureCode.PROVIDER_UNAVAILABLE)
    assert adapter.probe_calls == []
    assert adapter.transcribe_calls == []


def test_preflight_rejects_an_unavailable_exact_observation() -> None:
    model = _model()
    observation = RuntimeObservation(
        provider_id=model.provider_id,
        model_id=model.model_id,
        available=False,
        capabilities=None,
        detail_code="package.missing",
    )
    adapter = _Adapter(_provider(), (model,), observation=observation)
    coordinator, _ = _coordinator(model=model, adapter=adapter)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.preflight(coordinator.resolve(_request()))

    _assert_failure(caught, TranscriptionFailureCode.PROVIDER_UNAVAILABLE)
    assert adapter.probe_calls == ["model-a"]
    assert adapter.transcribe_calls == []


def test_preflight_sanitizes_probe_exceptions() -> None:
    secret = "RAW-PROBE-SECRET"
    model = _model()
    adapter = _Adapter(
        _provider(),
        (model,),
        observation=RuntimeError(secret),
    )
    coordinator, _ = _coordinator(model=model, adapter=adapter)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.preflight(coordinator.resolve(_request()))

    _assert_failure(
        caught,
        TranscriptionFailureCode.PROVIDER_UNAVAILABLE,
        secret=secret,
    )
    assert adapter.transcribe_calls == []


def test_probe_exception_after_setting_cancellation_is_cancelled() -> None:
    cancelled = False
    token = _Token(lambda _: cancelled)
    model = _model()

    def cancelled_probe(model_id: str) -> RuntimeObservation:
        nonlocal cancelled
        del model_id
        cancelled = True
        raise RuntimeError("raw probe cancellation detail")

    adapter = _Adapter(_provider(), (model,), observation=cancelled_probe)
    coordinator, _ = _coordinator(model=model, adapter=adapter)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.preflight(coordinator.resolve(_request(cancellation=token)))

    _assert_failure(
        caught,
        TranscriptionFailureCode.CANCELLED,
        secret="raw probe cancellation detail",
    )
    assert adapter.probe_calls == ["model-a"]
    assert adapter.transcribe_calls == []


def test_cancellation_set_during_successful_probe_discards_observation() -> None:
    cancelled = False
    token = _Token(lambda _: cancelled)
    model = _model()

    def cancelled_probe(model_id: str) -> RuntimeObservation:
        nonlocal cancelled
        cancelled = True
        return RuntimeObservation(
            provider_id=model.provider_id,
            model_id=model_id,
            available=True,
            capabilities=model.capabilities,
        )

    adapter = _Adapter(_provider(), (model,), observation=cancelled_probe)
    coordinator, _ = _coordinator(model=model, adapter=adapter)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.preflight(coordinator.resolve(_request(cancellation=token)))

    _assert_failure(caught, TranscriptionFailureCode.CANCELLED)
    assert adapter.probe_calls == ["model-a"]
    assert adapter.transcribe_calls == []


@pytest.mark.parametrize("probe_result", ["unavailable", "incompatible"])
def test_cancellation_during_probe_wins_over_probe_result_classification(
    probe_result: str,
) -> None:
    cancelled = False
    token = _Token(lambda _: cancelled)
    model = _model()

    def cancelled_probe(model_id: str) -> RuntimeObservation:
        nonlocal cancelled
        cancelled = True
        if probe_result == "unavailable":
            return RuntimeObservation(
                provider_id=model.provider_id,
                model_id=model_id,
                available=False,
                capabilities=None,
            )
        return RuntimeObservation(
            provider_id="incompatible-provider",
            model_id=model_id,
            available=True,
            capabilities=model.capabilities,
        )

    adapter = _Adapter(_provider(), (model,), observation=cancelled_probe)
    coordinator, _ = _coordinator(model=model, adapter=adapter)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.preflight(coordinator.resolve(_request(cancellation=token)))

    _assert_failure(caught, TranscriptionFailureCode.CANCELLED)
    assert adapter.probe_calls == ["model-a"]
    assert adapter.transcribe_calls == []


@pytest.mark.parametrize(
    "observation",
    [
        RuntimeObservation(
            provider_id="other-provider",
            model_id="model-a",
            available=True,
            capabilities=_capabilities(),
        ),
        RuntimeObservation(
            provider_id="test-provider",
            model_id="model-a",
            available=True,
            capabilities=_capabilities(vad=False),
        ),
        RuntimeObservation(
            provider_id="test-provider",
            model_id="model-a",
            available=True,
            capabilities=_capabilities(
                execution_devices=frozenset(
                    {ExecutionDevice.CPU, ExecutionDevice.CUDA, ExecutionDevice.METAL}
                )
            ),
        ),
    ],
)
def test_preflight_maps_identity_semantic_and_runtime_escalation_to_artifact_incompatible(
    observation: RuntimeObservation,
) -> None:
    model = _model()
    adapter = _Adapter(_provider(), (model,), observation=observation)
    coordinator, _ = _coordinator(model=model, adapter=adapter)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.preflight(coordinator.resolve(_request()))

    _assert_failure(caught, TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE)
    assert adapter.transcribe_calls == []


@pytest.mark.parametrize(
    ("request_overrides", "runtime_overrides"),
    [
        (
            {"precision": "float32"},
            {"precisions": frozenset({"int8"})},
        ),
        (
            {"device": ExecutionDevice.CUDA},
            {"execution_devices": frozenset({ExecutionDevice.CPU})},
        ),
    ],
)
def test_preflight_rejects_runtime_narrowing_of_the_selected_configuration(
    request_overrides: dict[str, object],
    runtime_overrides: dict[str, object],
) -> None:
    model = _model()
    observation = RuntimeObservation(
        provider_id=model.provider_id,
        model_id=model.model_id,
        available=True,
        capabilities=_capabilities(**runtime_overrides),
    )
    adapter = _Adapter(_provider(), (model,), observation=observation)
    coordinator, _ = _coordinator(model=model, adapter=adapter)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.preflight(coordinator.resolve(_request(**request_overrides)))

    _assert_failure(caught, TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE)
    assert adapter.transcribe_calls == []


def test_preflight_allows_auto_when_any_concrete_runtime_device_remains() -> None:
    model = _model()
    runtime = replace(
        model.capabilities,
        execution_devices=frozenset({ExecutionDevice.CUDA}),
    )
    adapter = _Adapter(
        _provider(),
        (model,),
        observation=RuntimeObservation(
            provider_id=model.provider_id,
            model_id=model.model_id,
            available=True,
            capabilities=runtime,
        ),
    )
    coordinator, _ = _coordinator(model=model, adapter=adapter)

    observation = coordinator.preflight(
        coordinator.resolve(_request(device=ExecutionDevice.AUTO))
    )

    assert observation.capabilities == runtime
    assert adapter.probe_calls == ["model-a"]
    assert adapter.transcribe_calls == []


def test_preflight_revalidates_resolved_precision_before_probe() -> None:
    coordinator, adapter = _coordinator()
    resolved = coordinator.resolve(_request())

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.preflight(replace(resolved, precision="stale-precision"))

    _assert_failure(caught, TranscriptionFailureCode.UNSUPPORTED_CAPABILITY)
    assert adapter.probe_calls == []


def test_cancellation_before_probe_never_calls_the_adapter() -> None:
    token = _Token(lambda _: True)
    coordinator, adapter = _coordinator()
    resolved = coordinator.resolve(_request(cancellation=token))

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.preflight(resolved)

    _assert_failure(caught, TranscriptionFailureCode.CANCELLED)
    assert adapter.probe_calls == []
    assert adapter.transcribe_calls == []


def test_cancellation_before_probe_preserves_exact_resolved_identity() -> None:
    token = _Token(lambda _: True)
    registry = build_builtin_registry(POLICY)
    coordinator = TranscriptionCoordinator(
        registry,
        TranscriptionRouter(POLICY),
        PipelineCapabilities(),
    )
    request = TranscriptionRequest(
        attempt_id="attempt-resolved-cancel",
        source=BufferAudioSource(b"\x00\x00", 16_000),
        language="en",
        device=ExecutionDevice.CPU,
        timestamps=TimestampGranularity.NONE,
        cancellation=token,
    )
    resolved = coordinator.resolve(request)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.preflight(resolved)

    failure = caught.value.failure
    assert failure.code is TranscriptionFailureCode.CANCELLED
    assert failure.provider_id == POLICY.parakeet_provider_id
    assert failure.model_id == POLICY.parakeet_v2_model_id
    assert failure.precision == "int8"
    assert failure.requested_device is ExecutionDevice.CPU
    assert failure.effective_device is None


def test_cancellation_immediately_before_execution_never_calls_transcribe() -> None:
    token = _Token(lambda call: call >= 4)
    coordinator, adapter = _coordinator()

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.transcribe(_request(cancellation=token))

    _assert_failure(caught, TranscriptionFailureCode.CANCELLED)
    assert adapter.probe_calls == ["model-a"]
    assert adapter.transcribe_calls == []


def test_success_normalizes_provenance_warnings_and_progress_without_percentages() -> (
    None
):
    events: list[TranscriptionProgress] = []
    adapter = _Adapter(_provider(), (_model(),), output=_output())
    coordinator, _ = _coordinator(adapter=adapter)

    result = coordinator.transcribe(_request(progress=events.append))

    assert len(adapter.transcribe_calls) == 1
    assert result.text == "hello"
    assert result.provenance.schema_version == 1
    assert result.provenance.attempt_id == "attempt-1"
    assert result.provenance.batch_id == "batch-1"
    assert result.provenance.job_id == "job-1"
    assert result.provenance.retry_of_attempt_id == "attempt-0"
    assert result.provenance.retry_of_job_id == "job-0"
    assert result.provenance.provider_id == "test-provider"
    assert result.provenance.model_id == "model-a"
    assert result.provenance.artifact_root is None
    assert result.provenance.artifact_dependencies == ()
    assert result.provenance.precision == "int8"
    assert result.provenance.requested_device is ExecutionDevice.AUTO
    assert result.provenance.effective_device is ExecutionDevice.CPU
    assert result.provenance.requested_language == "en"
    assert result.provenance.effective_language == "en"
    assert result.provenance.detected_language is None
    assert result.provenance.task is TranscriptionTask.TRANSCRIBE
    assert result.warnings == ()
    assert [event.phase for event in events] == [
        TranscriptionPhase.QUEUED,
        TranscriptionPhase.LOADING,
        TranscriptionPhase.TRANSCRIBING,
        TranscriptionPhase.POST_PROCESSING,
        TranscriptionPhase.COMPLETE,
    ]
    assert all(event.fraction is None for event in events)
    assert all(event.detail_code is None for event in events)


def test_adapter_progress_is_wrapped_and_cannot_break_coordinator_ordering() -> None:
    events: list[TranscriptionProgress] = []
    raw_sink = events.append
    adapter_sinks: list[object] = []

    def malicious_progress(
        request: ResolvedTranscriptionRequest,
    ) -> ProviderTranscriptionOutput:
        sink = request.request.progress
        assert sink is not None
        adapter_sinks.append(sink)
        sink(
            TranscriptionProgress(
                attempt_id="malicious-attempt",
                batch_id="malicious-batch",
                job_id="malicious-job",
                phase=TranscriptionPhase.COMPLETE,
                fraction=0.5,
            )
        )
        sink(
            TranscriptionProgress(
                attempt_id="malicious-attempt",
                batch_id=None,
                job_id=None,
                phase=TranscriptionPhase.TRANSCRIBING,
                fraction=0.7,
                detail_code="decode.segment-7",
            )
        )
        sink(
            TranscriptionProgress(
                attempt_id="malicious-attempt",
                batch_id=None,
                job_id=None,
                phase=TranscriptionPhase.TRANSCRIBING,
                fraction=0.2,
            )
        )
        sink(
            TranscriptionProgress(
                attempt_id="malicious-attempt",
                batch_id=None,
                job_id=None,
                phase=TranscriptionPhase.QUEUED,
                fraction=0.9,
            )
        )
        return _output(
            effective_language=request.effective_language,
            produced_capabilities=ProducedCapabilities(
                timestamps=request.request.timestamps,
                punctuation=True,
                capitalization=True,
                vad=request.request.vad,
                diarization=request.request.diarization,
            ),
        )

    adapter = _Adapter(_provider(), (_model(),), output=malicious_progress)
    coordinator, _ = _coordinator(adapter=adapter)

    result = coordinator.transcribe(_request(progress=raw_sink))

    assert result.text == "hello"
    assert adapter_sinks and adapter_sinks[0] is not raw_sink
    assert [event.phase for event in events] == [
        TranscriptionPhase.QUEUED,
        TranscriptionPhase.LOADING,
        TranscriptionPhase.TRANSCRIBING,
        TranscriptionPhase.TRANSCRIBING,
        TranscriptionPhase.POST_PROCESSING,
        TranscriptionPhase.COMPLETE,
    ]
    assert [event.fraction for event in events] == [
        None,
        None,
        None,
        0.7,
        None,
        None,
    ]
    forwarded = events[3]
    assert forwarded.attempt_id == "attempt-1"
    assert forwarded.batch_id == "batch-1"
    assert forwarded.job_id == "job-1"
    assert forwarded.detail_code == "decode.segment-7"


def test_adapter_progress_close_waits_for_in_flight_delivery() -> None:
    events: list[TranscriptionProgress] = []
    delivery_started = Event()
    release_delivery = Event()
    transcription_finished = Event()
    callback_threads: list[Thread] = []
    result_holder: list[object] = []

    def blocking_sink(event: TranscriptionProgress) -> None:
        if event.phase is TranscriptionPhase.TRANSCRIBING and event.fraction == 0.5:
            delivery_started.set()
            assert release_delivery.wait(timeout=2)
        events.append(event)

    def retained_progress(
        request: ResolvedTranscriptionRequest,
    ) -> ProviderTranscriptionOutput:
        sink = request.request.progress
        assert sink is not None
        callback = Thread(
            target=sink,
            args=(
                TranscriptionProgress(
                    attempt_id="retained-attempt",
                    batch_id=None,
                    job_id=None,
                    phase=TranscriptionPhase.TRANSCRIBING,
                    fraction=0.5,
                ),
            ),
            daemon=True,
        )
        callback_threads.append(callback)
        callback.start()
        assert delivery_started.wait(timeout=2)
        return _output(
            effective_language=request.effective_language,
            produced_capabilities=ProducedCapabilities(
                timestamps=request.request.timestamps,
                punctuation=True,
                capitalization=True,
                vad=request.request.vad,
                diarization=request.request.diarization,
            ),
        )

    adapter = _Adapter(_provider(), (_model(),), output=retained_progress)
    coordinator, _ = _coordinator(adapter=adapter)

    def run_transcription() -> None:
        result_holder.append(coordinator.transcribe(_request(progress=blocking_sink)))
        transcription_finished.set()

    transcription = Thread(target=run_transcription, daemon=True)
    transcription.start()
    assert delivery_started.wait(timeout=2)
    transcription_finished.wait(timeout=0.2)
    release_delivery.set()
    transcription.join(timeout=2)
    for callback in callback_threads:
        callback.join(timeout=2)

    assert not transcription.is_alive()
    assert result_holder
    assert [event.phase for event in events] == [
        TranscriptionPhase.QUEUED,
        TranscriptionPhase.LOADING,
        TranscriptionPhase.TRANSCRIBING,
        TranscriptionPhase.TRANSCRIBING,
        TranscriptionPhase.POST_PROCESSING,
        TranscriptionPhase.COMPLETE,
    ]
    assert events[-1].phase is TranscriptionPhase.COMPLETE


def test_v3_normalization_keeps_routing_only_language_fields_and_warning() -> None:
    declarations = build_builtin_declarations(POLICY)
    parakeet_models = tuple(
        model
        for model in declarations.models
        if model.provider_id == POLICY.parakeet_provider_id
    )
    provider = next(
        provider
        for provider in declarations.providers
        if provider.provider_id == POLICY.parakeet_provider_id
    )
    adapter = _Adapter(
        provider,
        parakeet_models,
        output=lambda request: _output(
            effective_language=request.effective_language,
            detected_language=None,
            warnings=(
                TranscriptionWarningCode.REQUESTED_LANGUAGE_NOT_ENFORCED,
                TranscriptionWarningCode.REQUESTED_LANGUAGE_NOT_ENFORCED,
            ),
            produced_capabilities=ProducedCapabilities(
                timestamps=request.request.timestamps,
                punctuation=True,
                capitalization=True,
                vad=request.request.vad,
                diarization=request.request.diarization,
            ),
        ),
    )
    registry = build_builtin_registry(POLICY, adapters=(adapter,))
    coordinator = TranscriptionCoordinator(
        registry,
        TranscriptionRouter(POLICY),
        PipelineCapabilities(timestamps=frozenset({TimestampGranularity.SEGMENT})),
    )

    result = coordinator.transcribe(
        TranscriptionRequest(
            attempt_id="attempt-v3",
            source=BufferAudioSource(b"\x00\x00", 16_000),
            language="es",
            timestamps=TimestampGranularity.SEGMENT,
        )
    )

    assert result.provenance.requested_language == "es"
    assert result.provenance.effective_language == "auto"
    assert result.provenance.detected_language is None
    assert result.warnings == (
        TranscriptionWarningCode.REQUESTED_LANGUAGE_NOT_ENFORCED,
    )


def test_faster_whisper_auto_preserves_trustworthy_detected_language() -> None:
    declarations = build_builtin_declarations(POLICY)
    model = next(
        model
        for model in declarations.models
        if model.provider_id == POLICY.faster_whisper_provider_id
    )
    provider = next(
        provider
        for provider in declarations.providers
        if provider.provider_id == POLICY.faster_whisper_provider_id
    )
    adapter = _Adapter(
        provider,
        (model,),
        output=_output(
            effective_language="auto",
            detected_language="ja",
        ),
    )
    registry = build_builtin_registry(POLICY, adapters=(adapter,))
    coordinator = TranscriptionCoordinator(
        registry,
        TranscriptionRouter(POLICY),
        PipelineCapabilities(),
    )

    result = coordinator.transcribe(
        TranscriptionRequest(
            attempt_id="attempt-auto",
            source=BufferAudioSource(b"\x00\x00", 16_000),
            language="auto",
            timestamps=TimestampGranularity.SEGMENT,
        )
    )

    assert result.provenance.requested_language == "auto"
    assert result.provenance.effective_language == "auto"
    assert result.provenance.detected_language == "ja"


def test_faster_whisper_rejects_auto_as_a_detected_language() -> None:
    declarations = build_builtin_declarations(POLICY)
    model = next(
        model
        for model in declarations.models
        if model.provider_id == POLICY.faster_whisper_provider_id
    )
    provider = next(
        provider
        for provider in declarations.providers
        if provider.provider_id == POLICY.faster_whisper_provider_id
    )
    adapter = _Adapter(
        provider,
        (model,),
        output=_output(
            effective_language="auto",
            detected_language="auto",
        ),
    )
    registry = build_builtin_registry(POLICY, adapters=(adapter,))
    coordinator = TranscriptionCoordinator(
        registry,
        TranscriptionRouter(POLICY),
        PipelineCapabilities(),
    )
    request = TranscriptionRequest(
        attempt_id="attempt-auto",
        source=BufferAudioSource(b"\x00\x00", 16_000),
        language="auto",
        timestamps=TimestampGranularity.SEGMENT,
    )

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.transcribe(request)

    _assert_failure(caught, TranscriptionFailureCode.INFERENCE_FAILED)


@pytest.mark.parametrize(
    "output",
    [
        _output(effective_language="es"),
        _output(detected_language="es"),
        _output(effective_device=ExecutionDevice.CUDA),
        _output(
            produced_capabilities=ProducedCapabilities(
                timestamps=TimestampGranularity.WORD,
                punctuation=True,
                capitalization=True,
                vad=False,
                diarization=False,
            )
        ),
        _output(
            produced_capabilities=ProducedCapabilities(
                timestamps=TimestampGranularity.SEGMENT,
                punctuation=True,
                capitalization=True,
                vad=True,
                diarization=False,
            )
        ),
        _output(warnings=(TranscriptionWarningCode.REQUESTED_LANGUAGE_NOT_ENFORCED,)),
        _output(
            segments=(
                TranscriptionSegment(1.0, 2.0, "later"),
                TranscriptionSegment(0.0, 1.0, "earlier"),
            )
        ),
    ],
)
def test_transcribe_rejects_contradictory_or_unsupported_provider_output(
    output: ProviderTranscriptionOutput,
) -> None:
    model = _model(
        capabilities=_capabilities(execution_devices=frozenset({ExecutionDevice.CPU}))
    )
    adapter = _Adapter(_provider(), (model,), output=output)
    coordinator, _ = _coordinator(model=model, adapter=adapter)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.transcribe(_request())

    _assert_failure(caught, TranscriptionFailureCode.INFERENCE_FAILED)
    assert len(adapter.transcribe_calls) == 1


def test_cancellation_requested_during_adapter_call_discards_its_success() -> None:
    cancelled = False
    token = _Token(lambda _: cancelled)
    events: list[TranscriptionProgress] = []

    def cancelled_success(
        request: ResolvedTranscriptionRequest,
    ) -> ProviderTranscriptionOutput:
        nonlocal cancelled
        cancelled = True
        return _output(
            effective_language=request.effective_language,
            produced_capabilities=ProducedCapabilities(
                timestamps=request.request.timestamps,
                punctuation=True,
                capitalization=True,
                vad=request.request.vad,
                diarization=request.request.diarization,
            ),
        )

    adapter = _Adapter(_provider(), (_model(),), output=cancelled_success)
    coordinator, _ = _coordinator(adapter=adapter)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.transcribe(_request(cancellation=token, progress=events.append))

    _assert_failure(caught, TranscriptionFailureCode.CANCELLED)
    assert len(adapter.transcribe_calls) == 1
    assert TranscriptionPhase.POST_PROCESSING not in {event.phase for event in events}
    assert TranscriptionPhase.COMPLETE not in {event.phase for event in events}


def test_adapter_exception_after_setting_cancellation_is_cancelled() -> None:
    cancelled = False
    token = _Token(lambda _: cancelled)

    def cancelled_failure(
        request: ResolvedTranscriptionRequest,
    ) -> ProviderTranscriptionOutput:
        nonlocal cancelled
        del request
        cancelled = True
        raise RuntimeError("raw provider cancellation detail")

    adapter = _Adapter(_provider(), (_model(),), output=cancelled_failure)
    coordinator, _ = _coordinator(adapter=adapter)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.transcribe(_request(cancellation=token))

    _assert_failure(
        caught,
        TranscriptionFailureCode.CANCELLED,
        secret="raw provider cancellation detail",
    )
    assert len(adapter.transcribe_calls) == 1


def test_adapter_exception_is_sanitized_and_never_invokes_fallback() -> None:
    secret = "RAW-ADAPTER-SECRET"
    adapter = _Adapter(
        _provider(),
        (_model(),),
        output=RuntimeError(secret),
    )
    coordinator, _ = _coordinator(adapter=adapter)

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.transcribe(_request())

    _assert_failure(caught, TranscriptionFailureCode.INFERENCE_FAILED, secret=secret)
    assert len(adapter.transcribe_calls) == 1


def test_adapter_failure_does_not_probe_or_execute_a_registered_fallback() -> None:
    declarations = build_builtin_declarations(POLICY)
    faster_model = next(
        model
        for model in declarations.models
        if model.provider_id == POLICY.faster_whisper_provider_id
    )
    faster_provider = next(
        provider
        for provider in declarations.providers
        if provider.provider_id == POLICY.faster_whisper_provider_id
    )
    selected_model = _model()
    selected_adapter = _Adapter(
        _provider(),
        (selected_model,),
        output=RuntimeError("selected provider failed"),
    )
    faster_adapter = _Adapter(faster_provider, (faster_model,))
    registry = build_builtin_registry(
        POLICY,
        adapters=(selected_adapter, faster_adapter),
        extra_declarations=CatalogDeclarations(
            providers=(_provider(),),
            models=(selected_model,),
        ),
    )
    coordinator = TranscriptionCoordinator(
        registry,
        TranscriptionRouter(POLICY),
        PipelineCapabilities(),
    )

    with pytest.raises(TranscriptionCoordinatorError):
        coordinator.transcribe(_request())

    assert len(selected_adapter.transcribe_calls) == 1
    assert faster_adapter.probe_calls == []
    assert faster_adapter.transcribe_calls == []


@pytest.mark.parametrize("capability", ["timestamps", "vad", "diarization"])
def test_transcribe_accepts_capabilities_supplied_by_the_composed_pipeline(
    capability: str,
) -> None:
    requested_timestamp = (
        TimestampGranularity.SEGMENT
        if capability == "timestamps"
        else TimestampGranularity.NONE
    )
    model = _model(
        capabilities=_capabilities(
            timestamps=frozenset({TimestampGranularity.NONE}),
            vad=False,
            diarization=False,
        )
    )
    produced = ProducedCapabilities(
        timestamps=requested_timestamp,
        punctuation=True,
        capitalization=True,
        vad=capability == "vad",
        diarization=capability == "diarization",
    )
    output = _output(
        segments=(
            (TranscriptionSegment(0.0, 1.0, "hello"),)
            if requested_timestamp is TimestampGranularity.SEGMENT
            else ()
        ),
        produced_capabilities=produced,
    )
    adapter = _Adapter(_provider(), (model,), output=output)
    coordinator, _ = _coordinator(
        model=model,
        adapter=adapter,
        pipeline=PipelineCapabilities(
            timestamps=(
                frozenset({TimestampGranularity.SEGMENT})
                if capability == "timestamps"
                else frozenset()
            ),
            vad=capability == "vad",
            diarization=capability == "diarization",
        ),
    )

    result = coordinator.transcribe(
        _request(
            timestamps=requested_timestamp,
            vad=capability == "vad",
            diarization=capability == "diarization",
        )
    )

    assert result.produced_capabilities == produced


@pytest.mark.parametrize(
    "failing_phase",
    [
        TranscriptionPhase.QUEUED,
        TranscriptionPhase.LOADING,
        TranscriptionPhase.TRANSCRIBING,
        TranscriptionPhase.POST_PROCESSING,
        TranscriptionPhase.COMPLETE,
    ],
)
def test_progress_callback_failure_is_sanitized_and_stops_execution(
    failing_phase: TranscriptionPhase,
) -> None:
    secret = "RAW-CALLBACK-SECRET"

    def progress(event: object) -> None:
        if getattr(event, "phase") is failing_phase:
            raise RuntimeError(secret)

    coordinator, adapter = _coordinator()

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        coordinator.transcribe(_request(progress=progress))

    _assert_failure(caught, TranscriptionFailureCode.INFERENCE_FAILED, secret=secret)
    if failing_phase in {TranscriptionPhase.QUEUED, TranscriptionPhase.LOADING}:
        assert adapter.transcribe_calls == []


def _action_coordinator(
    *,
    model: ModelMetadata | None = None,
    pipeline: PipelineCapabilities | None = None,
    include_faster_whisper: bool = True,
) -> TranscriptionCoordinator:
    selected_model = model or _model()
    declarations = CatalogDeclarations(
        providers=(_provider(),),
        models=(selected_model,),
    )
    registry = (
        build_builtin_registry(POLICY, extra_declarations=declarations)
        if include_faster_whisper
        else ProviderRegistry.sealed(declarations)
    )
    return TranscriptionCoordinator(
        registry,
        TranscriptionRouter(POLICY),
        pipeline or PipelineCapabilities(),
    )


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (
            TranscriptionFailureCode.MODEL_NOT_INSTALLED,
            (
                TranscriptionAction.INSTALL_MODEL,
                TranscriptionAction.CHOOSE_INSTALLED_MODEL,
                TranscriptionAction.RETRY_WITH_FASTER_WHISPER,
            ),
        ),
        (
            TranscriptionFailureCode.ARTIFACT_CORRUPT,
            (
                TranscriptionAction.INSTALL_MODEL,
                TranscriptionAction.CHOOSE_INSTALLED_MODEL,
                TranscriptionAction.RETRY_WITH_FASTER_WHISPER,
            ),
        ),
        (
            TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
            (
                TranscriptionAction.CHOOSE_INSTALLED_MODEL,
                TranscriptionAction.RETRY_WITH_FASTER_WHISPER,
            ),
        ),
        (
            TranscriptionFailureCode.PROVIDER_UNAVAILABLE,
            (
                TranscriptionAction.RETRY_SAME_CONFIGURATION,
                TranscriptionAction.RETRY_WITH_FASTER_WHISPER,
            ),
        ),
        (
            TranscriptionFailureCode.PROVIDER_REMOVED,
            (TranscriptionAction.RETRY_WITH_FASTER_WHISPER,),
        ),
        (
            TranscriptionFailureCode.UNSUPPORTED_LANGUAGE,
            (TranscriptionAction.RETRY_WITH_FASTER_WHISPER,),
        ),
        (
            TranscriptionFailureCode.UNSUPPORTED_CAPABILITY,
            (TranscriptionAction.RETRY_WITH_FASTER_WHISPER,),
        ),
        (
            TranscriptionFailureCode.INSUFFICIENT_DISK_SPACE,
            (TranscriptionAction.RETRY_WITH_FASTER_WHISPER,),
        ),
        (
            TranscriptionFailureCode.INSUFFICIENT_MEMORY,
            (TranscriptionAction.RETRY_WITH_FASTER_WHISPER,),
        ),
        (
            TranscriptionFailureCode.INFERENCE_FAILED,
            (TranscriptionAction.RETRY_WITH_FASTER_WHISPER,),
        ),
        (
            TranscriptionFailureCode.ENGINE_CRASHED,
            (
                TranscriptionAction.RETRY_SAME_CONFIGURATION,
                TranscriptionAction.RETRY_WITH_FASTER_WHISPER,
            ),
        ),
        (
            TranscriptionFailureCode.CANCELLED,
            (TranscriptionAction.RETRY_SAME_CONFIGURATION,),
        ),
    ],
)
def test_failure_action_matrix_is_exact_for_a_compatible_non_faster_whisper_request(
    code: TranscriptionFailureCode,
    expected: tuple[TranscriptionAction, ...],
) -> None:
    coordinator = _action_coordinator()

    decision = coordinator.failure_decision(_request(), _failure(code))

    assert decision.failure.code is code
    assert decision.actions == expected
    assert type(decision.actions) is tuple
    assert decision.device_retry_policy == DeviceRetryPolicy.no_retry()


def test_automatic_only_unsupported_language_adds_change_to_auto_action() -> None:
    model = _model(
        capabilities=_capabilities(
            languages=frozenset(),
            automatic_language=True,
            language_input_mode=LanguageInputMode.AUTOMATIC_ONLY,
        ),
        enforces_language_hint=False,
    )
    coordinator = _action_coordinator(model=model)

    decision = coordinator.failure_decision(
        _request(language="en"),
        _failure(TranscriptionFailureCode.UNSUPPORTED_LANGUAGE),
    )

    assert decision.actions == (
        TranscriptionAction.RETRY_WITH_FASTER_WHISPER,
        TranscriptionAction.CHANGE_LANGUAGE_TO_AUTO,
    )


def test_change_to_auto_is_not_offered_for_non_automatic_only_models() -> None:
    coordinator = _action_coordinator()

    decision = coordinator.failure_decision(
        _request(language="en"),
        _failure(TranscriptionFailureCode.UNSUPPORTED_LANGUAGE),
    )

    assert TranscriptionAction.CHANGE_LANGUAGE_TO_AUTO not in decision.actions


@pytest.mark.parametrize(
    ("code", "remaining"),
    [
        (
            TranscriptionFailureCode.MODEL_NOT_INSTALLED,
            (
                TranscriptionAction.INSTALL_MODEL,
                TranscriptionAction.CHOOSE_INSTALLED_MODEL,
            ),
        ),
        (TranscriptionFailureCode.UNSUPPORTED_CAPABILITY, ()),
        (
            TranscriptionFailureCode.ENGINE_CRASHED,
            (TranscriptionAction.RETRY_SAME_CONFIGURATION,),
        ),
    ],
)
def test_faster_whisper_action_is_removed_when_target_is_not_declared(
    code: TranscriptionFailureCode,
    remaining: tuple[TranscriptionAction, ...],
) -> None:
    coordinator = _action_coordinator(include_faster_whisper=False)

    decision = coordinator.failure_decision(_request(), _failure(code))

    assert decision.actions == remaining


@pytest.mark.parametrize(
    ("constraint", "value"),
    [
        ("diarization", True),
        ("device", ExecutionDevice.METAL),
    ],
)
def test_faster_whisper_action_is_removed_when_it_cannot_satisfy_original_constraints(
    constraint: str,
    value: object,
) -> None:
    model = _model(
        capabilities=_capabilities(
            execution_devices=frozenset(
                {
                    ExecutionDevice.CPU,
                    ExecutionDevice.CUDA,
                    ExecutionDevice.METAL,
                }
            )
        )
    )
    coordinator = _action_coordinator(model=model)

    decision = coordinator.failure_decision(
        _request(**{constraint: value}),
        _failure(TranscriptionFailureCode.UNSUPPORTED_CAPABILITY),
    )

    assert decision.actions == ()


def test_faster_whisper_failure_never_offers_faster_whisper_again() -> None:
    coordinator = TranscriptionCoordinator(
        build_builtin_registry(POLICY),
        TranscriptionRouter(POLICY),
        PipelineCapabilities(),
    )
    request = TranscriptionRequest(
        attempt_id="attempt-fw",
        source=BufferAudioSource(b"\x00\x00", 16_000),
        provider_id=POLICY.faster_whisper_provider_id,
        model_id=POLICY.faster_whisper_model_id,
        language="en",
        timestamps=TimestampGranularity.SEGMENT,
    )

    decision = coordinator.failure_decision(
        request,
        _failure(
            TranscriptionFailureCode.ENGINE_CRASHED,
            provider_id=POLICY.faster_whisper_provider_id,
            model_id=POLICY.faster_whisper_model_id,
        ),
    )

    assert decision.actions == (TranscriptionAction.RETRY_SAME_CONFIGURATION,)


def test_failure_decision_is_frozen_typed_data_and_rejects_arbitrary_actions() -> None:
    decision = TranscriptionFailureDecision(
        failure=_failure(TranscriptionFailureCode.CANCELLED),
        actions=(TranscriptionAction.RETRY_SAME_CONFIGURATION,),
        device_retry_policy=DeviceRetryPolicy.no_retry(),
    )

    assert not hasattr(decision, "__dict__")
    with pytest.raises(FrozenInstanceError):
        decision.actions = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        TranscriptionFailureDecision(
            failure=decision.failure,
            actions=("adapter-action",),  # type: ignore[arg-type]
            device_retry_policy=DeviceRetryPolicy.no_retry(),
        )


@pytest.mark.parametrize(
    ("requested_device", "failed_device"),
    [
        (ExecutionDevice.CUDA, ExecutionDevice.CUDA),
        (ExecutionDevice.METAL, ExecutionDevice.METAL),
        (ExecutionDevice.AUTO, ExecutionDevice.CUDA),
        (ExecutionDevice.AUTO, ExecutionDevice.METAL),
    ],
)
def test_coordinator_device_policy_allows_only_one_recycled_accelerator_initialization_retry(
    requested_device: ExecutionDevice,
    failed_device: ExecutionDevice,
) -> None:
    policy = device_retry_policy_for_failure(
        requested_device=requested_device,
        failed_device=failed_device,
        origin=DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION,
        retry_device=ExecutionDevice.CPU,
        worker_will_recycle=True,
    )

    assert policy == DeviceRetryPolicy(
        retry_device=ExecutionDevice.CPU,
        max_retries=1,
        requires_worker_recycling=True,
        same_provider_model_only=True,
    )


@pytest.mark.parametrize(
    (
        "requested_device",
        "failed_device",
        "origin",
        "retry_device",
        "worker_will_recycle",
    ),
    [
        (
            ExecutionDevice.CPU,
            ExecutionDevice.CPU,
            DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION,
            ExecutionDevice.CPU,
            True,
        ),
        (
            ExecutionDevice.AUTO,
            ExecutionDevice.CPU,
            DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION,
            ExecutionDevice.CPU,
            True,
        ),
        (
            ExecutionDevice.AUTO,
            ExecutionDevice.AUTO,
            DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION,
            ExecutionDevice.CPU,
            True,
        ),
        (
            ExecutionDevice.CUDA,
            ExecutionDevice.METAL,
            DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION,
            ExecutionDevice.CPU,
            True,
        ),
        (
            ExecutionDevice.CUDA,
            ExecutionDevice.CUDA,
            DeviceFailureOrigin.INFERENCE,
            ExecutionDevice.CPU,
            True,
        ),
        (
            ExecutionDevice.METAL,
            ExecutionDevice.METAL,
            DeviceFailureOrigin.ENGINE_CRASH,
            ExecutionDevice.CPU,
            True,
        ),
        (
            ExecutionDevice.CUDA,
            ExecutionDevice.CUDA,
            DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION,
            ExecutionDevice.CUDA,
            True,
        ),
        (
            ExecutionDevice.METAL,
            ExecutionDevice.METAL,
            DeviceFailureOrigin.EXECUTION_PROVIDER_INITIALIZATION,
            ExecutionDevice.CPU,
            False,
        ),
    ],
)
def test_coordinator_device_policy_fails_closed_for_every_other_case(
    requested_device: ExecutionDevice,
    failed_device: ExecutionDevice,
    origin: DeviceFailureOrigin,
    retry_device: ExecutionDevice,
    worker_will_recycle: bool,
) -> None:
    policy = device_retry_policy_for_failure(
        requested_device=requested_device,
        failed_device=failed_device,
        origin=origin,
        retry_device=retry_device,
        worker_will_recycle=worker_will_recycle,
    )

    assert policy == DeviceRetryPolicy.no_retry()
