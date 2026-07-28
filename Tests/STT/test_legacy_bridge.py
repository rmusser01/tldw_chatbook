"""Tests for the temporary adapter over retained transcription providers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from tldw_chatbook.STT.contracts import (
    BufferAudioSource,
    CancellationGranularity,
    ExecutionDevice,
    FileAudioSource,
    InputKind,
    LanguageInputMode,
    PipelineCapabilities,
    ResolvedTranscriptionRequest,
    TimestampGranularity,
    TranscriptionFailureCode,
    TranscriptionPhase,
    TranscriptionProgress,
    TranscriptionRequest,
    TranscriptionTask,
)
from tldw_chatbook.STT.coordinator import (
    TranscriptionCoordinator,
    TranscriptionCoordinatorError,
)
from tldw_chatbook.STT.legacy_bridge import (
    LegacyTranscriptionBridge,
    LegacyTranscriptionBridgeError,
)
from tldw_chatbook.STT.registry import (
    CapabilitySet,
    CatalogDeclarations,
    ModelMetadata,
    ProviderMetadata,
    ProviderRegistry,
)
from tldw_chatbook.STT.routing import RoutingPolicy, TranscriptionRouter


def _capabilities(
    *,
    devices: frozenset[ExecutionDevice] = frozenset({ExecutionDevice.CPU}),
    precisions: frozenset[str] = frozenset({"int8"}),
) -> CapabilitySet:
    return CapabilitySet(
        languages=frozenset({"en", "fr"}),
        automatic_language=True,
        tasks=frozenset({TranscriptionTask.TRANSCRIBE, TranscriptionTask.TRANSLATE}),
        inputs=frozenset({InputKind.FILE, InputKind.BUFFER}),
        timestamps=frozenset(
            {
                TimestampGranularity.NONE,
                TimestampGranularity.SEGMENT,
            }
        ),
        true_streaming=False,
        batch=True,
        cancellation=CancellationGranularity.BEFORE_EXECUTION,
        vad=True,
        diarization=True,
        punctuation=True,
        capitalization=True,
        language_input_mode=LanguageInputMode.AUTOMATIC,
        execution_devices=devices,
        precisions=precisions,
    )


def _provider() -> ProviderMetadata:
    return ProviderMetadata(
        provider_id="retained-whisper",
        display_name="Retained Whisper",
        local_processing=True,
    )


def _model(
    capabilities: CapabilitySet | None = None,
    *,
    default_precision: str = "int8",
) -> ModelMetadata:
    return ModelMetadata(
        provider_id="retained-whisper",
        model_id="base",
        display_name="Base",
        capabilities=capabilities or _capabilities(),
        default_precision=default_precision,
        semantic_default_eligible=False,
        enforces_language_hint=True,
    )


def _request(
    source: FileAudioSource | BufferAudioSource,
    *,
    task: TranscriptionTask = TranscriptionTask.TRANSCRIBE,
    language: str = "fr",
    precision: str = "int8",
    device: ExecutionDevice = ExecutionDevice.CPU,
    timestamps: TimestampGranularity = TimestampGranularity.SEGMENT,
    diarization: bool = True,
    progress: object = None,
) -> ResolvedTranscriptionRequest:
    request = TranscriptionRequest(
        attempt_id="attempt-1",
        batch_id="batch-1",
        job_id="job-1",
        source=source,
        provider_id="retained-whisper",
        model_id="base",
        language=language,
        task=task,
        precision=precision,
        device=device,
        timestamps=timestamps,
        vad=True,
        diarization=diarization,
        progress=progress,  # type: ignore[arg-type]
    )
    return ResolvedTranscriptionRequest(
        request=request,
        provider_id="retained-whisper",
        model_id="base",
        requested_language=language,
        effective_language=language,
        precision=precision,
    )


class _Backend:
    def __init__(self) -> None:
        self.config = {
            "default_provider": "faster-whisper",
            "default_language": "en",
            "device": "cpu",
            "compute_type": "int8",
        }
        self.execution = {
            "device": ExecutionDevice.CPU,
            "precision": "int8",
        }
        self.on_transcribe: Callable[[], None] | None = None
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.result: dict[str, Any] = {
            "text": "bonjour",
            "segments": [
                {
                    "Time_Start": 0.0,
                    "Time_End": 1.25,
                    "Text": "bonjour",
                    "speaker_label": "speaker-1",
                }
            ],
            "language": "fr",
            "detected_language": "fr",
            "device": "cpu",
            "duration": 1.25,
            "diarization_performed": True,
        }
        self.error: Exception | None = None

    def transcribe(self, *args: object, **kwargs: object) -> dict[str, Any]:
        self.calls.append(("transcribe", args, kwargs))
        if self.on_transcribe is not None:
            self.on_transcribe()
        callback = kwargs.get("progress_callback")
        if callable(callback):
            callback(25, "path=/private/audio.wav token=secret", {"secret": "value"})
        if self.error is not None:
            raise self.error
        return self.result

    def transcribe_buffer(self, *args: object, **kwargs: object) -> dict[str, Any]:
        self.calls.append(("transcribe_buffer", args, kwargs))
        callback = kwargs.get("progress_callback")
        if callable(callback):
            callback(50, "buffer token=secret", None)
        if self.error is not None:
            raise self.error
        return self.result

    def get_available_providers(self) -> list[str]:
        self.calls.append(("get_available_providers", (), {}))
        return ["faster-whisper"]

    def list_available_models(
        self, provider: str | None = None
    ) -> dict[str, list[str]]:
        self.calls.append(("list_available_models", (provider,), {}))
        return {"faster-whisper": ["base"]}

    def cleanup(self) -> None:
        self.calls.append(("cleanup", (), {}))


def _bridge(
    backend: _Backend,
    model: ModelMetadata | None = None,
) -> LegacyTranscriptionBridge:
    return LegacyTranscriptionBridge(
        backend_factory=lambda: backend,
        provider_metadata=_provider(),
        models=(model or _model(),),
        legacy_provider_id="faster-whisper",
        execution_observer=lambda retained: (
            retained.execution["device"],
            retained.execution["precision"],
        ),
    )


def _coordinator(
    backend: _Backend,
    model: ModelMetadata | None = None,
) -> TranscriptionCoordinator:
    selected_model = model or _model()
    bridge = _bridge(backend, selected_model)
    registry = ProviderRegistry.sealed(
        CatalogDeclarations(
            providers=(_provider(),),
            models=(selected_model,),
        ),
        adapters=(bridge,),
    )
    return TranscriptionCoordinator(
        registry=registry,
        router=TranscriptionRouter(
            RoutingPolicy(validated_v3_languages=frozenset({"fr"}))
        ),
        pipeline=PipelineCapabilities(),
    )


def test_bridge_module_does_not_import_the_legacy_service() -> None:
    command = (
        "import sys; import tldw_chatbook.STT.legacy_bridge; "
        "assert 'tldw_chatbook.Local_Ingestion.transcription_service' "
        "not in sys.modules"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())

    completed = subprocess.run(
        [sys.executable, "-c", command],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_bridge_exposes_exact_metadata_and_probe_mapping() -> None:
    backend = _Backend()
    bridge = _bridge(backend)

    assert bridge.provider() == _provider()
    assert bridge.describe() == (_model(),)
    observation = bridge.probe("base")

    assert observation.provider_id == "retained-whisper"
    assert observation.model_id == "base"
    assert observation.available is True
    assert observation.capabilities == _capabilities()
    assert backend.calls == [
        ("get_available_providers", (), {}),
        ("list_available_models", ("faster-whisper",), {}),
    ]


def test_adapter_binding_requires_an_exact_execution_observer() -> None:
    with pytest.raises(ValueError, match="execution observer"):
        LegacyTranscriptionBridge(
            backend_factory=_Backend,
            provider_metadata=_provider(),
            models=(_model(),),
            legacy_provider_id="faster-whisper",
        )


def test_bridge_converts_file_request_and_normalizes_legacy_dictionary() -> None:
    backend = _Backend()
    events: list[TranscriptionProgress] = []
    resolved = _request(
        FileAudioSource(Path("/tmp/example.wav")), progress=events.append
    )

    output = _bridge(backend).transcribe(resolved)

    method, args, kwargs = backend.calls[-1]
    assert method == "transcribe"
    assert args == ()
    assert kwargs == {
        "audio_path": "/tmp/example.wav",
        "provider": "faster-whisper",
        "model": "base",
        "language": "fr",
        "source_lang": "fr",
        "target_lang": None,
        "vad_filter": True,
        "diarize": True,
        "progress_callback": kwargs["progress_callback"],
        "batch_route_resolved": True,
    }
    assert output.text == "bonjour"
    assert output.effective_language == "fr"
    assert output.detected_language is None
    assert output.effective_device is ExecutionDevice.CPU
    assert output.duration_seconds == 1.25
    assert output.segments[0].start_seconds == 0.0
    assert output.segments[0].end_seconds == 1.25
    assert output.segments[0].text == "bonjour"
    assert output.segments[0].speaker == "speaker-1"
    assert output.produced_capabilities.diarization is True
    assert events == [
        TranscriptionProgress(
            attempt_id="attempt-1",
            batch_id="batch-1",
            job_id="job-1",
            phase=TranscriptionPhase.TRANSCRIBING,
            fraction=0.25,
        )
    ]


def test_bridge_converts_buffer_request_without_disk_staging() -> None:
    backend = _Backend()
    source = BufferAudioSource(
        audio=b"\x00\x01\x02\x03",
        sample_rate=16_000,
        channels=2,
        sample_width=1,
    )

    _bridge(backend).transcribe(_request(source))

    assert backend.calls[-1] == (
        "transcribe_buffer",
        (),
        {
            "audio_data": b"\x00\x01\x02\x03",
            "sample_rate": 16_000,
            "channels": 2,
            "sample_width": 1,
            "provider": "faster-whisper",
            "model": "base",
            "language": "fr",
            "vad_filter": True,
            "diarize": True,
            "task": "transcribe",
        },
    )


def test_bridge_forwards_translation_to_the_legacy_provider_explicitly() -> None:
    backend = _Backend()

    _bridge(backend).transcribe(
        _request(
            FileAudioSource(Path("/tmp/example.wav")),
            task=TranscriptionTask.TRANSLATE,
        )
    )

    assert backend.calls[-1][2]["target_lang"] == "en"


def test_bridge_forwards_buffer_translation_task_explicitly() -> None:
    backend = _Backend()

    _bridge(backend).transcribe(
        _request(
            BufferAudioSource(b"\x00\x00", 16_000),
            task=TranscriptionTask.TRANSLATE,
        )
    )

    assert backend.calls[-1][2]["task"] == "translate"


def test_bridge_converts_buffer_progress_without_forwarding_status_text() -> None:
    backend = _Backend()
    events: list[TranscriptionProgress] = []

    _bridge(backend).transcribe(
        _request(
            BufferAudioSource(b"\x00\x00", 16_000),
            progress=events.append,
        )
    )

    assert events == [
        TranscriptionProgress(
            attempt_id="attempt-1",
            batch_id="batch-1",
            job_id="job-1",
            phase=TranscriptionPhase.TRANSCRIBING,
            fraction=0.5,
        )
    ]


def test_bridge_sanitizes_backend_errors() -> None:
    backend = _Backend()
    backend.error = RuntimeError(
        "failed at /Users/person/private.wav with token super-secret"
    )

    with pytest.raises(LegacyTranscriptionBridgeError) as caught:
        _bridge(backend).transcribe(_request(FileAudioSource(Path("/tmp/example.wav"))))

    rendered = f"{caught.value!s} {caught.value!r}"
    assert rendered == (
        "The retained speech-to-text provider failed. LegacyTranscriptionBridgeError()"
    )
    assert "private.wav" not in rendered
    assert "super-secret" not in rendered


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    [
        ("provider", "remote-whisper"),
        ("model", "large"),
    ],
)
def test_bridge_rejects_mismatched_optional_result_identity(
    field_name: str,
    wrong_value: str,
) -> None:
    backend = _Backend()
    backend.result[field_name] = wrong_value

    with pytest.raises(LegacyTranscriptionBridgeError) as caught:
        _bridge(backend).transcribe(_request(FileAudioSource(Path("/tmp/input.wav"))))

    assert str(caught.value) == "The retained speech-to-text provider failed."


def test_bridge_normalizes_legacy_unknown_language_values() -> None:
    backend = _Backend()
    backend.result["language"] = "unknown"
    backend.result["detected_language"] = "unknown"

    output = _bridge(backend).transcribe(
        _request(FileAudioSource(Path("/tmp/example.wav")))
    )

    assert output.effective_language == "fr"
    assert output.detected_language is None


def test_coordinator_bridge_auto_language_records_concrete_detection() -> None:
    backend = _Backend()
    backend.result.pop("detected_language")
    request = _request(
        FileAudioSource(Path("/tmp/example.wav")),
        language="auto",
    ).request

    result = _coordinator(backend).transcribe(request)

    assert result.provenance.requested_language == "auto"
    assert result.provenance.effective_language == "auto"
    assert result.provenance.detected_language == "fr"


def test_coordinator_bridge_explicit_language_does_not_report_detection() -> None:
    backend = _Backend()
    request = _request(FileAudioSource(Path("/tmp/example.wav"))).request

    result = _coordinator(backend).transcribe(request)

    assert result.provenance.requested_language == "fr"
    assert result.provenance.effective_language == "fr"
    assert result.provenance.detected_language is None


def test_coordinator_bridge_uses_observed_nondefault_device_and_precision() -> None:
    capabilities = _capabilities(
        devices=frozenset({ExecutionDevice.CPU, ExecutionDevice.CUDA}),
        precisions=frozenset({"int8", "float32"}),
    )
    model = _model(capabilities)
    backend = _Backend()
    backend.execution["device"] = ExecutionDevice.CUDA
    backend.execution["precision"] = "float32"
    backend.result.pop("device")
    request = _request(
        FileAudioSource(Path("/tmp/example.wav")),
        device=ExecutionDevice.CUDA,
        precision="float32",
    ).request

    result = _coordinator(backend, model).transcribe(request)

    assert result.provenance.requested_device is ExecutionDevice.CUDA
    assert result.provenance.effective_device is ExecutionDevice.CUDA
    assert result.provenance.precision == "float32"
    assert backend.execution["device"] is ExecutionDevice.CUDA
    assert backend.execution["precision"] == "float32"
    assert backend.config["device"] == "cpu"
    assert backend.config["compute_type"] == "int8"


def test_coordinator_bridge_rejects_unobserved_precision() -> None:
    capabilities = _capabilities(
        precisions=frozenset({"int8", "float32"}),
    )
    model = _model(capabilities)
    backend = _Backend()
    request = _request(
        FileAudioSource(Path("/tmp/example.wav")),
        precision="float32",
    ).request

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        _coordinator(backend, model).transcribe(request)

    assert caught.value.failure.phase is TranscriptionPhase.LOADING


def test_coordinator_bridge_does_not_guess_a_missing_device() -> None:
    backend = _Backend()
    backend.execution.pop("device")
    request = _request(FileAudioSource(Path("/tmp/example.wav"))).request

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        _coordinator(backend).transcribe(request)

    assert caught.value.failure.phase is TranscriptionPhase.LOADING


def test_coordinator_bridge_allows_none_timestamps_without_incidental_segments() -> (
    None
):
    backend = _Backend()
    backend.result["diarization_performed"] = False
    request = _request(
        FileAudioSource(Path("/tmp/example.wav")),
        timestamps=TimestampGranularity.NONE,
        diarization=False,
    ).request

    result = _coordinator(backend).transcribe(request)

    assert result.segments == ()
    assert result.produced_capabilities.timestamps is TimestampGranularity.NONE
    assert any(call[0] == "transcribe" for call in backend.calls)


def test_coordinator_bridge_rejects_none_timestamps_with_diarization_before_execution() -> (
    None
):
    backend = _Backend()
    request = _request(
        FileAudioSource(Path("/tmp/example.wav")),
        timestamps=TimestampGranularity.NONE,
        diarization=True,
    ).request

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        _coordinator(backend).transcribe(request)

    assert caught.value.failure.code is TranscriptionFailureCode.INFERENCE_FAILED
    assert caught.value.failure.phase is TranscriptionPhase.TRANSCRIBING
    assert not any(
        call[0] in {"transcribe", "transcribe_buffer"} for call in backend.calls
    )


def test_coordinator_bridge_rejects_word_timestamps_without_execution() -> None:
    backend = _Backend()
    request = _request(
        FileAudioSource(Path("/tmp/example.wav")),
        timestamps=TimestampGranularity.WORD,
    ).request

    with pytest.raises(TranscriptionCoordinatorError) as caught:
        _coordinator(backend).transcribe(request)

    assert caught.value.failure.phase is TranscriptionPhase.QUEUED
    assert backend.calls == []


def test_bridge_fails_if_execution_snapshot_changes_during_call() -> None:
    backend = _Backend()

    def change_execution() -> None:
        backend.execution["precision"] = "float32"

    backend.on_transcribe = change_execution

    with pytest.raises(LegacyTranscriptionBridgeError) as caught:
        _bridge(backend).transcribe(_request(FileAudioSource(Path("/tmp/input.wav"))))

    assert str(caught.value) == "The retained speech-to-text provider failed."


def test_bridge_config_can_be_replaced_explicitly() -> None:
    backend = _Backend()
    bridge = LegacyTranscriptionBridge(backend_factory=lambda: backend)
    replacement = {"default_provider": "remote-whisper"}

    bridge.config = replacement

    assert bridge.config is replacement


def test_close_does_not_construct_an_unused_backend() -> None:
    constructed = False

    def factory() -> _Backend:
        nonlocal constructed
        constructed = True
        return _Backend()

    bridge = LegacyTranscriptionBridge(
        backend_factory=factory,
        provider_metadata=_provider(),
        models=(_model(),),
        legacy_provider_id="faster-whisper",
        execution_observer=lambda retained: (
            retained.execution["device"],
            retained.execution["precision"],
        ),
    )

    bridge.close()

    assert constructed is False
