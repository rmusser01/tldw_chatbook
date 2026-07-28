"""Tests for the temporary adapter over retained transcription providers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.STT.contracts import (
    BufferAudioSource,
    CancellationGranularity,
    ExecutionDevice,
    FileAudioSource,
    InputKind,
    LanguageInputMode,
    ResolvedTranscriptionRequest,
    TimestampGranularity,
    TranscriptionPhase,
    TranscriptionProgress,
    TranscriptionRequest,
    TranscriptionTask,
)
from tldw_chatbook.STT.legacy_bridge import (
    LegacyTranscriptionBridge,
    LegacyTranscriptionBridgeError,
)
from tldw_chatbook.STT.registry import (
    CapabilitySet,
    ModelMetadata,
    ProviderMetadata,
)


def _capabilities() -> CapabilitySet:
    return CapabilitySet(
        languages=frozenset({"en", "fr"}),
        automatic_language=True,
        tasks=frozenset({TranscriptionTask.TRANSCRIBE, TranscriptionTask.TRANSLATE}),
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
        cancellation=CancellationGranularity.BEFORE_EXECUTION,
        vad=True,
        diarization=True,
        punctuation=True,
        capitalization=True,
        language_input_mode=LanguageInputMode.AUTOMATIC,
        execution_devices=frozenset({ExecutionDevice.CPU}),
        precisions=frozenset({"int8"}),
    )


def _provider() -> ProviderMetadata:
    return ProviderMetadata(
        provider_id="retained-whisper",
        display_name="Retained Whisper",
        local_processing=True,
    )


def _model() -> ModelMetadata:
    return ModelMetadata(
        provider_id="retained-whisper",
        model_id="base",
        display_name="Base",
        capabilities=_capabilities(),
        default_precision="int8",
        semantic_default_eligible=False,
        enforces_language_hint=True,
    )


def _request(
    source: FileAudioSource | BufferAudioSource,
    *,
    task: TranscriptionTask = TranscriptionTask.TRANSCRIBE,
    progress: object = None,
) -> ResolvedTranscriptionRequest:
    request = TranscriptionRequest(
        attempt_id="attempt-1",
        batch_id="batch-1",
        job_id="job-1",
        source=source,
        provider_id="retained-whisper",
        model_id="base",
        language="fr",
        task=task,
        precision="int8",
        device=ExecutionDevice.CPU,
        vad=True,
        diarization=True,
        progress=progress,  # type: ignore[arg-type]
    )
    return ResolvedTranscriptionRequest(
        request=request,
        provider_id="retained-whisper",
        model_id="base",
        requested_language="fr",
        effective_language="fr",
        precision="int8",
    )


class _Backend:
    def __init__(self) -> None:
        self.config = {
            "default_provider": "faster-whisper",
            "default_language": "en",
        }
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


def _bridge(backend: _Backend) -> LegacyTranscriptionBridge:
    return LegacyTranscriptionBridge(
        backend_factory=lambda: backend,
        provider_metadata=_provider(),
        models=(_model(),),
        legacy_provider_id="faster-whisper",
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
    assert output.detected_language == "fr"
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


def test_bridge_normalizes_legacy_unknown_language_values() -> None:
    backend = _Backend()
    backend.result["language"] = "unknown"
    backend.result["detected_language"] = "unknown"

    output = _bridge(backend).transcribe(
        _request(FileAudioSource(Path("/tmp/example.wav")))
    )

    assert output.effective_language == "fr"
    assert output.detected_language is None


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
    )

    bridge.close()

    assert constructed is False
