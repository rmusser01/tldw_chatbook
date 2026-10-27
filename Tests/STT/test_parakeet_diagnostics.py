"""Content-free Parakeet failure context through the production worker boundary."""

from __future__ import annotations

import pickle
import shutil
import subprocess
import threading
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V2_MODEL
from tldw_chatbook.STT.contracts import (
    BufferAudioSource,
    ExecutionDevice,
    FileAudioSource,
    TranscriptionFailureCode,
)
from tldw_chatbook.STT.executor import (
    ExecutorFailure,
    ExecutorRequest,
    LocalSTTExecutor,
    ModelIdentity,
    snapshot_local_source,
)
from tldw_chatbook.STT.executor_worker import (
    _failure_from_exception,
    _failure_from_worker_exception,
    _parakeet_provider,
)

SECRET = "private-transcript C:/private/model.onnx https://private.invalid/key"


class _NativeModel:
    def __init__(self, *, error=None, result="safe transcript") -> None:
        self.error = error
        self.result = result
        self.asr = self

    def recognize(self, *_args, **_kwargs):
        if self.error is not None:
            raise self.error
        return self.result

    def resampler(self, waveforms, lengths, _sample_rate):
        return waveforms, lengths

    def _get_sample_rate(self):
        return 16_000

    def recognize_batch(self, *_args):
        if self.error is not None:
            raise self.error
        return iter((SimpleNamespace(text=self.result),))


def _request(tmp_path: Path, *, buffer: bool = False) -> ExecutorRequest:
    return ExecutorRequest(
        generation=1,
        attempt_id="diagnostic-attempt",
        job_id="diagnostic-job",
        source=(
            BufferAudioSource(b"\0\0\1\0", 16_000)
            if buffer
            else FileAudioSource(tmp_path / "source.wav")
        ),
        identity=ModelIdentity(
            provider_id="parakeet-onnx",
            model_id=PARAKEET_V2_MODEL,
            root_revision=None,
            closure_fingerprint=None,
            precision="int8",
            device=ExecutionDevice.CPU,
        ),
        options={"language": "en"},
    )


def _native_api(monkeypatch, *, model=None, load_error=None, vad_error=None):
    from tldw_chatbook.STT import parakeet_onnx

    def load(*_args, **_kwargs):
        if load_error is not None:
            raise load_error
        return model or _NativeModel()

    def segment(*_args):
        if vad_error is not None:
            raise vad_error
        return iter((iter(((0, 2),)),))

    vad = SimpleNamespace(segment_batch=segment)
    monkeypatch.setattr(
        parakeet_onnx,
        "_onnx_asr_api",
        lambda: (
            SimpleNamespace(load_model=load, load_vad=lambda *_a, **_k: vad),
            lambda *_a, **_k: ([np.zeros(2)], [2], 16_000),
            lambda chunks: (chunks, [len(chunk) for chunk in chunks]),
        ),
    )
    return vad


def _wav(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0\1\0")


def _terminal(request, operation):
    try:
        operation()
    except Exception as error:
        return pickle.loads(pickle.dumps(_failure_from_exception(request, error)))
    raise AssertionError("expected a failed transcription")


def _assert_diagnostic(failure, stage, category):
    assert failure.diagnostic is not None
    assert failure.diagnostic.stage.value == stage
    assert failure.diagnostic.exception_category.value == category
    assert SECRET not in repr(failure)
    assert SECRET.encode() not in pickle.dumps(failure)
    assert "private" not in repr(failure)


@pytest.mark.parametrize("buffer", (False, True))
def test_native_runtime_error_survives_sanitized_worker_boundary_without_message(
    tmp_path,
    monkeypatch,
    buffer,
):
    request = _request(tmp_path, buffer=buffer)
    _wav(tmp_path / "source.wav")
    _native_api(monkeypatch, model=_NativeModel(error=RuntimeError(SECRET)))
    provider = _parakeet_provider(request, tmp_path, None, lambda: False)
    if buffer:

        def operation():
            return provider.buffer_runner(
                request.source,
                segment_end_frames=(),
                attempt_id=request.attempt_id,
                job_id=request.job_id,
                language="en",
                transcription_context={},
            )
    else:

        def operation():
            return provider.runner(str(request.source.path))

    failure = _terminal(request, operation)
    assert failure.code is TranscriptionFailureCode.INFERENCE_FAILED
    assert failure.recovery_actions == ("retry_faster_whisper",)
    _assert_diagnostic(failure, "inference", "RuntimeError")


@pytest.mark.parametrize(
    ("stage", "expected_category"),
    (
        ("model-load", "RuntimeError"),
        ("audio-decode", "wave.Error"),
        ("media-preparation", "subprocess.CalledProcessError"),
        ("vad", "RuntimeError"),
        ("normalization", "ValueError"),
        ("provenance", "ValueError"),
    ),
)
def test_real_runtime_isolates_stage_before_worker_sanitization(
    tmp_path,
    monkeypatch,
    stage,
    expected_category,
):
    class _InvalidText:
        def __str__(self):
            raise ValueError(SECRET)

    request = _request(tmp_path)
    _wav(request.source.path)
    _native_api(
        monkeypatch,
        model=_NativeModel(
            result=_InvalidText() if stage == "normalization" else "safe"
        ),
        load_error=RuntimeError(SECRET) if stage == "model-load" else None,
        vad_error=RuntimeError(SECRET) if stage == "vad" else None,
    )
    if stage == "audio-decode":
        request.source.path.write_bytes(b"invalid wav data")
    if stage == "media-preparation":
        request = replace(request, source=FileAudioSource(tmp_path / "source.mp3"))

        def fail_ffmpeg(*_args, **_kwargs):
            raise subprocess.CalledProcessError(1, SECRET, stderr=SECRET)

        monkeypatch.setattr(subprocess, "run", fail_ffmpeg)
        request = replace(request, options={"ffmpeg_path": "fake-ffmpeg"})

    def operation():
        provider = _parakeet_provider(request, tmp_path, None, lambda: False)
        return provider.runner(
            str(request.source.path),
            vad_filter=stage == "vad",
            retry_source_failure_provenance={"secret": SECRET}
            if stage == "provenance"
            else None,
        )

    if stage == "vad":
        # Model loading is real; supply a managed VAD through the native API result.
        from tldw_chatbook.Model_Artifacts import ArtifactRef

        root = ArtifactRef("parakeet-v2", "root", "int8")
        dependency = ArtifactRef("silero-vad", "dependency", "f32")
        handle = SimpleNamespace(
            root=root,
            closure=(root, dependency),
            paths=((root, tmp_path), (dependency, tmp_path)),
        )
        provider = _parakeet_provider(request, tmp_path, handle, lambda: False)
        failure = _terminal(
            request, lambda: provider.runner(str(request.source.path), vad_filter=True)
        )
    else:
        failure = _terminal(request, operation)
    expected_code = (
        TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE
        if stage == "model-load"
        else TranscriptionFailureCode.INFERENCE_FAILED
    )
    assert failure.code is expected_code
    _assert_diagnostic(failure, stage, expected_category)


def test_untrusted_exception_class_name_is_not_diagnostic_metadata(
    tmp_path, monkeypatch
):
    attacker_error = type("private_attacker_controlled_class", (RuntimeError,), {})
    request = _request(tmp_path)
    _wav(request.source.path)
    _native_api(monkeypatch, model=_NativeModel(error=attacker_error(SECRET)))
    provider = _parakeet_provider(request, tmp_path, None, lambda: False)
    failure = _terminal(request, lambda: provider.runner(str(request.source.path)))
    _assert_diagnostic(failure, "inference", "unknown")


def test_diagnostic_protocol_rejects_arbitrary_text(tmp_path):
    from tldw_chatbook.STT.executor import STTFailureDiagnostic
    from tldw_chatbook.STT.failure_diagnostics import (
        STTExceptionCategory,
        STTFailureStage,
    )

    with pytest.raises(TypeError):
        STTFailureDiagnostic(SECRET, STTExceptionCategory.UNKNOWN)
    with pytest.raises(TypeError):
        STTFailureDiagnostic(STTFailureStage.INFERENCE, SECRET)
    with pytest.raises(TypeError):
        ExecutorFailure(
            1,
            "attempt",
            TranscriptionFailureCode.INFERENCE_FAILED,
            diagnostic={"stage": SECRET},
        )


def test_cancellation_never_acquires_failure_diagnostic(tmp_path, monkeypatch):
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxCancelled

    request = _request(tmp_path)
    _wav(request.source.path)
    _native_api(monkeypatch)
    provider = _parakeet_provider(request, tmp_path, None, lambda: True)
    with pytest.raises(ParakeetOnnxCancelled) as raised:
        provider.runner(str(request.source.path))
    failure = _failure_from_worker_exception(request, raised.value, cancelled=True)
    assert failure.code is TranscriptionFailureCode.CANCELLED
    assert failure.recovery_actions == ()
    assert failure.diagnostic is None


def test_real_spawned_worker_delivers_typed_native_failure_diagnostic(tmp_path):
    from Tests.STT.executor_test_support import parakeet_diagnostic_executor_worker

    model = tmp_path / "model.onnx"
    model.write_bytes(b"native model double")
    snapshot = snapshot_local_source((model,))
    request = _request(tmp_path, buffer=True)
    identity = replace(request.identity, local_snapshot_token=snapshot.token)
    terminal = threading.Event()
    failures = []

    def on_failure(failure):
        failures.append(failure)
        terminal.set()

    executor = LocalSTTExecutor(
        worker_target=parakeet_diagnostic_executor_worker,
        startup_timeout=10.0,
        graceful_shutdown_timeout=0.2,
        force_stop_timeout=2.0,
    )
    try:
        executor.submit(
            attempt_id=request.attempt_id,
            job_id=request.job_id,
            source=request.source,
            identity=identity,
            options=request.options,
            local_source=snapshot,
            on_event=lambda _event: None,
            on_result=lambda _result: terminal.set(),
            on_failure=on_failure,
        )
        assert terminal.wait(15.0)
        assert len(failures) == 1
        _assert_diagnostic(failures[0], "inference", "RuntimeError")
    finally:
        executor.close()


@pytest.mark.parametrize("suffix", (".wav", ".mp4"))
@pytest.mark.bootstrap_profile
def test_file_parser_preserves_typed_diagnostic_across_audio_and_video_results(
    tmp_path,
    monkeypatch,
    suffix,
):
    from tldw_chatbook.STT.executor_worker import _default_parse_job
    from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
        DirectLocalSTTIngestError,
    )

    source = tmp_path / ("source" + suffix)
    _wav(source)
    request = replace(
        _request(tmp_path),
        source=FileAudioSource(source),
        options={
            "transcription_provider": "parakeet-onnx",
            "transcription_model": PARAKEET_V2_MODEL,
            "language": "en",
            "perform_analysis": False,
        },
    )
    _native_api(monkeypatch, model=_NativeModel(error=RuntimeError(SECRET)))
    if suffix == ".mp4":
        original_which = shutil.which
        monkeypatch.setattr(
            shutil,
            "which",
            lambda command, *args, **kwargs: "fake-ffmpeg"
            if command == "ffmpeg"
            else original_which(command, *args, **kwargs),
        )

        def ffmpeg(command, **_kwargs):
            _wav(Path(command[-1]))
            return subprocess.CompletedProcess(command, 0, "", "")

        monkeypatch.setattr(subprocess, "run", ffmpeg)
    provider = _parakeet_provider(request, tmp_path, None, lambda: False)
    with pytest.raises(DirectLocalSTTIngestError) as raised:
        _default_parse_job(
            source, dict(request.options), transcription_runner=provider.runner
        )
    failure = pickle.loads(pickle.dumps(_failure_from_exception(request, raised.value)))
    assert failure.code is TranscriptionFailureCode.INFERENCE_FAILED
    _assert_diagnostic(failure, "inference", "RuntimeError")


@pytest.mark.parametrize(
    ("module", "name", "expected"),
    (
        (
            "onnxruntime.capi.onnxruntime_pybind11_state",
            "InvalidArgument",
            "onnxruntime.InvalidArgument",
        ),
        (
            "onnxruntime.capi.onnxruntime_pybind11_state",
            "private_attacker_name",
            "unknown",
        ),
        ("untrusted_plugin", "RuntimeError", "unknown"),
        ("untrusted_plugin", "InvalidArgument", "unknown"),
    ),
)
def test_exception_categories_admit_only_fixed_native_module_and_names(
    tmp_path,
    monkeypatch,
    module,
    name,
    expected,
):
    native_type = type(name, (Exception,), {"__module__": module})
    request = _request(tmp_path)
    _wav(request.source.path)
    _native_api(monkeypatch, model=_NativeModel(error=native_type(SECRET)))
    provider = _parakeet_provider(request, tmp_path, None, lambda: False)
    failure = _terminal(request, lambda: provider.runner(str(request.source.path)))
    _assert_diagnostic(failure, "inference", expected)


def test_lazy_vad_failure_keeps_segment_order_and_original_stage(tmp_path, monkeypatch):
    from tldw_chatbook.Model_Artifacts import ArtifactRef

    model = _NativeModel()
    vad = _native_api(monkeypatch, model=model)
    recognized = []

    def recognize_batch(*_args):
        recognized.append(True)
        return iter((SimpleNamespace(text="safe"),))

    def ranges():
        yield (0, 1)
        assert recognized, "native VAD iteration must retain its existing lazy order"
        raise RuntimeError(SECRET)

    model.recognize_batch = recognize_batch
    vad.segment_batch = lambda *_args: iter((ranges(),))
    root = ArtifactRef("parakeet-v2", "root", "int8")
    dependency = ArtifactRef("silero-vad", "dependency", "f32")
    handle = SimpleNamespace(
        root=root,
        closure=(root, dependency),
        paths=((root, tmp_path), (dependency, tmp_path)),
    )
    request = _request(tmp_path)
    _wav(request.source.path)
    provider = _parakeet_provider(request, tmp_path, handle, lambda: False)
    failure = _terminal(
        request,
        lambda: provider.runner(str(request.source.path), vad_filter=True),
    )
    assert recognized == [True]
    _assert_diagnostic(failure, "vad", "RuntimeError")
