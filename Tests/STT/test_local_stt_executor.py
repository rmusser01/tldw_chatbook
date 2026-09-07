from __future__ import annotations

import os
import pickle
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Model_Artifacts.test_service import (
    install_descriptor_payload,
    installed_root_and_dependency,
    single_file_descriptor,
)
from Tests.STT.executor_test_support import (
    device_retry_executor_worker,
    fake_executor_worker,
    private_log_executor_worker,
    protocol_executor_worker,
    resident_executor_worker,
)
from tldw_chatbook.Local_Ingestion.stt_batch_routing import (
    PARAKEET_V2_MODEL,
    PARAKEET_V3_MODEL,
)
from tldw_chatbook.Model_Artifacts import (
    ArtifactInUseError,
    ArtifactOperationLease,
    ArtifactRef,
    ArtifactRole,
    LeaseMode,
    ModelArtifactService,
)
from tldw_chatbook.STT.contracts import (
    BufferAudioSource,
    DeviceFailureOrigin,
    ExecutionDevice,
    FileAudioSource,
    ProducedCapabilities,
    TimestampGranularity,
    TranscriptionFailureCode,
    TranscriptionProvenance,
    TranscriptionResult,
    TranscriptionTask,
    TranscriptionTimings,
)
from tldw_chatbook.STT.executor import (
    ExecutorBusyError,
    ExecutorEvent,
    ExecutorFailure,
    ExecutorRequest,
    ExecutorResident,
    ExecutorResult,
    ExecutorUnavailableError,
    LocalSourceChangedError,
    LocalSourceSnapshot,
    LocalSTTExecutor,
    ModelIdentity,
    WorkerPhase,
    _AttemptTerminalGuard,
    snapshot_local_source,
    validate_local_source_snapshot,
)
from tldw_chatbook.STT.executor_worker import (
    ProviderRuntime,
    _default_parse_job,
    _failure_from_exception,
    _failure_from_worker_exception,
    _load_resident,
    _parakeet_provider,
    _ProviderLoadFailure,
    _run_executor_worker,
    _validate_reuse,
)

FAST_LEASE_TIMEOUT_SECONDS = 0.01


def _identity(**overrides: object) -> ModelIdentity:
    values: dict[str, object] = {
        "provider_id": "parakeet-onnx",
        "model_id": "nemo-parakeet-tdt-0.6b-v2",
        "root_revision": "revision-a",
        "closure_fingerprint": "fingerprint-a",
        "precision": "int8",
        "device": ExecutionDevice.CPU,
        "local_snapshot_token": "private-snapshot-token",
    }
    values.update(overrides)
    return ModelIdentity(**values)


def _request() -> ExecutorRequest:
    return ExecutorRequest(
        generation=3,
        attempt_id="attempt-1",
        job_id="job-1",
        source=FileAudioSource(Path("/private/media/interview.wav")),
        identity=_identity(local_snapshot_token=None),
        options={"transcription_model_dir": "/private/models/parakeet"},
        managed_store_root=Path("/private/models/managed"),
        managed_artifact_ref=("parakeet-v2", "revision-a", "int8"),
    )


def test_protocol_objects_are_frozen_slotted_and_picklable() -> None:
    request = _request()
    lease_refs = (
        ("parakeet-v2", "revision-a", "int8"),
        ("silero-vad", "vad-revision", "f32"),
    )
    envelopes = (
        request,
        ExecutorEvent(3, "attempt-1", WorkerPhase.LOADING),
        ExecutorResident(3, "attempt-1", request.identity, lease_refs),
        ExecutorResult(3, "attempt-1", {"content": "hello"}),
        ExecutorFailure(
            generation=3,
            attempt_id="attempt-1",
            code=TranscriptionFailureCode.ENGINE_CRASHED,
            recovery_actions=("retry_faster_whisper",),
            failed_attempt={"attempt_id": "attempt-1"},
            device_failure_origin=DeviceFailureOrigin.ENGINE_CRASH,
        ),
    )

    assert all(pickle.loads(pickle.dumps(value)) == value for value in envelopes)
    assert all(hasattr(type(value), "__slots__") for value in envelopes)
    with pytest.raises(FrozenInstanceError):
        request.generation = 4  # type: ignore[misc]


def test_resident_managed_lease_references_require_canonical_tuples() -> None:
    with pytest.raises(ValueError, match="managed_lease_refs"):
        ExecutorResident(
            3,
            "attempt-1",
            _identity(),
            (("parakeet-v2", "", "int8"),),
        )


def test_executor_request_accepts_file_and_buffer_sources_without_a_job_id() -> None:
    file_request = ExecutorRequest(
        generation=3,
        attempt_id="file-attempt",
        job_id=None,
        source=FileAudioSource(Path("/private/media/interview.wav")),
        identity=_identity(),
        options={},
    )
    buffer_request = ExecutorRequest(
        generation=3,
        attempt_id="buffer-attempt",
        job_id=None,
        source=BufferAudioSource(b"\x00\x00\x01\x00", 16_000),
        identity=_identity(),
        options={},
        segment_end_frames=(2,),
    )

    assert file_request.job_id is None
    assert type(file_request.source) is FileAudioSource
    assert buffer_request.segment_end_frames == (2,)


@pytest.mark.parametrize(
    ("source", "segment_end_frames", "error"),
    [
        (Path("speech.wav"), (), TypeError),
        (FileAudioSource(Path("speech.wav")), (1,), ValueError),
        (BufferAudioSource(b"\x00\x00\x01\x00", 16_000), (0, 2), ValueError),
        (BufferAudioSource(b"\x00\x00\x01\x00", 16_000), (2, 2), ValueError),
        (BufferAudioSource(b"\x00\x00\x01\x00", 16_000), (3,), ValueError),
        (BufferAudioSource(b"\x00\x00\x01\x00", 16_000), (1,), ValueError),
    ],
)
def test_executor_request_rejects_invalid_source_or_buffer_boundaries(
    source: object,
    segment_end_frames: tuple[int, ...],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ExecutorRequest(
            generation=3,
            attempt_id="attempt-1",
            job_id=None,
            source=source,  # type: ignore[arg-type]
            identity=_identity(),
            options={},
            segment_end_frames=segment_end_frames,
        )


def test_executor_request_source_variants_pickle_without_leaking_private_inputs() -> (
    None
):
    snapshot = LocalSourceSnapshot(
        token="private-snapshot-token",
        paths=(Path("/private/models/model.onnx"),),
        identities=((7, 11, 1024, 123456),),
    )
    requests = (
        ExecutorRequest(
            generation=3,
            attempt_id="file-attempt",
            job_id=None,
            source=FileAudioSource(Path("/private/media/interview.wav")),
            identity=_identity(),
            options={"private": "/private/options"},
            local_source=snapshot,
        ),
        ExecutorRequest(
            generation=3,
            attempt_id="buffer-attempt",
            job_id=None,
            source=BufferAudioSource(b"private-pcm", 16_000, sample_width=1),
            identity=_identity(),
            options={},
            segment_end_frames=(11,),
        ),
    )

    assert all(pickle.loads(pickle.dumps(request)) == request for request in requests)
    rendered = "".join(repr(request) for request in requests)
    assert "/private/" not in rendered
    assert "private-pcm" not in rendered
    assert "private-snapshot-token" not in rendered


def test_model_identity_equality_includes_every_residency_component() -> None:
    baseline = _identity()

    changed = (
        _identity(provider_id="transcribe-cpp"),
        _identity(model_id="local-gguf:whisper"),
        _identity(root_revision="revision-b"),
        _identity(closure_fingerprint="fingerprint-b"),
        _identity(precision="f32"),
        _identity(device=ExecutionDevice.METAL),
        _identity(local_snapshot_token="replacement-snapshot"),
    )

    assert all(candidate != baseline for candidate in changed)


def test_protocol_repr_redacts_private_paths_options_and_snapshot_tokens() -> None:
    request = _request()
    failure = ExecutorFailure(
        generation=3,
        attempt_id="attempt-1",
        code=TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
        recovery_actions=("choose_another_gguf", "retry_faster_whisper"),
        failed_attempt={"private": "/private/models/secret.gguf"},
    )

    snapshot = LocalSourceSnapshot(
        token="private-snapshot-token",
        paths=(Path("/private/models/parakeet/encoder-model.int8.onnx"),),
        identities=((7, 11, 1024, 123456),),
    )
    rendered = repr(request) + repr(request.identity) + repr(snapshot)
    assert "/private/" not in rendered
    assert "private-snapshot-token" not in rendered
    assert "/private/" not in repr(failure)


def test_managed_artifact_reference_is_validated() -> None:
    values = {
        "generation": 3,
        "attempt_id": "attempt-1",
        "job_id": "job-1",
        "source": FileAudioSource(Path("media.wav")),
        "identity": _identity(),
        "options": {},
        "managed_store_root": Path("store"),
    }

    with pytest.raises(ValueError, match="managed_artifact_ref"):
        ExecutorRequest(**values, managed_artifact_ref=("parakeet-v2", "", "int8"))


def test_external_request_accepts_exact_managed_dependencies() -> None:
    snapshot = LocalSourceSnapshot(
        token="private-snapshot-token",
        paths=(Path("/private/models/encoder.onnx"),),
        identities=((7, 11, 1024, 123456),),
    )

    request = ExecutorRequest(
        generation=3,
        attempt_id="external-parakeet",
        job_id="job-external-parakeet",
        source=FileAudioSource(Path("speech.wav")),
        identity=_identity(closure_fingerprint=None),
        options={},
        local_source=snapshot,
        managed_store_root=Path("store"),
        managed_dependency_refs=(("silero-vad", "vad-revision", "f32"),),
    )

    assert request.managed_artifact_ref is None
    assert request.managed_dependency_refs == (("silero-vad", "vad-revision", "f32"),)


def test_external_request_rejects_managed_root_and_dependency_without_store() -> None:
    snapshot = LocalSourceSnapshot(
        token="private-snapshot-token",
        paths=(Path("/private/models/encoder.onnx"),),
        identities=((7, 11, 1024, 123456),),
    )
    values = {
        "generation": 3,
        "attempt_id": "external-parakeet",
        "job_id": "job-external-parakeet",
        "source": FileAudioSource(Path("speech.wav")),
        "identity": _identity(closure_fingerprint=None),
        "options": {},
        "local_source": snapshot,
    }

    with pytest.raises(ValueError, match="mutually exclusive"):
        ExecutorRequest(
            **values,
            managed_store_root=Path("store"),
            managed_artifact_ref=("parakeet-v2", "root-revision", "int8"),
        )
    with pytest.raises(ValueError, match="managed_store_root"):
        ExecutorRequest(
            **values,
            managed_dependency_refs=(("silero-vad", "vad-revision", "f32"),),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        ExecutorRequest(
            **{**values, "local_source": None},
            managed_store_root=Path("store"),
            managed_artifact_ref=("parakeet-v2", "root-revision", "int8"),
            managed_dependency_refs=(("silero-vad", "vad-revision", "f32"),),
        )


def test_cpu_retry_preserves_managed_dependency_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Connection:
        def __init__(self) -> None:
            self.sent: list[ExecutorRequest] = []

        def send(self, request: ExecutorRequest) -> None:
            self.sent.append(request)

    class _Cancellation:
        def clear(self) -> None:
            return None

    dependency_refs = (("silero-vad", "vad-revision", "f32"),)
    request = ExecutorRequest(
        generation=1,
        attempt_id="cpu-retry-dependencies",
        job_id="job-cpu-retry-dependencies",
        source=FileAudioSource(Path("speech.wav")),
        identity=_identity(
            device=ExecutionDevice.METAL,
            closure_fingerprint=None,
            local_snapshot_token=None,
        ),
        options={},
        managed_store_root=Path("store"),
        managed_dependency_refs=dependency_refs,
    )
    callbacks = _Callbacks()
    executor = LocalSTTExecutor()
    connection = _Connection()
    executor._active_request = request
    executor._active_callbacks = callbacks  # type: ignore[assignment]
    executor._busy = True
    monkeypatch.setattr(executor, "_retire_idle_worker_locked", lambda: True)

    def start_worker() -> None:
        executor._worker_generation = 2
        executor._connection = connection  # type: ignore[assignment]
        executor._cancellation_event = _Cancellation()

    monkeypatch.setattr(executor, "_start_worker_locked", start_worker)

    executor._retry_on_cpu(request, callbacks)  # type: ignore[arg-type]

    assert connection.sent[0].managed_dependency_refs == dependency_refs


def test_worker_phase_is_restricted_to_worker_owned_transitions() -> None:
    assert {phase.value for phase in WorkerPhase} == {
        "preparing",
        "loading",
        "transcribing",
        "post-processing",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"generation": 0},
        {"attempt_id": ""},
    ],
)
def test_protocol_rejects_empty_or_invalid_required_identity(
    kwargs: dict[str, object],
) -> None:
    values = {
        "generation": 3,
        "attempt_id": "attempt-1",
        "job_id": "job-1",
        "source": FileAudioSource(Path("media.wav")),
        "identity": _identity(),
        "options": {},
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        ExecutorRequest(**values)


def test_model_identity_rejects_empty_provider() -> None:
    with pytest.raises(ValueError, match="provider_id"):
        _identity(provider_id="")


def test_failure_requires_stable_typed_code_and_bounded_actions() -> None:
    with pytest.raises(TypeError):
        ExecutorFailure(  # type: ignore[arg-type]
            3,
            "attempt-1",
            "engine_crashed",
        )


@pytest.mark.parametrize(
    ("provider_id", "error", "expected_actions"),
    [
        (
            "parakeet-onnx",
            LocalSourceChangedError("private path"),
            ("retry_faster_whisper",),
        ),
        (
            "transcribe-cpp",
            LocalSourceChangedError("private path"),
            ("choose_another_gguf", "retry_faster_whisper"),
        ),
        (
            "parakeet-onnx",
            _ProviderLoadFailure(TranscriptionFailureCode.PROVIDER_UNAVAILABLE),
            ("retry_faster_whisper",),
        ),
    ],
)
def test_worker_generated_failures_always_offer_provider_recovery(
    provider_id: str,
    error: BaseException,
    expected_actions: tuple[str, ...],
) -> None:
    request = _request()
    request = ExecutorRequest(
        generation=request.generation,
        attempt_id=request.attempt_id,
        job_id=request.job_id,
        source=request.source,
        identity=_identity(provider_id=provider_id, local_snapshot_token=None),
        options={},
    )

    failure = _failure_from_exception(request, error)

    assert failure.recovery_actions == expected_actions
    with pytest.raises(ValueError):
        ExecutorFailure(
            3,
            "attempt-1",
            TranscriptionFailureCode.ENGINE_CRASHED,
            recovery_actions=tuple(f"action-{index}" for index in range(9)),
        )


def test_terminal_guard_accepts_exactly_one_matching_terminal_envelope() -> None:
    guard = _AttemptTerminalGuard(generation=3, attempt_id="attempt-1")
    matching = ExecutorResult(3, "attempt-1", {"content": "hello"})

    assert guard.accept(matching) is True
    assert guard.accept(matching) is False
    assert guard.accept(ExecutorResult(2, "attempt-1", {})) is False
    assert guard.accept(ExecutorResult(3, "attempt-2", {})) is False


def test_terminal_guard_does_not_consume_slot_for_stale_envelope() -> None:
    guard = _AttemptTerminalGuard(generation=3, attempt_id="attempt-1")

    assert guard.accept(ExecutorResult(2, "attempt-1", {})) is False
    assert guard.accept(ExecutorResult(3, "attempt-1", {})) is True


class _Callbacks:
    def __init__(self) -> None:
        self.events: list[ExecutorEvent] = []
        self.results: list[ExecutorResult] = []
        self.failures: list[ExecutorFailure] = []
        self.terminal = threading.Event()

    def on_event(self, event: ExecutorEvent) -> None:
        self.events.append(event)

    def on_result(self, result: ExecutorResult) -> None:
        self.results.append(result)
        self.terminal.set()

    def on_failure(self, failure: ExecutorFailure) -> None:
        self.failures.append(failure)
        self.terminal.set()


def _executor(*, completed_job_limit: int = 20) -> LocalSTTExecutor:
    return LocalSTTExecutor(
        worker_target=fake_executor_worker,
        completed_job_limit=completed_job_limit,
        startup_timeout=5.0,
        graceful_shutdown_timeout=0.2,
        force_stop_timeout=2.0,
    )


def _protocol_executor() -> LocalSTTExecutor:
    return LocalSTTExecutor(
        worker_target=protocol_executor_worker,
        startup_timeout=5.0,
        graceful_shutdown_timeout=0.2,
        force_stop_timeout=2.0,
    )


def _submit(
    executor: LocalSTTExecutor,
    callbacks: _Callbacks,
    *,
    attempt_id: str,
    identity: ModelIdentity | None = None,
    mode: str = "succeed",
    explicit_retry: bool = False,
) -> int:
    return executor.submit(
        attempt_id=attempt_id,
        job_id=f"job-{attempt_id}",
        source=FileAudioSource(Path("fixture.wav")),
        identity=identity or _identity(root_revision=None, closure_fingerprint=None),
        options={"test_mode": mode},
        on_event=callbacks.on_event,
        on_result=callbacks.on_result,
        on_failure=callbacks.on_failure,
        explicit_retry=explicit_retry,
    )


def _wait_for_terminal(callbacks: _Callbacks) -> None:
    assert callbacks.terminal.wait(10.0)


def _wait_until(predicate: object, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:  # type: ignore[operator]
        time.sleep(0.01)
    assert predicate()  # type: ignore[operator]


@pytest.mark.integration
def test_executor_starts_under_textual_filenoless_stderr() -> None:
    """The first real spawn must survive Textual's ``fileno() == -1`` stderr."""

    repo_root = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "PYTHONPATH": str(repo_root)}
    environment.pop("PYTEST_CURRENT_TEST", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                import threading


                class TextualLikeStderr:
                    def fileno(self):
                        return -1

                    def write(self, *_args, **_kwargs):
                        return 0

                    def flush(self):
                        pass


                if __name__ == "__main__":
                    sys.stderr = TextualLikeStderr()

                    from Tests.STT.executor_test_support import protocol_executor_worker
                    from tldw_chatbook.STT.contracts import (
                        BufferAudioSource,
                        ExecutionDevice,
                    )
                    from tldw_chatbook.STT.executor import LocalSTTExecutor, ModelIdentity

                    terminal = threading.Event()
                    results = []
                    failures = []

                    def on_result(value):
                        results.append(value)
                        terminal.set()

                    def on_failure(value):
                        failures.append(value)
                        terminal.set()

                    executor = LocalSTTExecutor(
                        worker_target=protocol_executor_worker,
                        startup_timeout=5.0,
                        graceful_shutdown_timeout=0.2,
                        force_stop_timeout=2.0,
                    )
                    try:
                        executor.submit(
                            attempt_id="textual-stderr",
                            job_id=None,
                            source=BufferAudioSource(b"\\x00\\x00", 16_000),
                            identity=ModelIdentity(
                                provider_id="parakeet-onnx",
                                model_id="nemo-parakeet-tdt-0.6b-v2",
                                root_revision=None,
                                closure_fingerprint=None,
                                precision="int8",
                                device=ExecutionDevice.CPU,
                            ),
                            options={},
                            segment_end_frames=(1,),
                            on_result=on_result,
                            on_failure=on_failure,
                        )
                        assert terminal.wait(10.0)
                        assert len(results) == 1, failures
                    except BaseException:
                        import traceback

                        traceback.print_exc(file=sys.stdout)
                        raise
                    finally:
                        executor.close()
                    print("STT_WORKER_OK")
                """
            ),
        ],
        cwd=repo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "STT_WORKER_OK" in completed.stdout


def test_worker_file_source_uses_existing_parser_with_path_options_and_runner() -> None:
    executor = _protocol_executor()
    callbacks = _Callbacks()
    source = FileAudioSource(Path("/private/audio/file.wav"))
    options = {"transcription_provider": "parakeet-onnx", "timestamps": False}
    try:
        executor.submit(
            attempt_id="file-attempt",
            job_id="file-job",
            source=source,
            identity=_identity(root_revision=None, closure_fingerprint=None),
            options=options,
            on_result=callbacks.on_result,
            on_failure=callbacks.on_failure,
        )
        _wait_for_terminal(callbacks)
    finally:
        executor.close()

    assert callbacks.failures == []
    assert callbacks.results[0].payload == {
        "file_path": "/private/audio/file.wav",
        "options": options,
        "runner_payload": {
            "audio_path": "/private/audio/file.wav",
            "kwargs": {"provider": "parakeet-onnx"},
        },
    }


def test_worker_buffer_source_bypasses_file_parser_and_returns_buffer_payload() -> None:
    executor = _protocol_executor()
    callbacks = _Callbacks()
    source = BufferAudioSource(b"\x00\x00\x01\x00", 16_000)
    try:
        executor.submit(
            attempt_id="buffer-attempt",
            job_id=None,
            source=source,
            identity=_identity(root_revision=None, closure_fingerprint=None),
            options={},
            segment_end_frames=(2,),
            on_result=callbacks.on_result,
            on_failure=callbacks.on_failure,
        )
        _wait_for_terminal(callbacks)
    finally:
        executor.close()

    assert callbacks.failures == []
    assert callbacks.results[0].payload == {
        "audio_bytes": 4,
        "sample_rate": 16_000,
        "segment_end_frames": (2,),
    }


@pytest.mark.parametrize(
    ("identity", "options"),
    [
        (
            _identity(
                provider_id="transcribe-cpp",
                model_id="local-gguf:whisper",
                precision="native",
                root_revision=None,
                closure_fingerprint=None,
            ),
            {},
        ),
        (
            _identity(root_revision=None, closure_fingerprint=None),
            {"test_no_buffer_runner": True},
        ),
    ],
)
def test_worker_rejects_buffer_without_parakeet_buffer_capability(
    identity: ModelIdentity,
    options: dict[str, object],
) -> None:
    executor = _protocol_executor()
    callbacks = _Callbacks()
    try:
        executor.submit(
            attempt_id="unsupported-buffer",
            job_id=None,
            source=BufferAudioSource(b"\x00\x00", 16_000),
            identity=identity,
            options=options,
            segment_end_frames=(1,),
            on_result=callbacks.on_result,
            on_failure=callbacks.on_failure,
        )
        _wait_for_terminal(callbacks)
    finally:
        executor.close()

    assert callbacks.results == []
    assert callbacks.failures[0].code is TranscriptionFailureCode.UNSUPPORTED_CAPABILITY
    assert callbacks.failures[0].recovery_actions == ("retry_faster_whisper",)


def test_worker_passes_each_buffer_requests_current_provenance_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = (
        ExecutorRequest(
            generation=1,
            attempt_id="attempt-one",
            job_id="job-one",
            source=BufferAudioSource(b"\x00\x00", 16_000),
            identity=_identity(root_revision=None, closure_fingerprint=None),
            options={
                "language": "en",
                "transcription_context": {"batch_id": "batch-one"},
            },
            segment_end_frames=(1,),
        ),
        ExecutorRequest(
            generation=1,
            attempt_id="attempt-two",
            job_id=None,
            source=BufferAudioSource(b"\x01\x00", 16_000),
            identity=_identity(root_revision=None, closure_fingerprint=None),
            options={
                "language": "fr",
                "transcription_context": {"batch_id": "batch-two"},
            },
            segment_end_frames=(1,),
        ),
    )
    received: list[dict[str, object]] = []

    def provider_builder(*_args, **_kwargs) -> ProviderRuntime:
        def buffer_runner(source, **kwargs):
            received.append({"audio": source.audio, **kwargs})
            return {
                "attempt_id": kwargs["attempt_id"],
                "job_id": kwargs["job_id"],
                "language": kwargs["language"],
            }

        return ProviderRuntime(
            runner=lambda *_args, **_kwargs: {},
            buffer_runner=buffer_runner,
            close=lambda: None,
        )

    class _Connection:
        def __init__(self) -> None:
            self.commands = iter((*requests, ("close", 1)))
            self.sent: list[object] = []

        def send(self, value: object) -> None:
            self.sent.append(value)

        def recv(self) -> object:
            return next(self.commands)

        def close(self) -> None:
            return None

    class _Event:
        def wait(self, _timeout: float) -> bool:
            return True

        def is_set(self) -> bool:
            return False

    connection = _Connection()
    monkeypatch.setattr(
        "tldw_chatbook.STT.executor_worker.enter_worker_containment",
        lambda: SimpleNamespace(pid=1),
    )

    _run_executor_worker(
        connection,  # type: ignore[arg-type]
        _Event(),
        _Event(),
        1,
        str(tmp_path),
        provider_builder=provider_builder,
        parse_job=lambda *_args, **_kwargs: {},
    )

    assert received == [
        {
            "audio": b"\x00\x00",
            "segment_end_frames": (1,),
            "attempt_id": "attempt-one",
            "job_id": "job-one",
            "language": "en",
            "transcription_context": {"batch_id": "batch-one"},
        },
        {
            "audio": b"\x01\x00",
            "segment_end_frames": (1,),
            "attempt_id": "attempt-two",
            "job_id": None,
            "language": "fr",
            "transcription_context": {"batch_id": "batch-two"},
        },
    ]
    results = [item for item in connection.sent if type(item) is ExecutorResult]
    assert [item.payload for item in results] == [
        {"attempt_id": "attempt-one", "job_id": "job-one", "language": "en"},
        {"attempt_id": "attempt-two", "job_id": None, "language": "fr"},
    ]


def test_parakeet_buffer_runner_serializes_normalized_result_without_synthetic_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.STT.parakeet_onnx import (
        ParakeetBufferResult,
        ParakeetOnnxRuntime,
    )

    captured: dict[str, object] = {}

    class _Runtime:
        def transcribe_buffer(self, **kwargs):
            captured.update(kwargs)
            normalized = TranscriptionResult(
                text="ordinary text console stop",
                segments=(),
                provenance=TranscriptionProvenance(
                    schema_version=1,
                    attempt_id=kwargs["attempt_id"],
                    batch_id=None,
                    job_id=kwargs["job_id"],
                    retry_of_attempt_id=None,
                    retry_of_job_id=None,
                    provider_id="parakeet-onnx",
                    model_id=PARAKEET_V2_MODEL,
                    artifact_root=None,
                    artifact_dependencies=(),
                    precision="int8",
                    requested_device=ExecutionDevice.CPU,
                    effective_device=ExecutionDevice.CPU,
                    requested_language=kwargs["language"],
                    effective_language="en",
                    detected_language=None,
                    task=TranscriptionTask.TRANSCRIBE,
                ),
                produced_capabilities=ProducedCapabilities(
                    timestamps=TimestampGranularity.NONE,
                    punctuation=True,
                    capitalization=True,
                    vad=False,
                    diarization=False,
                ),
                duration_seconds=0.5,
                timings=TranscriptionTimings(total_seconds=0.1),
            )
            return ParakeetBufferResult(
                normalized=normalized,
                logical_segments=("ordinary text", "console stop"),
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(ParakeetOnnxRuntime, "load", lambda **_kwargs: _Runtime())
    request = ExecutorRequest(
        generation=1,
        attempt_id="first-load-attempt",
        job_id="first-load-job",
        source=BufferAudioSource(b"\x00\x00\x01\x00", 16_000),
        identity=_identity(
            root_revision=None,
            closure_fingerprint=None,
            local_snapshot_token=None,
        ),
        options={
            "language": "en",
            "transcription_context": {"batch_id": "first-load-batch"},
        },
        segment_end_frames=(2,),
    )
    provider = _parakeet_provider(request, tmp_path, None, lambda: False)

    payload = provider.buffer_runner(
        request.source,
        segment_end_frames=(1, 2),
        attempt_id="current-attempt",
        job_id=None,
        language="fr",
        transcription_context={"batch_id": "current-batch"},
    )

    assert payload == {
        "text": "ordinary text console stop",
        "logical_segments": ("ordinary text", "console stop"),
        "duration": 0.5,
        "transcription_model": PARAKEET_V2_MODEL,
        "transcription_provenance": {
            "schema_version": 1,
            "attempt_id": "current-attempt",
            "batch_id": "current-batch",
            "job_id": None,
            "retry_of_attempt_id": None,
            "retry_of_job_id": None,
            "provider_id": "parakeet-onnx",
            "model_id": PARAKEET_V2_MODEL,
            "artifact_root": None,
            "artifact_dependencies": [],
            "precision": "int8",
            "requested_device": "cpu",
            "effective_device": "cpu",
            "requested_language": "fr",
            "effective_language": "en",
            "detected_language": None,
            "task": "transcribe",
            "produced_capabilities": {
                "timestamps": "none",
                "punctuation": True,
                "capitalization": True,
                "vad": False,
                "diarization": False,
            },
            "warnings": [],
            "failed_attempt": None,
        },
    }
    assert captured["attempt_id"] == "current-attempt"
    assert captured["job_id"] is None
    assert captured["language"] == "fr"


def test_parakeet_buffer_runner_rejects_non_int16_pcm_as_unsupported_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxRuntime

    class _Runtime:
        def transcribe_buffer(self, **_kwargs):
            raise AssertionError("unsupported PCM must not reach native inference")

        def close(self) -> None:
            return None

    monkeypatch.setattr(ParakeetOnnxRuntime, "load", lambda **_kwargs: _Runtime())
    request = ExecutorRequest(
        generation=1,
        attempt_id="unsupported-width",
        job_id=None,
        source=BufferAudioSource(b"\x00", 16_000, sample_width=1),
        identity=_identity(
            root_revision=None,
            closure_fingerprint=None,
            local_snapshot_token=None,
        ),
        options={"language": "en"},
        segment_end_frames=(1,),
    )
    provider = _parakeet_provider(request, tmp_path, None, lambda: False)

    with pytest.raises(_ProviderLoadFailure) as raised:
        provider.buffer_runner(
            request.source,
            segment_end_frames=(1,),
            attempt_id=request.attempt_id,
            job_id=None,
            language="en",
            transcription_context={},
        )

    assert raised.value.code is TranscriptionFailureCode.UNSUPPORTED_CAPABILITY


def test_parakeet_buffer_failure_uses_current_request_metadata_without_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.STT.parakeet_onnx import (
        ParakeetOnnxFailure,
        ParakeetOnnxRuntime,
    )

    class _Runtime:
        def transcribe_buffer(self, **_kwargs):
            raise RuntimeError("private native inference detail")

        def close(self) -> None:
            return None

    monkeypatch.setattr(ParakeetOnnxRuntime, "load", lambda **_kwargs: _Runtime())
    request = ExecutorRequest(
        generation=1,
        attempt_id="first-attempt",
        job_id="first-job",
        source=BufferAudioSource(b"\x00\x00", 16_000),
        identity=_identity(
            model_id=PARAKEET_V3_MODEL,
            root_revision=None,
            closure_fingerprint=None,
            local_snapshot_token=None,
        ),
        options={
            "language": "en",
            "transcription_context": {"batch_id": "first-batch"},
        },
        segment_end_frames=(1,),
    )
    provider = _parakeet_provider(request, tmp_path, None, lambda: False)

    try:
        provider.buffer_runner(
            request.source,
            segment_end_frames=(1,),
            attempt_id="current-attempt",
            job_id=None,
            language="fr",
            transcription_context={},
        )
    except ParakeetOnnxFailure as error:
        failure = _failure_from_exception(request, error)
    else:
        raise AssertionError("buffer inference failure was not raised")

    assert failure.code is TranscriptionFailureCode.INFERENCE_FAILED
    assert failure.recovery_actions == ("retry_faster_whisper",)
    assert failure.failed_attempt is not None
    assert failure.failed_attempt["attempt_id"] == "current-attempt"
    assert failure.failed_attempt["batch_id"] is None
    assert failure.failed_attempt["job_id"] is None
    assert failure.failed_attempt["requested_language"] == "fr"
    assert failure.failed_attempt["effective_language"] == "auto"
    assert "private" not in str(failure)


def test_controller_starts_lazily_and_reuses_same_worker_for_same_identity() -> None:
    executor = _executor()
    first = _Callbacks()
    second = _Callbacks()
    try:
        assert executor.generation == 0
        first_generation = _submit(executor, first, attempt_id="one")
        _wait_for_terminal(first)
        second_generation = _submit(executor, second, attempt_id="two")
        _wait_for_terminal(second)

        assert first_generation == second_generation
        assert (
            first.results[0].payload["worker_pid"]
            == second.results[0].payload["worker_pid"]
        )
        assert executor.resident_identity == _identity(
            root_revision=None,
            closure_fingerprint=None,
        )
    finally:
        executor.close()


def test_controller_recycles_idle_worker_when_identity_changes() -> None:
    executor = _executor()
    first = _Callbacks()
    second = _Callbacks()
    try:
        first_generation = _submit(executor, first, attempt_id="one")
        _wait_for_terminal(first)
        changed = _identity(
            model_id="local-gguf:whisper",
            root_revision=None,
            closure_fingerprint=None,
        )
        second_generation = _submit(
            executor,
            second,
            attempt_id="two",
            identity=changed,
        )
        _wait_for_terminal(second)

        assert second_generation > first_generation
        assert (
            first.results[0].payload["worker_pid"]
            != second.results[0].payload["worker_pid"]
        )
    finally:
        executor.close()


def test_controller_recycles_after_completed_job_bound() -> None:
    executor = _executor(completed_job_limit=1)
    first = _Callbacks()
    second = _Callbacks()
    try:
        first_generation = _submit(executor, first, attempt_id="one")
        _wait_for_terminal(first)
        second_generation = _submit(executor, second, attempt_id="two")
        _wait_for_terminal(second)

        assert second_generation > first_generation
    finally:
        executor.close()


def test_reader_does_not_reap_worker_after_controller_detaches_it() -> None:
    executor = _executor()

    class JoinProbe:
        joined = False

        def join(self, timeout: float | None = None) -> None:
            self.joined = True

    stale_process = JoinProbe()

    executor._handle_worker_exit(1, stale_process)

    assert stale_process.joined is False


def test_controller_has_one_active_request_and_cooperative_cancel_is_attempt_scoped() -> (
    None
):
    executor = _executor()
    held = _Callbacks()
    successor = _Callbacks()
    try:
        _submit(executor, held, attempt_id="held", mode="hold")
        _wait_until(
            lambda: any(
                event.phase is WorkerPhase.TRANSCRIBING for event in held.events
            )
        )
        with pytest.raises(ExecutorBusyError):
            _submit(executor, successor, attempt_id="blocked")
        assert executor.cancel("wrong-attempt") is False
        assert executor.cancel("held") is True
        _wait_for_terminal(held)
        assert held.failures[0].code is TranscriptionFailureCode.CANCELLED

        _submit(executor, successor, attempt_id="successor")
        _wait_for_terminal(successor)
        assert successor.results
    finally:
        executor.close()


@pytest.mark.parametrize("mode", ["stale_then_succeed", "duplicate"])
def test_controller_drops_stale_and_duplicate_terminals(mode: str) -> None:
    executor = _executor()
    callbacks = _Callbacks()
    try:
        _submit(executor, callbacks, attempt_id="one", mode=mode)
        _wait_for_terminal(callbacks)
        time.sleep(0.05)

        assert len(callbacks.results) == 1
        assert callbacks.results[0].payload["content"] == "transcript"
        assert callbacks.failures == []
    finally:
        executor.close()


def test_force_stop_detaches_before_kill_and_cleans_generation_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _executor()
    callbacks = _Callbacks()
    termination_entered = threading.Event()
    allow_termination = threading.Event()
    try:
        _submit(executor, callbacks, attempt_id="held", mode="ignore_cancel")
        _wait_until(
            lambda: any(
                event.phase is WorkerPhase.TRANSCRIBING for event in callbacks.events
            )
        )
        scratch = executor._scratch_path
        assert scratch is not None and scratch.is_dir()
        tree = executor._tree
        assert tree is not None
        original_terminate_tree = tree.terminate_tree

        def gated_terminate_tree(**kwargs: float) -> bool:
            termination_entered.set()
            assert allow_termination.wait(10.0)
            return original_terminate_tree(**kwargs)

        monkeypatch.setattr(tree, "terminate_tree", gated_terminate_tree)

        assert executor.force_stop("held") is True
        assert termination_entered.wait(10.0)
        assert scratch.exists() is True
        allow_termination.set()
        _wait_for_terminal(callbacks)
        assert callbacks.failures[0].code is TranscriptionFailureCode.CANCELLED
        assert executor.wait_for_retirement(10.0) is True
        assert executor.retiring is False
        assert scratch.exists() is False
        assert executor.busy is False
    finally:
        allow_termination.set()
        executor.wait_for_retirement(10.0)
        executor.close()


def test_failed_force_stop_quarantines_executor_and_prevents_second_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _executor()
    callbacks = _Callbacks()
    try:
        _submit(executor, callbacks, attempt_id="held", mode="ignore_cancel")
        _wait_until(
            lambda: any(
                event.phase is WorkerPhase.TRANSCRIBING for event in callbacks.events
            )
        )
        tree = executor._tree
        assert tree is not None
        original = tree.terminate_tree
        monkeypatch.setattr(tree, "terminate_tree", lambda **_kwargs: False)

        assert executor.force_stop("held") is True
        _wait_until(lambda: executor.retiring is False)
        assert executor.unavailable is True
        with pytest.raises(ExecutorUnavailableError):
            _submit(executor, _Callbacks(), attempt_id="blocked")

        monkeypatch.setattr(tree, "terminate_tree", original)
        assert original(term_timeout=2.0, kill_timeout=2.0) is True
    finally:
        executor.close()


def test_loading_crash_marks_only_active_identity_unhealthy() -> None:
    executor = _executor()
    callbacks = _Callbacks()
    identity = _identity(root_revision=None, closure_fingerprint=None)
    try:
        _submit(
            executor,
            callbacks,
            attempt_id="crash",
            identity=identity,
            mode="crash_loading",
        )
        _wait_for_terminal(callbacks)

        assert callbacks.failures[0].code is TranscriptionFailureCode.ENGINE_CRASHED
        assert callbacks.failures[0].recovery_actions == ("retry_faster_whisper",)
        assert executor.unhealthy_identity == identity
        with pytest.raises(ExecutorUnavailableError):
            _submit(executor, _Callbacks(), attempt_id="blocked", identity=identity)
        assert executor.clear_unhealthy_identity(identity) is True
    finally:
        executor.close()


def test_typed_device_failure_retries_once_on_cpu_in_fresh_generation() -> None:
    executor = _executor()
    callbacks = _Callbacks()
    accelerated = _identity(
        root_revision=None,
        closure_fingerprint=None,
        device=ExecutionDevice.METAL,
    )
    try:
        first_generation = _submit(
            executor,
            callbacks,
            attempt_id="retry",
            identity=accelerated,
            mode="device_failure",
        )
        _wait_for_terminal(callbacks)

        assert callbacks.failures == []
        assert callbacks.results[0].generation > first_generation
        assert callbacks.results[0].attempt_id == "retry"
        assert callbacks.results[0].payload["device"] == "cpu"
        assert callbacks.results[0].payload["cpu_fallback_requested_device"] == "metal"
    finally:
        executor.close()


def test_real_worker_typed_provider_failure_reloads_on_effective_cpu() -> None:
    executor = LocalSTTExecutor(
        worker_target=device_retry_executor_worker,
        startup_timeout=5.0,
        graceful_shutdown_timeout=0.2,
        force_stop_timeout=2.0,
    )
    callbacks = _Callbacks()
    accelerated = _identity(
        root_revision=None,
        closure_fingerprint=None,
        device=ExecutionDevice.METAL,
    )
    try:
        first_generation = _submit(
            executor,
            callbacks,
            attempt_id="real-worker-retry",
            identity=accelerated,
        )
        _wait_for_terminal(callbacks)

        assert callbacks.failures == []
        assert callbacks.results[0].generation > first_generation
        assert executor.resident_identity == _identity(
            root_revision=None,
            closure_fingerprint=None,
            device=ExecutionDevice.CPU,
        )
        assert callbacks.results[0].payload["cpu_fallback_requested_device"] == "metal"
    finally:
        executor.close()


def test_real_worker_suppresses_path_bearing_legacy_logs(
    capfd: pytest.CaptureFixture[str],
) -> None:
    executor = LocalSTTExecutor(
        worker_target=private_log_executor_worker,
        startup_timeout=5.0,
        graceful_shutdown_timeout=0.2,
        force_stop_timeout=2.0,
    )
    callbacks = _Callbacks()
    try:
        _submit(executor, callbacks, attempt_id="private-log")
        _wait_for_terminal(callbacks)
        assert callbacks.failures[0].code is TranscriptionFailureCode.INFERENCE_FAILED
    finally:
        executor.close()

    captured = capfd.readouterr()
    assert "/private/models/secret.onnx" not in captured.out
    assert "/private/models/secret.onnx" not in captured.err


def test_cpu_retry_start_failure_delivers_one_terminal_instead_of_stranding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _executor()
    callbacks = _Callbacks()
    accelerated = _identity(
        root_revision=None,
        closure_fingerprint=None,
        device=ExecutionDevice.METAL,
    )
    retry_gate = threading.Event()
    retry_reached = threading.Event()
    original_on_event = callbacks.on_event

    def hold_retry(event: ExecutorEvent) -> None:
        original_on_event(event)
        if event.phase is WorkerPhase.TRANSCRIBING:
            retry_reached.set()
            assert retry_gate.wait(10.0)

    try:
        executor.submit(
            attempt_id="retry-start-fails",
            job_id="job-retry-start-fails",
            source=FileAudioSource(Path("fixture.wav")),
            identity=accelerated,
            options={"test_mode": "device_failure"},
            on_event=hold_retry,
            on_result=callbacks.on_result,
            on_failure=callbacks.on_failure,
        )
        assert retry_reached.wait(10.0)
        monkeypatch.setattr(
            executor,
            "_start_worker_locked",
            lambda: (_ for _ in ()).throw(
                ExecutorUnavailableError("replacement unavailable")
            ),
        )
        retry_gate.set()
        _wait_for_terminal(callbacks)

        assert callbacks.results == []
        assert len(callbacks.failures) == 1
        assert callbacks.failures[0].code is TranscriptionFailureCode.ENGINE_CRASHED
        assert callbacks.failures[0].recovery_actions == ("retry_faster_whisper",)
        assert executor.busy is False
    finally:
        retry_gate.set()
        executor.close()


def test_close_is_idempotent_and_removes_idle_generation_scratch() -> None:
    executor = _executor()
    callbacks = _Callbacks()
    _submit(executor, callbacks, attempt_id="one")
    _wait_for_terminal(callbacks)
    scratch = executor._scratch_path
    assert scratch is not None and scratch.exists()

    executor.close()
    executor.close()

    assert scratch.exists() is False
    assert executor.busy is False


def test_local_source_snapshot_is_path_private_and_detects_replacement(
    tmp_path: Path,
) -> None:
    encoder = tmp_path / "private-encoder.onnx"
    decoder = tmp_path / "private-decoder.onnx"
    encoder.write_bytes(b"encoder-a")
    decoder.write_bytes(b"decoder-a")
    snapshot = snapshot_local_source((encoder, decoder))

    assert len(snapshot.token) == 64
    assert str(tmp_path) not in repr(snapshot)
    validate_local_source_snapshot(snapshot)

    encoder.write_bytes(b"encoder-replaced-with-different-bytes")
    with pytest.raises(LocalSourceChangedError) as raised:
        validate_local_source_snapshot(snapshot)
    assert str(tmp_path) not in str(raised.value)
    assert str(tmp_path) not in repr(raised.value)


def test_local_source_snapshot_rejects_symlink_without_path_leak(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.gguf"
    link = tmp_path / "private-link.gguf"
    target.write_bytes(b"model")
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(LocalSourceChangedError) as raised:
        snapshot_local_source((link,))

    assert str(link) not in str(raised.value)
    assert str(link) not in repr(raised.value)


def _resident_executor() -> LocalSTTExecutor:
    return LocalSTTExecutor(
        worker_target=resident_executor_worker,
        startup_timeout=5.0,
        graceful_shutdown_timeout=0.2,
        force_stop_timeout=2.0,
    )


def _managed_request_values(
    tmp_path: Path,
) -> tuple[object, object, object, ModelIdentity]:
    service, root, dependency = installed_root_and_dependency(tmp_path)
    service.activate(root.reference)
    leased = service.acquire(root.reference)
    fingerprint = leased.handle.closure_fingerprint
    leased.close()
    identity = _identity(
        root_revision=root.reference.revision,
        closure_fingerprint=fingerprint,
        local_snapshot_token=None,
    )
    return service, root, dependency, identity


def _managed_reference_tuple(reference: object) -> tuple[str, str, str]:
    return (reference.artifact_id, reference.revision, reference.variant)


def _submit_managed_runtime(
    executor: LocalSTTExecutor,
    tmp_path: Path,
    root: object,
    identity: ModelIdentity,
    callbacks: _Callbacks,
    *,
    attempt_id: str,
    hold: bool = False,
) -> int:
    options = {"transcription_provider": "parakeet-onnx"}
    if hold:
        options["test_worker_hold"] = True
    return executor.submit(
        attempt_id=attempt_id,
        job_id=f"job-{attempt_id}",
        source=FileAudioSource(tmp_path / "fixture.wav"),
        identity=identity,
        options=options,
        managed_store_root=tmp_path / "store",
        managed_artifact_ref=_managed_reference_tuple(root.reference),
        on_event=callbacks.on_event,
        on_result=callbacks.on_result,
        on_failure=callbacks.on_failure,
    )


def _external_dependency_request(
    tmp_path: Path,
    dependency: object,
    *,
    attempt_id: str,
) -> tuple[ExecutorRequest, Path]:
    model = tmp_path / "external" / "encoder.onnx"
    model.parent.mkdir(exist_ok=True)
    model.write_bytes(b"external-model")
    snapshot = snapshot_local_source((model,))
    reference = dependency.reference
    return (
        ExecutorRequest(
            generation=1,
            attempt_id=attempt_id,
            job_id=f"job-{attempt_id}",
            source=FileAudioSource(tmp_path / "speech.wav"),
            identity=_identity(
                root_revision="catalog-revision",
                closure_fingerprint=None,
                local_snapshot_token=snapshot.token,
            ),
            options={},
            local_source=snapshot,
            managed_store_root=tmp_path / "store",
            managed_dependency_refs=(
                (reference.artifact_id, reference.revision, reference.variant),
            ),
        ),
        model,
    )


def test_worker_reuses_runtime_and_holds_managed_closure_lease_until_exit(
    tmp_path: Path,
) -> None:
    service, root, dependency, identity = _managed_request_values(tmp_path)
    executor = _resident_executor()
    first = _Callbacks()
    second = _Callbacks()
    try:
        for attempt_id, callbacks in (("one", first), ("two", second)):
            executor.submit(
                attempt_id=attempt_id,
                job_id=f"job-{attempt_id}",
                source=FileAudioSource(tmp_path / "fixture.wav"),
                identity=identity,
                options={"transcription_provider": "parakeet-onnx"},
                managed_store_root=tmp_path / "store",
                managed_artifact_ref=(
                    root.reference.artifact_id,
                    root.reference.revision,
                    root.reference.variant,
                ),
                on_result=callbacks.on_result,
                on_failure=callbacks.on_failure,
            )
            _wait_for_terminal(callbacks)

        assert first.results[0].payload["runtime_load_number"] == 1
        assert second.results[0].payload["runtime_load_number"] == 1
        contender = ModelArtifactService(
            tmp_path / "store",
            lease_timeout_seconds=FAST_LEASE_TIMEOUT_SECONDS,
        )
        for reference in (root.reference, dependency.reference):
            with pytest.raises(ArtifactInUseError):
                contender.delete(reference)
    finally:
        executor.close()

    ModelArtifactService(tmp_path / "store").delete(dependency.reference)
    assert service.artifact_path(dependency.reference).exists() is False


@pytest.mark.parametrize("target_name", ("root", "dependency"))
def test_idle_resident_recycle_releases_exact_managed_lease(
    tmp_path: Path,
    target_name: str,
) -> None:
    service, root, dependency, identity = _managed_request_values(tmp_path)
    target = root.reference if target_name == "root" else dependency.reference
    executor = _resident_executor()
    callbacks = _Callbacks()
    try:
        _submit_managed_runtime(
            executor,
            tmp_path,
            root,
            identity,
            callbacks,
            attempt_id=f"idle-{target_name}",
        )
        _wait_for_terminal(callbacks)
        with pytest.raises(ArtifactInUseError):
            ModelArtifactService(
                tmp_path / "store",
                lease_timeout_seconds=FAST_LEASE_TIMEOUT_SECONDS,
            ).delete(target)

        assert (
            executor.recycle_idle_managed_reference(_managed_reference_tuple(target))
            is True
        )
        assert executor.resident_identity is None
        service.delete(target)
        assert service.artifact_path(target).exists() is False
    finally:
        executor.close()


def test_active_resident_refuses_recycle_without_cancelling_attempt(
    tmp_path: Path,
) -> None:
    _service, root, dependency, identity = _managed_request_values(tmp_path)
    executor = _resident_executor()
    callbacks = _Callbacks()
    try:
        _submit_managed_runtime(
            executor,
            tmp_path,
            root,
            identity,
            callbacks,
            attempt_id="active-recycle",
            hold=True,
        )
        _wait_until(
            lambda: any(
                event.phase is WorkerPhase.TRANSCRIBING for event in callbacks.events
            )
        )

        assert (
            executor.recycle_idle_managed_reference(
                _managed_reference_tuple(dependency.reference)
            )
            is False
        )
        assert executor.busy is True
        assert callbacks.terminal.is_set() is False
        with pytest.raises(ArtifactInUseError):
            ModelArtifactService(
                tmp_path / "store",
                lease_timeout_seconds=FAST_LEASE_TIMEOUT_SECONDS,
            ).delete(dependency.reference)
    finally:
        executor.force_stop("active-recycle")
        executor.close()


def test_nonmatching_resident_refuses_recycle_and_remains_reusable(
    tmp_path: Path,
) -> None:
    _service, root, _dependency, identity = _managed_request_values(tmp_path)
    executor = _resident_executor()
    first = _Callbacks()
    second = _Callbacks()
    try:
        first_generation = _submit_managed_runtime(
            executor,
            tmp_path,
            root,
            identity,
            first,
            attempt_id="nonmatching-first",
        )
        _wait_for_terminal(first)

        assert (
            executor.recycle_idle_managed_reference(
                ("other-model", "other-revision", "f32")
            )
            is False
        )
        second_generation = _submit_managed_runtime(
            executor,
            tmp_path,
            root,
            identity,
            second,
            attempt_id="nonmatching-second",
        )
        _wait_for_terminal(second)

        assert second_generation == first_generation
        assert second.results[0].payload["runtime_load_number"] == 1
    finally:
        executor.close()


def test_unproven_idle_recycle_cannot_report_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _service, root, _dependency, identity = _managed_request_values(tmp_path)
    executor = _resident_executor()
    callbacks = _Callbacks()
    try:
        _submit_managed_runtime(
            executor,
            tmp_path,
            root,
            identity,
            callbacks,
            attempt_id="unproven-idle",
        )
        _wait_for_terminal(callbacks)
        monkeypatch.setattr(executor, "_retire_idle_worker_locked", lambda: False)

        assert (
            executor.recycle_idle_managed_reference(
                _managed_reference_tuple(root.reference)
            )
            is False
        )
    finally:
        executor.close()


def test_provider_builder_receives_the_full_verified_managed_handle(
    tmp_path: Path,
) -> None:
    service, root, dependency, identity = _managed_request_values(tmp_path)
    request = ExecutorRequest(
        generation=1,
        attempt_id="managed-handle",
        job_id="job-managed-handle",
        source=FileAudioSource(tmp_path / "speech.wav"),
        identity=identity,
        options={},
        managed_store_root=tmp_path / "store",
        managed_artifact_ref=(
            root.reference.artifact_id,
            root.reference.revision,
            root.reference.variant,
        ),
    )
    captured = {}

    def builder(_request, model_root, handle, is_cancelled):
        from tldw_chatbook.STT.executor_worker import ProviderRuntime

        captured.update(
            model_root=model_root,
            handle=handle,
            is_cancelled=is_cancelled,
        )
        return ProviderRuntime(runner=lambda *_args, **_kwargs: {}, close=lambda: None)

    resident = _load_resident(request, builder, lambda: False)
    try:
        handle = captured["handle"]
        paths = dict(handle.paths)
        assert captured["model_root"] == paths[root.reference]
        assert paths[dependency.reference] == service.artifact_path(
            dependency.reference
        )
        assert handle.lease_keys == (
            *(reference.lease_key() for reference in handle.closure),
        )
        assert resident.managed_lease_refs == tuple(
            (
                reference.artifact_id,
                reference.revision,
                reference.variant,
            )
            for reference in handle.closure
        )
        assert captured["is_cancelled"]() is False
    finally:
        resident.close()


def test_external_runtime_holds_exact_vad_lease_across_reuse_and_close(
    tmp_path: Path,
) -> None:
    service, _root, dependency = installed_root_and_dependency(tmp_path)
    request, model = _external_dependency_request(
        tmp_path,
        dependency,
        attempt_id="external-vad",
    )
    captured: dict[str, object] = {}

    def builder(_request, model_root, handle, _is_cancelled):
        captured.update(model_root=model_root, handle=handle)
        return ProviderRuntime(runner=lambda *_args, **_kwargs: {}, close=lambda: None)

    resident = _load_resident(request, builder, lambda: False)
    try:
        handle = captured["handle"]
        assert captured["model_root"] == model.parent
        assert handle.references == (dependency.reference,)
        assert resident.managed_lease_refs == (
            (
                dependency.reference.artifact_id,
                dependency.reference.revision,
                dependency.reference.variant,
            ),
        )
        assert dict(handle.paths)[dependency.reference] == service.artifact_path(
            dependency.reference
        )
        _validate_reuse(request, resident)
        with pytest.raises(ArtifactInUseError):
            ModelArtifactService(
                tmp_path / "store",
                lease_timeout_seconds=FAST_LEASE_TIMEOUT_SECONDS,
            ).delete(dependency.reference)
    finally:
        resident.close()

    ModelArtifactService(tmp_path / "store").delete(dependency.reference)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("delete", TranscriptionFailureCode.MODEL_NOT_INSTALLED),
        ("corrupt", TranscriptionFailureCode.ARTIFACT_CORRUPT),
    ),
)
def test_external_runtime_reverifies_vad_before_resident_reuse(
    tmp_path: Path,
    mutation: str,
    expected_code: TranscriptionFailureCode,
) -> None:
    service, _root, dependency = installed_root_and_dependency(tmp_path)
    request, _model = _external_dependency_request(
        tmp_path,
        dependency,
        attempt_id="external-vad-first",
    )
    executor = _resident_executor()
    first = _Callbacks()
    second = _Callbacks()
    repaired = _Callbacks()
    payload = service.artifact_path(dependency.reference) / dependency.files[0].path
    try:
        first_generation = executor.submit(
            attempt_id="external-vad-first",
            job_id="job-external-vad-first",
            source=request.source,
            identity=request.identity,
            options={"transcription_provider": "parakeet-onnx"},
            local_source=request.local_source,
            managed_store_root=request.managed_store_root,
            managed_dependency_refs=request.managed_dependency_refs,
            on_result=first.on_result,
            on_failure=first.on_failure,
        )
        _wait_for_terminal(first)
        assert first.results

        if mutation == "delete":
            payload.unlink()
        else:
            payload.write_bytes(b"x" * dependency.files[0].size_bytes)

        executor.submit(
            attempt_id="external-vad-reuse",
            job_id="job-external-vad-reuse",
            source=request.source,
            identity=request.identity,
            options={"transcription_provider": "parakeet-onnx"},
            local_source=request.local_source,
            managed_store_root=request.managed_store_root,
            managed_dependency_refs=request.managed_dependency_refs,
            on_result=second.on_result,
            on_failure=second.on_failure,
        )
        _wait_for_terminal(second)

        assert second.results == []
        assert second.failures[0].code is expected_code
        assert str(tmp_path) not in repr(second.failures[0])
        _wait_until(lambda: executor.resident_identity is None)

        payload.write_bytes(b"dependency")
        repaired_generation = executor.submit(
            attempt_id="external-vad-repaired",
            job_id="job-external-vad-repaired",
            source=request.source,
            identity=request.identity,
            options={"transcription_provider": "parakeet-onnx"},
            local_source=request.local_source,
            managed_store_root=request.managed_store_root,
            managed_dependency_refs=request.managed_dependency_refs,
            on_result=repaired.on_result,
            on_failure=repaired.on_failure,
        )
        _wait_for_terminal(repaired)

        assert repaired_generation > first_generation
        assert repaired.results
        assert repaired.failures == []
    finally:
        executor.close()

    ModelArtifactService(tmp_path / "store").delete(dependency.reference)


def test_external_load_revalidates_model_after_acquiring_vad(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _service, _root, dependency = installed_root_and_dependency(tmp_path)
    request, model = _external_dependency_request(
        tmp_path,
        dependency,
        attempt_id="external-mutated-before-load",
    )
    original = ModelArtifactService.acquire_dependencies

    def acquire_then_mutate(service, references):
        leased = original(service, references)
        model.write_bytes(b"changed-after-vad-acquisition")
        return leased

    monkeypatch.setattr(
        ModelArtifactService,
        "acquire_dependencies",
        acquire_then_mutate,
    )

    with pytest.raises(LocalSourceChangedError):
        _load_resident(
            request,
            lambda *_args: (_ for _ in ()).throw(
                AssertionError("native load must not see changed model bytes")
            ),
            lambda: False,
        )

    ModelArtifactService(tmp_path / "store").delete(dependency.reference)


@pytest.mark.parametrize(
    ("condition", "expected"),
    (
        ("missing", TranscriptionFailureCode.MODEL_NOT_INSTALLED),
        ("corrupt", TranscriptionFailureCode.ARTIFACT_CORRUPT),
        ("contended", TranscriptionFailureCode.PROVIDER_UNAVAILABLE),
        ("wrong-role", TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE),
    ),
)
def test_external_dependency_failure_keeps_stable_worker_taxonomy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    condition: str,
    expected: TranscriptionFailureCode,
) -> None:
    import tldw_chatbook.Model_Artifacts as artifacts

    service = ModelArtifactService(tmp_path / "store")
    reference = ArtifactRef("silero-vad", f"{condition}-revision", "f32")
    role = ArtifactRole.ROOT if condition == "wrong-role" else ArtifactRole.DEPENDENCY
    dependency = single_file_descriptor(reference, role, b"dependency")
    if condition != "missing":
        install_descriptor_payload(service, tmp_path, dependency, b"dependency")
    if condition == "corrupt":
        (service.artifact_path(reference) / dependency.files[0].path).write_bytes(
            b"x" * dependency.files[0].size_bytes
        )
    lease = None
    if condition == "contended":
        lease = ArtifactOperationLease(
            service.locks_path,
            reference.lease_key(),
            LeaseMode.EXCLUSIVE,
            timeout_seconds=0.1,
        )
        lease.acquire()
        monkeypatch.setattr(
            artifacts,
            "ModelArtifactService",
            lambda root: ModelArtifactService(
                root,
                lease_timeout_seconds=FAST_LEASE_TIMEOUT_SECONDS,
            ),
        )
    request, _model = _external_dependency_request(
        tmp_path,
        dependency,
        attempt_id=f"external-{condition}",
    )

    class _Connection:
        def __init__(self) -> None:
            self.commands = iter((request, ("close", 1)))
            self.sent: list[object] = []

        def send(self, value: object) -> None:
            self.sent.append(value)

        def recv(self) -> object:
            return next(self.commands)

        def close(self) -> None:
            return None

    class _Event:
        def wait(self, _timeout: float) -> bool:
            return True

        def is_set(self) -> bool:
            return False

    connection = _Connection()
    monkeypatch.setattr(
        "tldw_chatbook.STT.executor_worker.enter_worker_containment",
        lambda: SimpleNamespace(pid=1),
    )
    try:
        _run_executor_worker(
            connection,  # type: ignore[arg-type]
            _Event(),
            _Event(),
            1,
            str(tmp_path),
            provider_builder=lambda *_args: (_ for _ in ()).throw(
                AssertionError("invalid dependency reached native load")
            ),
            parse_job=lambda *_args, **_kwargs: {},
        )
    finally:
        if lease is not None:
            lease.release()

    failures = [item for item in connection.sent if type(item) is ExecutorFailure]
    assert len(failures) == 1
    failure = failures[0]
    assert failure.code is expected
    assert str(tmp_path) not in repr(failure)


def test_dependency_reference_change_rejects_reuse_and_releases_old_lease(
    tmp_path: Path,
) -> None:
    _service, _root, dependency = installed_root_and_dependency(tmp_path)
    request, _model = _external_dependency_request(
        tmp_path,
        dependency,
        attempt_id="external-first-dependency",
    )
    resident = _load_resident(
        request,
        lambda *_args: ProviderRuntime(
            runner=lambda *_args, **_kwargs: {}, close=lambda: None
        ),
        lambda: False,
    )
    changed = ExecutorRequest(
        generation=1,
        attempt_id="external-changed-dependency",
        job_id="job-external-changed-dependency",
        source=request.source,
        identity=request.identity,
        options={},
        local_source=request.local_source,
    )
    try:
        with pytest.raises(LocalSourceChangedError):
            _validate_reuse(changed, resident)
    finally:
        resident.close()

    ModelArtifactService(tmp_path / "store").delete(dependency.reference)


def test_executor_recycles_before_dispatch_when_external_vad_reference_changes(
    tmp_path: Path,
) -> None:
    service, _root, first_dependency = installed_root_and_dependency(tmp_path)
    second_reference = ArtifactRef("silero-vad", "replacement-vad-revision", "int8")
    second_dependency = single_file_descriptor(
        second_reference,
        ArtifactRole.DEPENDENCY,
        b"replacement-dependency",
    )
    install_descriptor_payload(
        service,
        tmp_path,
        second_dependency,
        b"replacement-dependency",
    )
    request, _model = _external_dependency_request(
        tmp_path,
        first_dependency,
        attempt_id="first-vad",
    )
    executor = _resident_executor()
    first = _Callbacks()
    second = _Callbacks()

    def submit(
        attempt_id: str,
        reference: ArtifactRef,
        callbacks: _Callbacks,
    ) -> int:
        return executor.submit(
            attempt_id=attempt_id,
            job_id=f"job-{attempt_id}",
            source=request.source,
            identity=request.identity,
            options={"transcription_provider": "parakeet-onnx"},
            local_source=request.local_source,
            managed_store_root=request.managed_store_root,
            managed_dependency_refs=(
                (reference.artifact_id, reference.revision, reference.variant),
            ),
            on_result=callbacks.on_result,
            on_failure=callbacks.on_failure,
        )

    try:
        first_generation = submit("first-vad", first_dependency.reference, first)
        _wait_for_terminal(first)
        with pytest.raises(ArtifactInUseError):
            ModelArtifactService(
                tmp_path / "store",
                lease_timeout_seconds=FAST_LEASE_TIMEOUT_SECONDS,
            ).delete(first_dependency.reference)

        second_generation = submit("second-vad", second_dependency.reference, second)
        _wait_for_terminal(second)

        assert second_generation > first_generation
        assert second.results
        assert second.failures == []
        ModelArtifactService(tmp_path / "store").delete(first_dependency.reference)
        with pytest.raises(ArtifactInUseError):
            ModelArtifactService(
                tmp_path / "store",
                lease_timeout_seconds=FAST_LEASE_TIMEOUT_SECONDS,
            ).delete(second_dependency.reference)
    finally:
        executor.close()

    ModelArtifactService(tmp_path / "store").delete(second_dependency.reference)


def test_provider_exception_after_cancellation_is_reported_as_cancelled() -> None:
    request = _request()

    failure = _failure_from_worker_exception(
        request,
        RuntimeError("provider stopped at a segment boundary"),
        cancelled=True,
    )

    assert failure.code is TranscriptionFailureCode.CANCELLED
    assert failure.recovery_actions == ()


def test_parakeet_provider_persists_normalized_managed_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Model_Artifacts import ArtifactRef
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxRuntime

    root = ArtifactRef("parakeet-v2", "root-revision", "int8")
    dependency = ArtifactRef("silero-vad", "vad-revision", "f32")
    model_root = tmp_path / "model"
    vad_root = tmp_path / "vad"
    model_root.mkdir()
    vad_root.mkdir()
    handle = SimpleNamespace(
        root=root,
        closure=(root, dependency),
        paths=((root, model_root), (dependency, vad_root)),
    )
    captured = {}

    class _Runtime:
        def transcribe(self, **kwargs):
            captured["transcribe"] = kwargs
            return TranscriptionResult(
                text="managed transcript",
                segments=(),
                provenance=TranscriptionProvenance(
                    schema_version=1,
                    attempt_id=kwargs["attempt_id"],
                    batch_id=kwargs["batch_id"],
                    job_id=kwargs["job_id"],
                    retry_of_attempt_id=None,
                    retry_of_job_id=None,
                    provider_id="parakeet-onnx",
                    model_id=PARAKEET_V2_MODEL,
                    artifact_root=root.lease_key(),
                    artifact_dependencies=(dependency.lease_key(),),
                    precision="int8",
                    requested_device=ExecutionDevice.CPU,
                    effective_device=ExecutionDevice.CPU,
                    requested_language="en",
                    effective_language="en",
                    detected_language=None,
                    task=TranscriptionTask.TRANSCRIBE,
                ),
                produced_capabilities=ProducedCapabilities(
                    timestamps=TimestampGranularity.NONE,
                    punctuation=True,
                    capitalization=True,
                    vad=False,
                    diarization=False,
                ),
                duration_seconds=1.0,
                timings=TranscriptionTimings(total_seconds=0.1),
            )

        def close(self):
            return None

    def fake_load(**kwargs):
        captured["load"] = kwargs
        return _Runtime()

    monkeypatch.setattr(ParakeetOnnxRuntime, "load", fake_load)
    request = ExecutorRequest(
        generation=1,
        attempt_id="attempt-managed",
        job_id="job-managed",
        source=FileAudioSource(tmp_path / "speech.wav"),
        identity=_identity(
            root_revision=root.revision,
            closure_fingerprint="closure",
            local_snapshot_token=None,
        ),
        options={
            "language": "en",
            "timestamps": False,
            "transcription_context": {"batch_id": "batch-managed"},
        },
    )

    provider = _parakeet_provider(request, model_root, handle, lambda: False)
    payload = provider.runner(str(request.source.path))

    assert captured["load"]["model_root"] == model_root
    assert captured["load"]["vad_root"] == vad_root
    assert captured["load"]["artifact_root"] == root.lease_key()
    assert captured["load"]["artifact_dependencies"] == (dependency.lease_key(),)
    provenance = payload["transcription_provenance"]
    assert provenance["artifact_root"] == {
        "artifact_id": root.artifact_id,
        "revision": root.revision,
        "variant": root.variant,
    }
    assert provenance["artifact_dependencies"] == [
        {
            "artifact_id": dependency.artifact_id,
            "revision": dependency.revision,
            "variant": dependency.variant,
        }
    ]
    assert provenance["batch_id"] == "batch-managed"


def test_parakeet_provider_keeps_external_root_out_of_vad_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Model_Artifacts import ArtifactRef
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxRuntime

    dependency = ArtifactRef("silero-vad", "vad-revision", "f32")
    model_root = tmp_path / "private-external-model"
    vad_root = tmp_path / "managed-vad"
    model_root.mkdir()
    vad_root.mkdir()
    model_file = model_root / "encoder.onnx"
    model_file.write_bytes(b"model")
    snapshot = snapshot_local_source((model_file,))
    handle = SimpleNamespace(
        references=(dependency,),
        paths=((dependency, vad_root),),
        lease_keys=(dependency.lease_key(),),
    )
    captured: dict[str, object] = {}

    class _Runtime:
        def close(self) -> None:
            return None

    def fake_load(**kwargs):
        captured.update(kwargs)
        return _Runtime()

    monkeypatch.setattr(ParakeetOnnxRuntime, "load", fake_load)
    request = ExecutorRequest(
        generation=1,
        attempt_id="attempt-external",
        job_id="job-external",
        source=FileAudioSource(tmp_path / "speech.wav"),
        identity=_identity(
            root_revision="catalog-revision",
            closure_fingerprint=None,
            local_snapshot_token=snapshot.token,
        ),
        options={"language": "en"},
        local_source=snapshot,
        managed_store_root=tmp_path / "store",
        managed_dependency_refs=(("silero-vad", "vad-revision", "f32"),),
    )

    provider = _parakeet_provider(request, model_root, handle, lambda: False)

    assert captured["model_root"] == model_root
    assert captured["vad_root"] == vad_root
    assert captured["artifact_root"] is None
    assert captured["artifact_dependencies"] == (dependency.lease_key(),)
    assert str(model_root) not in repr(captured["artifact_dependencies"])
    provider.close()


@pytest.mark.parametrize("suffix", (".wav", ".mp4"))
def test_parakeet_failure_survives_media_parse_and_executor_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
        DirectLocalSTTIngestError,
    )
    from tldw_chatbook.Local_Ingestion.video_processing import LocalVideoProcessor
    from tldw_chatbook.Model_Artifacts import ArtifactRef
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxRuntime

    root = ArtifactRef("parakeet-v2", "root-revision", "int8")
    dependency = ArtifactRef("silero-vad", "vad-revision", "f32")
    model_root = tmp_path / "model"
    vad_root = tmp_path / "vad"
    model_root.mkdir()
    vad_root.mkdir()
    source = tmp_path / f"speech{suffix}"
    source.write_bytes(b"fixture")
    handle = SimpleNamespace(
        root=root,
        closure=(root, dependency),
        paths=((root, model_root), (dependency, vad_root)),
    )
    runtime = ParakeetOnnxRuntime(
        model=SimpleNamespace(recognize=lambda _path: "unused"),
        vad=None,
        model_id=PARAKEET_V2_MODEL,
        precision="int8",
        artifact_root=root.lease_key(),
        artifact_dependencies=(dependency.lease_key(),),
        model_load_seconds=0.1,
        audio_reader=lambda *_args, **_kwargs: None,
        pad_list=lambda _chunks: None,
        duration_reader=lambda _path: 40.0,
    )
    monkeypatch.setattr(ParakeetOnnxRuntime, "load", lambda **_kwargs: runtime)
    if suffix == ".mp4":
        extracted_audio = tmp_path / "extracted.wav"
        extracted_audio.write_bytes(b"fixture")
        monkeypatch.setattr(
            LocalVideoProcessor,
            "_extract_audio_from_video",
            lambda self, *_args, **_kwargs: str(extracted_audio),
        )
    request = ExecutorRequest(
        generation=1,
        attempt_id="attempt-parakeet-failure",
        job_id="job-parakeet-failure",
        source=FileAudioSource(source),
        identity=_identity(
            root_revision=root.revision,
            closure_fingerprint="closure",
            local_snapshot_token=None,
        ),
        options={
            "transcription_provider": "parakeet-onnx",
            "transcription_model": PARAKEET_V2_MODEL,
            "transcription_precision": "int8",
            "language": "en",
            "timestamps": True,
            "transcription_context": {
                "attempt_id": "attempt-parakeet-failure",
                "batch_id": "batch-parakeet-failure",
                "job_id": "job-parakeet-failure",
            },
        },
    )
    provider = _parakeet_provider(request, model_root, handle, lambda: False)

    with pytest.raises(DirectLocalSTTIngestError) as raised:
        _default_parse_job(
            source,
            dict(request.options),
            transcription_runner=provider.runner,
        )

    failure = _failure_from_exception(request, raised.value)
    assert failure.code is TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE
    assert failure.recovery_actions == ("retry_faster_whisper",)
    assert failure.failed_attempt is not None
    assert failure.failed_attempt["attempt_id"] == "attempt-parakeet-failure"
    assert failure.failed_attempt["batch_id"] == "batch-parakeet-failure"
    assert failure.failed_attempt["job_id"] == "job-parakeet-failure"
    assert failure.failed_attempt["provider_id"] == "parakeet-onnx"


def test_managed_parakeet_provider_rejects_a_closure_without_vad(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Model_Artifacts import ArtifactRef

    root = ArtifactRef("parakeet-v2", "root-revision", "int8")
    model_root = tmp_path / "model"
    model_root.mkdir()
    handle = SimpleNamespace(root=root, closure=(root,), paths=((root, model_root),))

    request = _request()
    with pytest.raises(Exception) as raised:
        _parakeet_provider(request, model_root, handle, lambda: False)

    failure = _failure_from_exception(request, raised.value)
    assert failure.code is TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE
    assert failure.failed_attempt is not None
    assert failure.failed_attempt["artifact_root"] == {
        "artifact_id": "parakeet-v2",
        "revision": "root-revision",
        "variant": "int8",
    }


@pytest.mark.parametrize(
    (
        "stage",
        "expected_code",
        "expected_effective_device",
        "model_id",
        "expected_attempt_id",
        "expected_language",
    ),
    [
        (
            "load",
            TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
            None,
            PARAKEET_V2_MODEL,
            "attempt-failure",
            "en",
        ),
        (
            "inference",
            TranscriptionFailureCode.INFERENCE_FAILED,
            "cpu",
            PARAKEET_V3_MODEL,
            "attempt-current",
            "fr",
        ),
    ],
)
def test_parakeet_runtime_failures_preserve_normalized_failed_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    expected_code: TranscriptionFailureCode,
    expected_effective_device: str | None,
    model_id: str,
    expected_attempt_id: str,
    expected_language: str,
) -> None:
    from tldw_chatbook.Model_Artifacts import ArtifactRef
    from tldw_chatbook.STT.parakeet_onnx import ParakeetOnnxRuntime

    root = ArtifactRef("parakeet-v2", "root-revision", "int8")
    dependency = ArtifactRef("silero-vad", "vad-revision", "f32")
    model_root = tmp_path / "model"
    vad_root = tmp_path / "vad"
    model_root.mkdir()
    vad_root.mkdir()
    handle = SimpleNamespace(
        root=root,
        closure=(root, dependency),
        paths=((root, model_root), (dependency, vad_root)),
    )

    class _FailingRuntime:
        def transcribe(self, **_kwargs):
            raise RuntimeError("private inference detail")

        def close(self):
            return None

    def fake_load(**_kwargs):
        if stage == "load":
            raise RuntimeError("private load detail")
        return _FailingRuntime()

    monkeypatch.setattr(ParakeetOnnxRuntime, "load", fake_load)
    request = ExecutorRequest(
        generation=1,
        attempt_id="attempt-failure",
        job_id="job-failure",
        source=FileAudioSource(tmp_path / "speech.wav"),
        identity=_identity(
            model_id=model_id,
            root_revision=root.revision,
            closure_fingerprint="closure",
            local_snapshot_token=None,
        ),
        options={
            "language": "en",
            "transcription_context": {"batch_id": "batch-failure"},
        },
    )

    try:
        provider = _parakeet_provider(request, model_root, handle, lambda: False)
        provider.runner(
            str(request.source.path),
            attempt_id="attempt-current",
            batch_id="batch-current",
            job_id="job-current",
            language="fr",
        )
    except Exception as error:
        failure = _failure_from_exception(request, error)
    else:
        raise AssertionError("Parakeet failure was not raised")

    assert failure.code is expected_code
    assert failure.recovery_actions == ("retry_faster_whisper",)
    assert failure.failed_attempt is not None
    assert failure.failed_attempt["attempt_id"] == expected_attempt_id
    assert failure.failed_attempt["batch_id"] == (
        "batch-failure" if stage == "load" else "batch-current"
    )
    assert failure.failed_attempt["job_id"] == (
        "job-failure" if stage == "load" else "job-current"
    )
    assert failure.failed_attempt["provider_id"] == "parakeet-onnx"
    assert failure.failed_attempt["model_id"] == model_id
    assert failure.failed_attempt["requested_language"] == expected_language
    assert failure.failed_attempt["error_code"] == expected_code.value
    assert failure.failed_attempt["effective_device"] == expected_effective_device
    assert "private" not in str(failure)


def test_loaded_residency_is_reported_before_first_parse_failure(
    tmp_path: Path,
) -> None:
    executor = _resident_executor()
    first = _Callbacks()
    second = _Callbacks()
    original = _identity(
        root_revision=None,
        closure_fingerprint=None,
        local_snapshot_token=None,
    )
    changed = _identity(
        model_id="replacement-model",
        root_revision=None,
        closure_fingerprint=None,
        local_snapshot_token=None,
    )
    try:
        first_generation = executor.submit(
            attempt_id="first-fails",
            job_id="job-first-fails",
            source=FileAudioSource(tmp_path / "fixture.wav"),
            identity=original,
            options={
                "transcription_provider": "parakeet-onnx",
                "test_worker_fail_parse": True,
            },
            on_result=first.on_result,
            on_failure=first.on_failure,
        )
        _wait_for_terminal(first)
        assert first.failures[0].code is TranscriptionFailureCode.INFERENCE_FAILED
        assert executor.resident_identity == original

        second_generation = executor.submit(
            attempt_id="second-succeeds",
            job_id="job-second-succeeds",
            source=FileAudioSource(tmp_path / "fixture.wav"),
            identity=changed,
            options={"transcription_provider": "parakeet-onnx"},
            on_result=second.on_result,
            on_failure=second.on_failure,
        )
        _wait_for_terminal(second)

        assert second_generation > first_generation
        assert second.results
        assert second.failures == []
    finally:
        executor.close()


def test_worker_crash_releases_managed_closure_lease(tmp_path: Path) -> None:
    service, root, dependency, identity = _managed_request_values(tmp_path)
    executor = _resident_executor()
    callbacks = _Callbacks()
    try:
        executor.submit(
            attempt_id="crash",
            job_id="job-crash",
            source=FileAudioSource(tmp_path / "fixture.wav"),
            identity=identity,
            options={
                "transcription_provider": "parakeet-onnx",
                "test_worker_crash": True,
            },
            managed_store_root=tmp_path / "store",
            managed_artifact_ref=(
                root.reference.artifact_id,
                root.reference.revision,
                root.reference.variant,
            ),
            on_result=callbacks.on_result,
            on_failure=callbacks.on_failure,
        )
        _wait_for_terminal(callbacks)
        assert callbacks.failures[0].code is TranscriptionFailureCode.ENGINE_CRASHED
    finally:
        executor.close()

    ModelArtifactService(tmp_path / "store").delete(dependency.reference)
    assert service.artifact_path(dependency.reference).exists() is False


def test_force_stop_releases_managed_closure_lease(tmp_path: Path) -> None:
    service, root, dependency, identity = _managed_request_values(tmp_path)
    executor = _resident_executor()
    callbacks = _Callbacks()
    try:
        executor.submit(
            attempt_id="held",
            job_id="job-held",
            source=FileAudioSource(tmp_path / "fixture.wav"),
            identity=identity,
            options={
                "transcription_provider": "parakeet-onnx",
                "test_worker_hold": True,
            },
            managed_store_root=tmp_path / "store",
            managed_artifact_ref=(
                root.reference.artifact_id,
                root.reference.revision,
                root.reference.variant,
            ),
            on_event=callbacks.on_event,
            on_result=callbacks.on_result,
            on_failure=callbacks.on_failure,
        )
        _wait_until(
            lambda: any(
                event.phase is WorkerPhase.TRANSCRIBING for event in callbacks.events
            )
        )
        assert executor.force_stop("held") is True
        _wait_for_terminal(callbacks)
        _wait_until(lambda: executor.retiring is False)
    finally:
        executor.close()

    ModelArtifactService(tmp_path / "store").delete(dependency.reference)
    assert service.artifact_path(dependency.reference).exists() is False


def test_worker_revalidates_unmanaged_local_snapshot_before_reuse(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"first-model")
    snapshot = snapshot_local_source((model,))
    identity = _identity(
        root_revision=None,
        closure_fingerprint=None,
        local_snapshot_token=snapshot.token,
    )
    executor = _resident_executor()
    first = _Callbacks()
    second = _Callbacks()
    try:
        executor.submit(
            attempt_id="one",
            job_id="job-one",
            source=FileAudioSource(tmp_path / "fixture.wav"),
            identity=identity,
            options={"transcription_provider": "parakeet-onnx"},
            local_source=snapshot,
            on_result=first.on_result,
            on_failure=first.on_failure,
        )
        _wait_for_terminal(first)
        model.write_bytes(b"replacement-model-with-new-identity")
        executor.submit(
            attempt_id="two",
            job_id="job-two",
            source=FileAudioSource(tmp_path / "fixture.wav"),
            identity=identity,
            options={"transcription_provider": "parakeet-onnx"},
            local_source=snapshot,
            on_result=second.on_result,
            on_failure=second.on_failure,
        )
        _wait_for_terminal(second)

        assert second.results == []
        assert second.failures[0].code is TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE
        assert second.failures[0].recovery_actions == ("retry_faster_whisper",)
    finally:
        executor.close()
