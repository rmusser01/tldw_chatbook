"""Real isolated model ownership with child-installed device/model/egress guards."""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import ModuleType, SimpleNamespace

import pytest


def install_child_guards():
    """Spawn does not inherit pytest monkeypatches; fail closed in each child."""
    import importlib.abc
    import socket

    class NoRealDependencies(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {
                "mlx",
                "torch",
                "transformers",
                "nemo",
                "onnx_asr",
                "faster_whisper",
                "parakeet_mlx",
                "sounddevice",
                "pyaudio",
                "huggingface_hub",
                "tldw_voice_aec",
            }:
                raise AssertionError("real model or device access forbidden")

    def no_network(*args, **kwargs):
        raise AssertionError("network access forbidden")

    sys.meta_path.insert(0, NoRealDependencies())
    socket.create_connection = no_network
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network


def fake_service_runtime(
    events, mode, provider, options, model, language, *, forbidden_endpoints=()
):
    """Replace only the heavy facade; exercise the actual child adapter."""
    install_child_guards()
    if forbidden_endpoints and sys.platform != "win32":
        inherited = set()
        for descriptor in os.listdir("/dev/fd"):
            try:
                metadata = os.fstat(int(descriptor))
            except OSError:
                continue
            inherited.add((metadata.st_dev, metadata.st_ino))
        assert not inherited.intersection(forbidden_endpoints), (
            "model inherited app protocol endpoint"
        )
    from tldw_chatbook.Audio.parakeet_voice_worker import _load_runtime

    def record(operation, *values):
        events.send((operation, os.getpid(), *values))

    class Candidate:
        def __init__(self):
            self.index = 0

        def __enter__(self):
            record("enter")
            if mode == "enter-error":
                raise RuntimeError("private context mutation")
            if mode == "enter-hang":
                time.sleep(60)
            return self

        def process_audio(self, pcm):
            assert isinstance(pcm, bytes)
            record("push", len(pcm))
            if mode == "crash":
                os._exit(19)
            if mode == "push-systemexit":
                raise SystemExit(17)
            if mode == "push-systemexit-zero":
                raise SystemExit(0)
            if mode == "push-keyboardinterrupt":
                raise KeyboardInterrupt()
            if mode == "push-hang":
                time.sleep(60)
            if mode == "long-text":
                return {"text": "x" * 65_537, "partial": False}
            if mode == "split-long-text":
                return {"partial": "x" * 32_769, "final": "y" * 32_768}
            if mode == "limit-text":
                return {"text": "x" * 65_536, "partial": False, "opaque_model": self}
            self.index += 1
            return (
                {"partial": "hello", "cumulative": True},
                {"text": "hullo", "partial": True, "cumulative": True},
                {"final": "world", "cumulative": False},
            )[min(self.index - 1, 2)]

        def __exit__(self, *args):
            record("exit")
            if mode == "exit-error":
                raise RuntimeError("private context mutation")
            if mode == "exit-hang":
                time.sleep(60)

        def close(self):
            record("candidate-close")

    class Service:
        def __init__(self):
            record("service")
            self.config = {"device": "cpu", "compute_type": "int8"}

        def create_streaming_transcriber(self, **kwargs):
            record("prepare", kwargs, dict(self.config))
            if mode == "prepare-error":
                raise RuntimeError("private preparation failure")
            if provider == "parakeet-mlx":
                return ParakeetCandidate()
            if mode.startswith("aclose-"):
                return AsyncCandidate()
            return None if mode == "batch" else Candidate()

        def transcribe_buffer(self, **kwargs):
            assert provider != "parakeet-mlx", (
                "Parakeet must retain in-memory model inference"
            )
            pcm = kwargs.pop("audio_data")
            record("rolling", len(pcm), kwargs)
            return {"text": "batch result", "partial": False, "cumulative": True}

        def cleanup(self):
            record("cleanup")
            if mode == "cleanup-error":
                raise RuntimeError("private cleanup failure")
            if mode == "cleanup-hang":
                time.sleep(60)

    class AsyncCandidate:
        def process_audio(self, pcm):
            return {"partial": "hello"}

        async def aclose(self):
            record("candidate-aclose-start")
            await asyncio.sleep(60 if mode == "aclose-hang" else 0.02)
            if mode == "aclose-error":
                raise RuntimeError("private async cleanup failure")
            record("candidate-aclose-done")

    class ParakeetCandidate:
        def __init__(self):
            self.model = ParakeetModel()

        def close(self):
            record("candidate-close")

    class ParakeetModel:
        preprocessor_config = "fake"

        def transcribe_stream(self, *, context_size):
            assert context_size == (64, 64)
            record("model-open")
            return ParakeetContext()

        def generate(self, mel):
            assert isinstance(mel, bytes)
            record("model-generate", len(mel))
            return [SimpleNamespace(text=f"rolling:{os.getpid()}:{len(mel)}")]

    class ParakeetContext:
        result = SimpleNamespace(text="")

        def __enter__(self):
            record("enter")
            return self

        def add_audio(self, audio):
            assert isinstance(audio, bytes)
            record("push", len(audio))
            self.result = SimpleNamespace(text=f"{os.getpid()}:{len(audio)}")

        def __exit__(self, *args):
            record("exit")

    if provider == "parakeet-mlx":
        from tldw_chatbook.Audio.parakeet_voice_worker import _ParakeetRuntime

        # Keep the real native/rolling adapter; replace its MLX conversion only.
        _ParakeetRuntime._audio = staticmethod(lambda pcm: pcm)
        audio_module = ModuleType("parakeet_mlx.audio")
        audio_module.get_logmel = lambda audio, config: audio
        sys.modules[audio_module.__name__] = audio_module

    fake_module = ModuleType("tldw_chatbook.Local_Ingestion.transcription_service")
    fake_module.TranscriptionService = Service
    sys.modules[fake_module.__name__] = fake_module
    return _load_runtime(model, language, provider=provider, options=options)


@pytest.fixture
def service_factory():
    owners = []

    def make(mode="native", **kwargs):
        from tldw_chatbook.Audio.parakeet_voice_worker import LocalVoiceSttProcess

        provider = kwargs.pop("provider", "faster-whisper")
        receiver, sender = multiprocessing.get_context("spawn").Pipe(False)
        service = LocalVoiceSttProcess(
            provider=provider, model="test-model", language="en", **kwargs
        )
        service._runtime_factory = partial(
            fake_service_runtime, sender, mode, provider, kwargs.get("options")
        )
        owners.append((service, receiver, sender))
        return service, receiver

    yield make
    for service, receiver, sender in owners:
        service.close()
        receiver.close()
        sender.close()
        assert service.reaped


def events_from(receiver):
    events = []
    while receiver.poll(0.01):
        events.append(receiver.recv())
    return events


def rolling(service, pcm=bytes(960)):
    return service.transcribe_buffer(
        audio_data=pcm, sample_rate=48_000, channels=1, sample_width=2
    )


def test_native_results_and_all_resource_operations_stay_in_model_pid(service_factory):
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

    service, receiver = service_factory()
    candidate = service.create_streaming_transcriber()
    with candidate.stream_context(context_size=(64, 64)) as stream:
        assert stream.process_audio(bytes(960)) == {
            "partial": "hello",
            "cumulative": True,
        }
        assert stream.process_audio(bytes(960)) == {
            "text": "hullo",
            "partial": True,
            "cumulative": True,
        }
        assert stream.process_audio(bytes(960)) == {
            "final": "world",
            "cumulative": False,
        }
    assert rolling(service)["text"] == "batch result"
    service.close()
    calls = events_from(receiver)
    assert [call[0] for call in calls] == [
        "service",
        "prepare",
        "enter",
        "push",
        "push",
        "push",
        "exit",
        "candidate-close",
        "rolling",
        "cleanup",
    ]
    assert {call[1] for call in calls} == {service._process.pid}
    assert service._process.pid != os.getpid()
    assert not service._process.is_alive()
    assert service.cleanup_outcome is AttemptCleanupOutcome.CLEAN


def test_parakeet_facade_selects_same_native_model_for_stream_and_rolling(
    service_factory,
):
    from tldw_chatbook.Audio.parakeet_voice_worker import LocalSttOptions

    service, receiver = service_factory(
        provider="parakeet-mlx", options=LocalSttOptions(precision="fp16")
    )
    candidate = service.create_streaming_transcriber()
    with candidate.model.transcribe_stream(context_size=(64, 64)) as stream:
        stream.add_pcm16(bytes(960))
        assert stream.result.text == f"{service._process.pid}:960"
    assert rolling(service) == {"text": f"rolling:{service._process.pid}:960"}
    service.close()
    calls = events_from(receiver)
    assert [call[0] for call in calls] == [
        "service",
        "prepare",
        "model-open",
        "enter",
        "push",
        "exit",
        "model-generate",
        "candidate-close",
        "cleanup",
    ]
    assert calls[1][2] == {
        "provider": "parakeet-mlx",
        "model": "test-model",
        "source_lang": "en",
        "precision": "fp16",
    }
    assert {call[1] for call in calls} == {service._process.pid}
    assert service._process.pid != os.getpid()


@pytest.mark.asyncio
async def test_generic_native_merges_partial_cumulative_and_final_segment(
    service_factory,
):
    from tldw_chatbook.Audio.duplex_contracts import AudioFrame
    from tldw_chatbook.Audio.voice_transcription import (
        _NativeStreamingStt,
        _SerialSttWorker,
    )

    service, _ = service_factory()
    worker = _SerialSttWorker()
    candidate = await worker.run(service.create_streaming_transcriber)
    native = _NativeStreamingStt(
        service,
        provider="faster-whisper",
        model="test-model",
        language="en",
        prepared_candidate=candidate,
        serial_worker=worker,
    )
    results, failures = [], []
    native.start(results.append, failures.append, lambda _: None)
    try:
        for n in range(3):
            native.submit(
                AudioFrame(n, n * 10_000_000, (n + 1) * 10_000_000, bytes(960))
            )
        await asyncio.wait_for(native._frames.join(), 3)
        assert [result.revisable_text or result.stable_text for result in results] == [
            "hello",
            "hullo",
            "hullo world",
        ]
        assert [result.is_final for result in results] == [False, False, True]
        assert [result.covered_through_ns for result in results] == [
            10_000_000,
            20_000_000,
            30_000_000,
        ]
        assert not failures
    finally:
        await native.close()
        await asyncio.to_thread(service.close)
        await worker.close()


@pytest.mark.asyncio
async def test_batch_only_prepare_preserves_rolling_fallback_and_settings(
    service_factory,
):
    from tldw_chatbook.Audio.parakeet_voice_worker import LocalSttOptions
    from tldw_chatbook.Audio.voice_transcription import (
        _prepare_streaming_candidate,
        _SerialSttWorker,
    )

    options = LocalSttOptions(device="cpu", compute_type="float32", precision="fp32")
    service, receiver = service_factory("batch", options=options)
    worker = _SerialSttWorker()
    try:
        candidate, native = await _prepare_streaming_candidate(
            service,
            provider="faster-whisper",
            model="test-model",
            language="en",
            serial_worker=worker,
        )
        assert candidate is None and native is False
        assert rolling(service) == {
            "text": "batch result",
            "partial": False,
            "cumulative": True,
        }
        service.close()
        calls = events_from(receiver)
        prepare = next(call for call in calls if call[0] == "prepare")
        assert prepare[2] == {
            "provider": "faster-whisper",
            "model": "test-model",
            "source_lang": "en",
            "precision": "fp32",
            "compute_type": "float32",
        }
        assert prepare[3] == {"device": "cpu", "compute_type": "float32"}
        assert {call[1] for call in calls} == {service._process.pid}
    finally:
        await asyncio.to_thread(service.close)
        await worker.close()


def test_generic_context_identity_excludes_rolling_and_stale_handles(service_factory):
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetFailure

    service, _ = service_factory()
    candidate = service.create_streaming_transcriber()
    context = candidate.stream_context(context_size=(64, 64))
    with context as stream:
        with pytest.raises(ParakeetFailure) as busy:
            rolling(service)
        assert busy.value.reason == "context_busy"
        with pytest.raises(ParakeetFailure) as stale:
            service._request("push", "stale", bytes(960))
        assert stale.value.reason == "identity_mismatch"
        assert stream.process_audio(bytes(960))["partial"] == "hello"
    with pytest.raises(ParakeetFailure) as stale:
        stream.process_audio(bytes(960))
    assert stale.value.reason == "identity_mismatch"


@pytest.mark.parametrize(
    "phase",
    ["enter-error", "exit-error", "enter-hang", "push-hang", "exit-hang", "crash"],
)
def test_abnormal_context_or_inference_fences_fallback_and_retains_forced_cleanup(
    service_factory, phase
):
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetServiceUnavailable
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

    service, _ = service_factory(phase)
    candidate = service.create_streaming_transcriber()
    service._request_timeout = 0.08
    with pytest.raises(ParakeetServiceUnavailable):
        with candidate.stream_context(context_size=(64, 64)) as stream:
            stream.process_audio(bytes(960))
    assert service.reaped
    assert service.cleanup_outcome is AttemptCleanupOutcome.FORCE_CLOSED
    with pytest.raises(ParakeetServiceUnavailable):
        rolling(service)
    service.close()
    assert service.cleanup_outcome is AttemptCleanupOutcome.FORCE_CLOSED


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("native", "clean"),
        ("cleanup-error", "force_closed"),
        ("cleanup-hang", "force_closed"),
    ],
)
def test_cleanup_receipt_requires_actual_candidate_and_service_cleanup(
    service_factory, mode, expected
):
    service, receiver = service_factory(mode)
    service.create_streaming_transcriber()
    service.close()
    assert service.cleanup_outcome.value == expected
    assert service.reaped
    assert [call[0] for call in events_from(receiver)] == [
        "service",
        "prepare",
        "candidate-close",
        "cleanup",
    ]


@pytest.mark.parametrize(
    "mode, expected, expected_calls",
    [
        (
            "aclose-clean",
            "clean",
            ["candidate-aclose-start", "candidate-aclose-done", "cleanup"],
        ),
        ("aclose-error", "force_closed", ["candidate-aclose-start", "cleanup"]),
        ("aclose-hang", "force_closed", ["candidate-aclose-start"]),
    ],
)
def test_async_only_candidate_cleanup_finishes_before_clean_receipt(
    service_factory, mode, expected, expected_calls
):
    service, receiver = service_factory(mode)
    service.create_streaming_transcriber()
    service.close()
    assert service.cleanup_outcome.value == expected
    calls = events_from(receiver)
    assert [call[0] for call in calls] == ["service", "prepare", *expected_calls]
    assert {call[1] for call in calls} == {service._process.pid}


@pytest.mark.parametrize("mode", ["long-text", "split-long-text"])
def test_pcm_and_total_text_bounds_apply_before_crossing_model_pipe(
    service_factory, mode
):
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetFailure

    service, _ = service_factory(mode)
    candidate = service.create_streaming_transcriber()
    with candidate.stream_context(context_size=(64, 64)) as stream:
        for pcm in (b"x", bytes(960_002)):
            with pytest.raises(ValueError):
                stream.process_audio(pcm)
        with pytest.raises(ParakeetFailure) as long_result:
            stream.process_audio(bytes(960))
        assert "x" * 20 not in str(long_result.value)


def test_exact_pcm_and_text_limits_succeed_without_serializing_native_metadata(
    service_factory,
):
    service, receiver = service_factory("limit-text")
    candidate = service.create_streaming_transcriber()
    with candidate.stream_context(context_size=(64, 64)) as stream:
        assert stream.process_audio(bytes(960_000)) == {
            "text": "x" * 65_536,
            "partial": False,
        }
    service.close()
    assert (
        next(call for call in events_from(receiver) if call[0] == "push")[2] == 960_000
    )


def test_failed_model_preparation_is_terminal_and_cannot_be_retried_as_rolling(
    service_factory,
):
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetServiceUnavailable
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

    service, receiver = service_factory("prepare-error")
    with pytest.raises(ParakeetServiceUnavailable):
        service.create_streaming_transcriber()
    assert service.reaped
    assert service.cleanup_outcome is AttemptCleanupOutcome.FORCE_CLOSED
    with pytest.raises(ParakeetServiceUnavailable):
        rolling(service)
    assert [call[0] for call in events_from(receiver)] == [
        "service",
        "prepare",
        "cleanup",
    ]


@pytest.mark.parametrize(
    "mode", ["push-systemexit", "push-systemexit-zero", "push-keyboardinterrupt"]
)
def test_terminal_cleanup_record_cannot_be_misread_as_inference_success(
    service_factory, mode
):
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetServiceUnavailable
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

    service, _ = service_factory(mode)
    candidate = service.create_streaming_transcriber()
    stream = candidate.stream_context(context_size=(64, 64)).__enter__()
    with pytest.raises(ParakeetServiceUnavailable):
        stream.process_audio(bytes(960))
    assert service.closed and service.reaped
    assert service.cleanup_outcome is AttemptCleanupOutcome.FORCE_CLOSED
    with pytest.raises(ParakeetServiceUnavailable):
        rolling(service)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["enter", "push", "exit"])
async def test_cancelled_context_and_close_observers_retain_original_model_outcome(
    service_factory, phase
):
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome
    from tldw_chatbook.Audio.voice_transcription import _SerialSttWorker

    service, receiver = service_factory(f"{phase}-hang")
    candidate = service.create_streaming_transcriber()
    worker = _SerialSttWorker()

    async def use_context():
        async with worker.parakeet_context(service, candidate.stream_context) as stream:
            await worker.run_owned(stream.process_audio, bytes(960))

    using = asyncio.create_task(use_context())
    closing = None
    try:
        async with asyncio.timeout(3):
            while True:
                if receiver.poll() and receiver.recv()[0] == phase:
                    break
                await asyncio.sleep(0.005)
        using.cancel()
        await asyncio.sleep(0)
        assert worker._model_lock.locked()
        closing = asyncio.create_task(asyncio.to_thread(service.close))
        async with asyncio.timeout(3):
            while not service.closed:
                await asyncio.sleep(0.005)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        await asyncio.wait_for(asyncio.gather(using, return_exceptions=True), 4)
        async with asyncio.timeout(3):
            while service.cleanup_outcome is AttemptCleanupOutcome.DETACHED:
                await asyncio.sleep(0.005)
        assert service.cleanup_outcome is AttemptCleanupOutcome.FORCE_CLOSED
        assert service.reaped
        assert not worker._model_lock.locked()
        service.close()
        assert service.cleanup_outcome is AttemptCleanupOutcome.FORCE_CLOSED
    finally:
        await asyncio.to_thread(service.close)
        await asyncio.gather(
            using, *([closing] if closing else []), return_exceptions=True
        )
        await worker.close()


def test_unopened_cleanup_is_clean_and_unconfirmed_reap_never_is():
    from types import SimpleNamespace
    from tldw_chatbook.Audio.parakeet_voice_worker import LocalVoiceSttProcess
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

    service = LocalVoiceSttProcess(provider="faster-whisper", model=None, language="en")
    service.close()
    assert service.cleanup_outcome is AttemptCleanupOutcome.CLEAN
    service = LocalVoiceSttProcess(provider="faster-whisper", model=None, language="en")
    service._process = SimpleNamespace(
        is_alive=lambda: True,
        join=lambda _: None,
        terminate=lambda: None,
        kill=lambda: None,
    )
    service._connection = SimpleNamespace(
        send=lambda _: None, poll=lambda: False, close=lambda: None
    )
    with pytest.raises(RuntimeError, match="not_reaped"):
        service.close()
    assert service.cleanup_outcome is AttemptCleanupOutcome.DETACHED
    assert service.closed and not service.reaped


@pytest.mark.parametrize(
    "values",
    [{"device": "http://remote"}, {"compute_type": True}, {"precision": "secret"}],
)
def test_typed_local_options_reject_invalid_values(values):
    from tldw_chatbook.Audio.parakeet_voice_worker import LocalSttOptions

    with pytest.raises(ValueError):
        LocalSttOptions(**values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"provider": "remote-whisper"},
        {"model": "x" * 513},
        {"language": "x" * 33},
        {"options": {"api_key": "secret"}},
    ],
)
def test_only_bounded_local_settings_can_construct_service(kwargs):
    from tldw_chatbook.Audio.parakeet_voice_worker import LocalVoiceSttProcess

    values = {"provider": "faster-whisper", "model": "test-model", "language": "en"}
    values.update(kwargs)
    with pytest.raises(ValueError):
        LocalVoiceSttProcess(**values)


def test_parent_eof_after_model_ready_closes_model_without_protocol_endpoints():
    # The audio entry's private descriptors must not enter the spawned model.
    script = """
import os
import multiprocessing
from functools import partial
from Tests.Audio.test_local_voice_stt_process import fake_service_runtime
from tldw_chatbook.Audio.parakeet_voice_worker import LocalVoiceSttProcess
from tldw_chatbook.Audio.voice_process_entry import isolate_standard_streams
if __name__ == "__main__":
    reader, writer = isolate_standard_streams()
    events, sender = multiprocessing.get_context("spawn").Pipe(False)
    endpoints = tuple((os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in (reader, writer))
    service = LocalVoiceSttProcess(provider="faster-whisper", model="test-model", language="en")
    service._runtime_factory = partial(fake_service_runtime, sender, "batch", "faster-whisper", None, forbidden_endpoints=endpoints)
    try:
        service.create_streaming_transcriber()
        os.write(writer, f"{os.getpid()}:{service._process.pid}\\n".encode())
        assert os.read(reader, 1) == b""
        service.close()
        assert service.reaped
        os.write(writer, service.cleanup_outcome.value.encode())
    finally:
        service.close()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=sys.platform != "win32",
    )
    reader = ThreadPoolExecutor(max_workers=1)
    ready = reader.submit(process.stdout.readline)
    try:
        # EOF must arrive after the grandchild is alive, not during preparation.
        identity = ready.result(timeout=5).decode().strip()
        audio_pid, model_pid = map(int, identity.split(":"))
        assert len({os.getpid(), audio_pid, model_pid}) == 3
        process.stdin.close()
        process.stdin = None
        output, error = process.communicate(timeout=10)
        assert process.returncode == 0, error.decode()
        assert output.decode() == "clean"
        if sys.platform != "win32":
            with pytest.raises(ProcessLookupError):
                os.kill(model_pid, 0)
    finally:
        # The audio PID could fail first; still contain its owned descendants.
        if sys.platform != "win32":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
        process.communicate(timeout=3)
        reader.shutdown(wait=True)


@pytest.mark.parametrize("precision", ["int8", "f32"])
def test_onnx_real_facade_and_validated_owner_execute_in_only_the_model_process(
    tmp_path, precision
):
    from Tests.STT.test_resident_buffer_runtime import (
        installed_sources,
        real_facade_process_runtime,
    )
    from tldw_chatbook.Audio.parakeet_voice_worker import (
        LocalSttOptions,
        LocalVoiceSttProcess,
    )
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome
    from tldw_chatbook.Model_Artifacts import ArtifactInUseError

    artifacts, root, dependency, external = installed_sources(tmp_path, precision)
    receiver, sender = multiprocessing.get_context("spawn").Pipe(False)
    options = LocalSttOptions(precision=precision)
    service = LocalVoiceSttProcess(
        provider="parakeet-onnx", model=None, language="en", options=options
    )
    service._runtime_factory = partial(
        real_facade_process_runtime,
        sender,
        artifacts.artifacts_path.parent,
        root,
        external,
        options,
        "normal",
        None,
        None,
    )
    try:
        assert service.create_streaming_transcriber() is None
        assert rolling(service) == {"text": "onnx result"}
        for reference in (root.reference, dependency.reference):
            with pytest.raises(ArtifactInUseError):
                artifacts.delete(reference)
        assert rolling(service) == {"text": "onnx result"}
        service.close()
        calls = events_from(receiver)
        assert [call[0] for call in calls] == [
            "load",
            "infer",
            "infer",
            "native-close",
            "facade-cleanup",
        ]
        assert {call[1] for call in calls} == {service._process.pid}
        assert service._process.pid != os.getpid()
        assert service.cleanup_outcome is AttemptCleanupOutcome.CLEAN
        artifacts.delete(root.reference)
        artifacts.delete(dependency.reference)
    finally:
        service.close()
        receiver.close()
        sender.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["close-error", "close-hang"])
async def test_onnx_failed_or_cancelled_cleanup_keeps_leases_until_real_model_exit(
    tmp_path, mode
):
    from Tests.STT.test_resident_buffer_runtime import (
        installed_sources,
        real_facade_process_runtime,
    )
    from tldw_chatbook.Audio.parakeet_voice_worker import LocalVoiceSttProcess
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome
    from tldw_chatbook.Model_Artifacts import ArtifactInUseError

    artifacts, root, dependency, external = installed_sources(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    receiver, sender = ctx.Pipe(False)
    entered, release = ctx.Event(), ctx.Event()
    service = LocalVoiceSttProcess(provider="parakeet-onnx", model=None, language="en")
    service._runtime_factory = partial(
        real_facade_process_runtime,
        sender,
        artifacts.artifacts_path.parent,
        root,
        external,
        None,
        mode,
        entered,
        release,
    )
    closing = None
    try:
        service.create_streaming_transcriber()
        # close-error does not gate inference; close-hang waits only at cleanup.
        if mode == "close-error":
            release.set()
        assert rolling(service)["text"] == "onnx result"
        closing = asyncio.create_task(asyncio.to_thread(service.close))
        if mode == "close-hang":
            assert await asyncio.to_thread(entered.wait, 2)
            for reference in (root.reference, dependency.reference):
                with pytest.raises(ArtifactInUseError):
                    artifacts.delete(reference)
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
        else:
            await closing
        async with asyncio.timeout(4):
            while service.cleanup_outcome is AttemptCleanupOutcome.DETACHED:
                await asyncio.sleep(0.005)
        assert service.reaped
        assert service.cleanup_outcome is AttemptCleanupOutcome.FORCE_CLOSED
        assert [call[0] for call in events_from(receiver)].count("native-close") == 1
        artifacts.delete(root.reference)
        artifacts.delete(dependency.reference)
    finally:
        release.set()
        await asyncio.to_thread(service.close)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        receiver.close()
        sender.close()


@pytest.mark.timeout(15)
@pytest.mark.parametrize("wire_failure", [None, "send", "close"])
def test_failed_model_cleanup_keeps_artifact_custody_through_terminal_os_exit(
    tmp_path, wire_failure
):
    from Tests.STT.test_resident_buffer_runtime import (
        installed_sources,
        retiring_facade_process_runtime,
    )
    from tldw_chatbook.Audio.parakeet_voice_worker import LocalVoiceSttProcess
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome
    from tldw_chatbook.Model_Artifacts import ArtifactRemovalAvailability

    artifacts, root, dependency, external = installed_sources(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    receiver, sender = ctx.Pipe(False)
    child_control, parent_control = ctx.Pipe(False)
    service = LocalVoiceSttProcess(provider="parakeet-onnx", model=None, language="en")
    service._runtime_factory = partial(
        retiring_facade_process_runtime,
        sender,
        child_control,
        wire_failure,
        artifacts.artifacts_path.parent,
        root,
        external,
        None,
        "close-error",
        None,
        None,
    )
    try:
        service.create_streaming_transcriber()
        assert rolling(service)["text"] == "onnx result"
        service._connection.send(("close", "", None))
        deadline = time.monotonic() + 3
        while True:
            assert receiver.poll(max(0, deadline - time.monotonic()))
            stage = receiver.recv()
            if stage[0] in {"pre-exit", "late-finalizer"}:
                break
        if wire_failure != "send":
            assert service._connection.poll(1)
            assert service._connection.recv() == (
                "cleanup",
                AttemptCleanupOutcome.FORCE_CLOSED,
            )
        assert service._process.is_alive()
        # This samples real exclusive-removal admission, not an owner flag.
        assert [
            artifacts.probe_removal_availability(ref)
            for ref in (root.reference, dependency.reference)
        ] == [ArtifactRemovalAvailability.BUSY] * 2
        assert stage[0] == "pre-exit", "uncertain cleanup must bypass Python finalizers"
        parent_control.send(None)
        service._process.join(3)
        assert not service._process.is_alive()
        assert service._process.exitcode != 0
        service.close()
        assert service.reaped
        assert service.cleanup_outcome is AttemptCleanupOutcome.FORCE_CLOSED
        artifacts.delete(root.reference)
        artifacts.delete(dependency.reference)
    finally:
        service.close()
        for pipe in (receiver, sender, child_control, parent_control):
            pipe.close()
