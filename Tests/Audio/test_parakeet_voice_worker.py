"""Real spawned-process checks, with no model download or audio device."""

from __future__ import annotations

import os
import asyncio
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from typing import get_args

import pytest


def test_failure_reason_vocabulary_is_fixed_and_bounded():
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetFailureReason

    assert set(get_args(ParakeetFailureReason)) == {
        "context_busy",
        "identity_mismatch",
        "closed",
        "disconnected",
        "timeout",
        "ownership_timeout",
        "unknown_native",
    }


class FakeRuntime:
    def __init__(self):
        from Tests.Audio.test_local_voice_stt_process import install_child_guards

        install_child_guards()
        self.mode = "original"

    def open(self, context_size):
        assert context_size == (64, 64)
        return self

    def __enter__(self):
        assert self.mode == "original"
        self.mode = "streaming"
        return self

    def __exit__(self, *_args):
        assert self.mode == "streaming"
        self.mode = "original"

    def push(self, stream, pcm):
        assert stream is self
        assert self.mode == "streaming"
        if pcm == b"hang":
            time.sleep(60)
        if pcm == b"fail":
            raise ValueError("private audio or provider detail")
        if pcm == b"imitate!":
            raise RuntimeError("stream_active")
        return f"{os.getpid()}:{len(pcm)}"

    def rolling(self, pcm):
        assert self.mode == "original"
        return f"rolling:{os.getpid()}:{len(pcm)}"


def fake_runtime(model, language):
    assert model == "test-model" and language == "en"
    return FakeRuntime()


class HungPrewarmRuntime(FakeRuntime):
    def push(self, stream, pcm):
        time.sleep(60)


def hung_prewarm_runtime(model, language):
    return HungPrewarmRuntime()


class GatedPushRuntime(FakeRuntime):
    def __init__(self, entered):
        super().__init__()
        self.entered = entered

    def push(self, stream, pcm):
        self.entered.set()
        return super().push(stream, pcm)


def gated_push_runtime(entered, model, language):
    return GatedPushRuntime(entered)


class UncertainRuntime(FakeRuntime):
    def __init__(self, phase):
        super().__init__()
        self.phase = phase

    def __enter__(self):
        result = super().__enter__()
        if self.phase == "enter":
            raise ValueError("private mutation during context entry")
        return result

    def __exit__(self, *_args):
        if self.phase == "exit":
            raise ValueError("private mutation during context exit")
        return super().__exit__()


def uncertain_runtime(model, language):
    return UncertainRuntime(language)


class GatedExitRuntime(FakeRuntime):
    def __init__(self, entered_exit):
        super().__init__()
        self.entered_exit = entered_exit

    def __exit__(self, *_args):
        self.entered_exit.set()
        time.sleep(60)


def gated_exit_runtime(entered_exit, model, language):
    return GatedExitRuntime(entered_exit)


@pytest.fixture
def service():
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetVoiceProcess

    instance = ParakeetVoiceProcess(model="test-model", language="en")
    instance._runtime_factory = fake_runtime
    try:
        yield instance
    finally:
        instance.close()


def test_native_and_rolling_share_child_and_keep_stream_identity(service):
    from tldw_chatbook.Audio.parakeet_voice_worker import (
        ParakeetFailure,
        ParakeetServiceUnavailable,
    )

    candidate = service.create_streaming_transcriber()
    with candidate.model.transcribe_stream(context_size=(64, 64)) as stream:
        stream.add_pcm16(bytes(960))
        child_pid, size = map(int, stream.result.text.split(":"))
        assert child_pid != os.getpid()
        assert size == 960
        with pytest.raises(ParakeetFailure, match="stream_active") as busy:
            with candidate.model.transcribe_stream(context_size=(64, 64)):
                pass
        assert busy.value.reason == "context_busy"
        stream.add_pcm16(bytes(1920))
        assert stream.result.text == f"{child_pid}:1920"
    with pytest.raises(ParakeetFailure, match="stream_identity") as mismatch:
        stream.add_pcm16(bytes(960))
    assert mismatch.value.reason == "identity_mismatch"
    result = service.transcribe_buffer(
        audio_data=bytes(960), sample_rate=48000, channels=1, sample_width=2
    )
    assert result == {"text": f"rolling:{child_pid}:960"}
    process = service._process
    service.close()
    assert not process.is_alive()
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

    assert service.cleanup_outcome is AttemptCleanupOutcome.CLEAN
    with pytest.raises(ParakeetServiceUnavailable, match="closed") as closed:
        service.create_streaming_transcriber()
    assert closed.value.reason == "closed"


def test_pcm_limits_and_exception_messages_do_not_cross_pipe(service):
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetFailure

    candidate = service.create_streaming_transcriber()
    with candidate.model.transcribe_stream(context_size=(64, 64)) as stream:
        for pcm in (b"x", bytes(960_002), "not pcm"):
            with pytest.raises(ValueError, match="pcm"):
                stream.add_pcm16(pcm)
        with pytest.raises(ParakeetFailure) as error:
            stream.add_pcm16(b"fail")
        assert str(error.value) == "parakeet_worker_ValueError"
        assert error.value.reason == "unknown_native"
        assert error.value.native_exception_type == "ValueError"
        with pytest.raises(ParakeetFailure) as imitation:
            stream.add_pcm16(b"imitate!")
        assert str(imitation.value) == "parakeet_worker_RuntimeError"
        assert imitation.value.reason == "unknown_native"
        assert imitation.value.native_exception_type == "RuntimeError"
    with pytest.raises(ValueError, match="format"):
        service.transcribe_buffer(
            audio_data=bytes(960), sample_rate=16000, channels=1, sample_width=2
        )


def test_hung_inference_shutdown_reaps_child_and_unblocks_caller(service):
    child_entered = multiprocessing.get_context("spawn").Event()
    service._runtime_factory = partial(gated_push_runtime, child_entered)
    candidate = service.create_streaming_transcriber()
    context = candidate.model.transcribe_stream(context_size=(64, 64))
    stream = context.__enter__()

    def infer():
        stream.add_pcm16(b"hang")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(infer)
        try:
            assert child_entered.wait(2)
            start = time.monotonic()
            service.close()
            assert time.monotonic() - start < 4
            with pytest.raises(RuntimeError) as closed:
                future.result(timeout=2)
            assert closed.value.reason == "closed"
        finally:
            service.close()
    assert not service._process.is_alive()
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

    assert service.cleanup_outcome is AttemptCleanupOutcome.FORCE_CLOSED


def test_dead_child_reports_typed_disconnected_category(service):
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetServiceUnavailable

    service.create_streaming_transcriber()
    service._process.terminate()
    service._process.join(1)

    with pytest.raises(ParakeetServiceUnavailable) as disconnected:
        service.create_streaming_transcriber()

    assert disconnected.value.reason == "disconnected"
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

    assert service.cleanup_outcome is AttemptCleanupOutcome.FORCE_CLOSED


def test_rpc_timeout_reaps_child(service):
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetRequestTimeout

    candidate = service.create_streaming_transcriber()
    stream = candidate.model.transcribe_stream(context_size=(64, 64)).__enter__()
    service._request_timeout = 0.05
    with pytest.raises(ParakeetRequestTimeout) as timeout:
        stream.add_pcm16(b"hang")
    assert isinstance(timeout.value, RuntimeError)
    assert timeout.value.reason == "timeout"
    assert not service._process.is_alive()


def test_startup_does_not_wait_indefinitely_for_another_model_loader(service):
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetRequestTimeout
    from tldw_chatbook.Utils.fd_protection import _fd_protection_lock

    with ThreadPoolExecutor(max_workers=1) as executor:
        with _fd_protection_lock:
            future = executor.submit(service.create_streaming_transcriber)
            with pytest.raises(
                ParakeetRequestTimeout, match="fd_protection_busy"
            ) as timeout:
                future.result(timeout=2)
            assert timeout.value.reason == "timeout"
        service.close()
    assert service._process is None


@pytest.mark.asyncio
async def test_dead_prewarm_worker_does_not_advertise_rolling_fallback(service):
    from tldw_chatbook.Chat.console_speculative_voice_session import (
        _prepare_streaming_candidate,
        _SerialSttWorker,
    )

    service._runtime_factory = hung_prewarm_runtime
    service._request_timeout = 0.05
    worker = _SerialSttWorker()
    try:
        with pytest.raises(TimeoutError):
            await _prepare_streaming_candidate(
                service,
                provider="parakeet-mlx",
                model="test-model",
                language="en",
                serial_worker=worker,
            )
    finally:
        await asyncio.to_thread(service.close)
        await worker.close()


def test_native_adapter_sends_pcm_without_importing_mlx(monkeypatch):
    from tldw_chatbook.Chat import console_speculative_voice_session as module

    seen = []
    stream = SimpleNamespace(
        add_pcm16=seen.append, result=SimpleNamespace(text="generated speech")
    )

    def forbidden(_pcm):
        pytest.fail("MLX must stay outside the audio process")

    monkeypatch.setattr(module, "_parakeet_mlx_audio", forbidden)
    result = module._NativeStreamingStt._push_parakeet_stream(
        stream, bytes(960), final=True
    )
    assert seen == [bytes(960)]
    assert result == {"text": "generated speech", "partial": False}


@pytest.mark.parametrize("phase", ["enter", "exit"])
def test_uncertain_context_retires_child_before_rolling(phase):
    from tldw_chatbook.Audio.parakeet_voice_worker import (
        ParakeetServiceUnavailable,
        ParakeetVoiceProcess,
    )

    instance = ParakeetVoiceProcess(model="test-model", language=phase)
    instance._runtime_factory = uncertain_runtime
    try:
        candidate = instance.create_streaming_transcriber()
        with pytest.raises(
            ParakeetServiceUnavailable, match="context_unavailable"
        ) as failure:
            with candidate.model.transcribe_stream(context_size=(64, 64)) as stream:
                stream.add_pcm16(bytes(960))
        assert failure.value.reason == "unknown_native"
        assert instance.closed and instance.reaped
        with pytest.raises(ParakeetServiceUnavailable):
            instance.transcribe_buffer(
                audio_data=bytes(960), sample_rate=48000, channels=1, sample_width=2
            )
    finally:
        instance.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["enter", "exit"])
async def test_uncertain_prewarm_cannot_offer_fallback(phase):
    from tldw_chatbook.Audio.parakeet_voice_worker import (
        ParakeetServiceUnavailable,
        ParakeetVoiceProcess,
    )
    from tldw_chatbook.Chat.console_speculative_voice_session import (
        _prepare_streaming_candidate,
        _SerialSttWorker,
    )

    instance = ParakeetVoiceProcess(model="test-model", language=phase)
    instance._runtime_factory = uncertain_runtime
    worker = _SerialSttWorker()
    try:
        with pytest.raises(ParakeetServiceUnavailable):
            await _prepare_streaming_candidate(
                instance,
                provider="parakeet-mlx",
                model="test-model",
                language=phase,
                serial_worker=worker,
            )
        assert instance.closed and instance.reaped
    finally:
        await asyncio.to_thread(instance.close)
        await worker.close()


@pytest.mark.asyncio
async def test_exact_full_chunk_quiet_boundary_uses_child_and_releases_context(service):
    from tldw_chatbook.Audio.duplex_contracts import AudioFrame
    from tldw_chatbook.Audio.voice_transcription import (
        _NativeStreamingStt,
        _SerialSttWorker,
    )
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

    worker = _SerialSttWorker()
    candidate = await worker.run(service.create_streaming_transcriber)
    native = _NativeStreamingStt(
        service,
        provider="parakeet-mlx",
        model="test-model",
        language="en",
        prepared_candidate=candidate,
        serial_worker=worker,
        quiet_seconds=0.01,
    )
    revisions, failures, settlements = [], [], []
    native.start(revisions.append, failures.append, settlements.append)
    try:
        for n in range(400):
            native.submit(
                AudioFrame(n, n * 10_000_000, (n + 1) * 10_000_000, bytes(960))
            )
        async with asyncio.timeout(3):
            while not revisions or not revisions[-1].is_final:
                await asyncio.sleep(0.005)
        assert revisions[-1].stable_text == f"{service._process.pid}:384000"
        assert revisions[-1].covered_through_ns == 4_000_000_000
        assert [revision.is_final for revision in revisions] == [False, True]
        assert settlements == list(range(400))
        async with worker.lease(service):
            result = await worker.run_owned(
                service.transcribe_buffer,
                audio_data=bytes(960),
                sample_rate=48_000,
                channels=1,
                sample_width=2,
            )
        assert result["text"] == f"rolling:{service._process.pid}:960"
        assert not failures
    finally:
        await native.close()
        await asyncio.to_thread(service.close)
        await worker.close()
    assert service.cleanup_outcome is AttemptCleanupOutcome.CLEAN
