"""Hardware-free model ownership and exact native chunk boundary regressions."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Audio.duplex_contracts import AudioFrame
from tldw_chatbook.Chat import console_speculative_voice_session as module


def frame(sequence):
    return AudioFrame(
        sequence,
        sequence * 10_000_000,
        (sequence + 1) * 10_000_000,
        sequence.to_bytes(2, "little") * 480,
    )


class Model:
    def __init__(self):
        self.entered = threading.Event()
        self.pushed = threading.Event()
        self.exited = threading.Event()
        self.audio = []
        self.contexts = 0

    def transcribe_stream(self, **kwargs):
        model = self

        class Context:
            result = SimpleNamespace(text="")

            def __enter__(self):
                model.contexts += 1
                model.entered.set()
                return self

            def add_pcm16(self, pcm):
                model.audio.append(pcm)
                self.result = SimpleNamespace(text=f"burst{model.contexts}")
                model.pushed.set()

            def __exit__(self, *_args):
                model.exited.set()

        return Context()


async def event(event):
    assert await asyncio.to_thread(event.wait, 2), "context event timed out"


def adapter(service, worker, candidate):
    return module._NativeStreamingStt(
        service,
        provider="parakeet-mlx",
        model="test-model",
        language="en",
        serial_worker=worker,
        prepared_candidate=candidate,
        quiet_seconds=0.01,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [399, 400, 401])
async def test_exact_chunk_quiet_exit_preserves_pcm_prefix_and_coverage(count):
    model = Model()
    worker = module._SerialSttWorker()
    native = adapter(None, worker, SimpleNamespace(model=model))
    revisions, failures, settlements = [], [], []
    native.start(revisions.append, failures.append, settlements.append)
    try:
        for i in range(count):
            native.submit(frame(i))
        await event(model.pushed)
        await event(model.exited)
        assert revisions[-1].covered_through_ns == count * 10_000_000
        assert revisions[-1].is_final
        assert sum(revision.is_final for revision in revisions) == 1
        model.exited.clear()
        native.submit(frame(count))
        await event(model.exited)
        assert model.contexts == 2
        assert revisions[-1].stable_text == "burst1 burst2"
        assert b"".join(model.audio) == b"".join(
            frame(i).pcm16 for i in range(count + 1)
        )
        assert settlements == list(range(count + 1))
        assert not failures
        await asyncio.wait_for(native._frames.join(), 1)
    finally:
        await native.close()
        await worker.close()


@pytest.mark.asyncio
async def test_shared_model_native_and_rolling_wait_for_context_restoration():
    from Tests.Audio.test_parakeet_voice_worker import fake_runtime
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetVoiceProcess

    service = ParakeetVoiceProcess(model="test-model", language="en")
    service._runtime_factory = fake_runtime
    worker = module._SerialSttWorker()
    candidate = await worker.run(service.create_streaming_transcriber)
    entered = threading.Event()
    request = service._request
    active_contexts = 0
    maximum_contexts = 0
    exit_count = 0
    both_exited = threading.Event()

    def observe_request(operation, *args, **kwargs):
        nonlocal active_contexts, maximum_contexts, exit_count
        if operation == "rolling":
            assert active_contexts == 0
        result = request(operation, *args, **kwargs)
        if operation == "open":
            active_contexts += 1
            maximum_contexts = max(maximum_contexts, active_contexts)
            entered.set()
        elif operation == "exit":
            active_contexts -= 1
            exit_count += 1
            if exit_count == 2:
                both_exited.set()
        return result

    service._request = observe_request
    first, second = (
        adapter(service, worker, candidate),
        adapter(service, worker, candidate),
    )
    first._quiet_seconds = 0.1
    seen, failures = [], []
    first.start(seen.append, failures.append, lambda _: None)
    second.start(seen.append, failures.append, lambda _: None)
    try:
        first.submit(frame(0))
        await event(entered)
        second.submit(frame(1))
        rolling = module._RollingWindowStt(
            service,
            provider="parakeet-mlx",
            model="test-model",
            language="en",
            serial_worker=worker,
        )
        result = await rolling.transcribe_window(
            pcm16=frame(2).pcm16, started_ns=0, ended_ns=10_000_000
        )
        await asyncio.wait_for(
            asyncio.gather(first._frames.join(), second._frames.join()), 3
        )
        assert result.tokens[0].text.startswith("rolling:")
        await event(both_exited)
        assert len(seen) == 2
        assert not failures
        assert active_contexts == 0
        assert maximum_contexts == 1
    finally:
        await first.close()
        await second.close()
        await asyncio.to_thread(service.close)
        await worker.close()


class GatedModel(Model):
    def __init__(self, phase):
        super().__init__()
        self.phase = phase
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.completed = threading.Event()
        self.exits = 0

    def gate(self, phase):
        if phase == self.phase:
            self.blocked.set()
            assert self.release.wait(3), "test did not release model operation"
            self.completed.set()

    def transcribe_stream(self, **kwargs):
        context = super().transcribe_stream(**kwargs)
        model = self

        class Context:
            def __enter__(self):
                model.gate("enter")
                context.__enter__()
                return self

            @property
            def result(self):
                return context.result

            def add_pcm16(self, pcm):
                model.gate("push")
                context.add_pcm16(pcm)

            def __exit__(self, *_args):
                model.gate("exit")
                model.exits += 1
                context.__exit__()

        return Context()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["enter", "push", "exit"])
@pytest.mark.parametrize("cancel_close", [False, True])
async def test_cancellation_retains_real_operation_and_checked_exit(
    phase, cancel_close
):
    model = GatedModel(phase)
    worker = module._SerialSttWorker()
    native = adapter(None, worker, SimpleNamespace(model=model))
    failures = []
    native.start(lambda _: None, failures.append, lambda _: None)
    native.submit(frame(0))
    try:
        await event(model.blocked)
        closing = (
            asyncio.create_task(native.close()) if cancel_close else native._worker
        )
        await asyncio.sleep(0)
        for _ in range(3):
            closing.cancel()
            await asyncio.sleep(0)
        assert worker._model_lock.locked()
        assert worker._receipts
        assert not native._worker.done()
        assert not model.exited.is_set()
        model.release.set()
        await asyncio.wait_for(
            asyncio.gather(native._worker, return_exceptions=True), 2
        )
        await asyncio.gather(closing, return_exceptions=True)
        assert model.completed.is_set()
        assert model.exits == 1
        assert not worker._model_lock.locked()
        assert not failures
        # Subsequent admission is proof of released ownership, not just a flag.
        async with worker.parakeet_context(None, Model().transcribe_stream):
            pass
    finally:
        model.release.set()
        await native.close()
        await worker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("arrival_wins", [False, True])
async def test_frame_at_full_chunk_quiet_boundary_is_owned_once(
    monkeypatch, arrival_wins
):
    model, worker = Model(), module._SerialSttWorker()
    native = adapter(None, worker, SimpleNamespace(model=model))
    revisions, settled = [], []
    original_wait = asyncio.wait_for
    injected = asyncio.Event()
    reads = 0

    async def boundary_wait(awaitable, timeout):
        nonlocal reads
        is_frame_get = (
            getattr(awaitable, "cr_frame", None) is not None
            and awaitable.cr_frame.f_locals.get("self") is native._frames
        )
        if is_frame_get:
            reads += 1
            if reads == 400:
                if arrival_wins:
                    native.submit(frame(400))
                    injected.set()
                else:
                    try:
                        return await original_wait(awaitable, 0)
                    except TimeoutError:
                        # Delivery occurs after the timed get was cancelled but
                        # before its observer sees expiry and closes the burst.
                        native.submit(frame(400))
                        injected.set()
                        raise
        return await original_wait(awaitable, timeout)

    monkeypatch.setattr(asyncio, "wait_for", boundary_wait)
    native.start(
        revisions.append, lambda _: pytest.fail("native failed"), settled.append
    )
    try:
        for i in range(400):
            native.submit(frame(i))
        await original_wait(injected.wait(), 2)
        await original_wait(native._frames.join(), 2)
        native._frames.put_nowait(None)
        await original_wait(native._worker, 2)
        await original_wait(native._frames.join(), 1)
        assert settled == list(range(401))
        assert b"".join(model.audio) == b"".join(frame(i).pcm16 for i in range(401))
        assert model.contexts == (1 if arrival_wins else 2)
        assert revisions[-1].covered_through_ns == 4_010_000_000
        assert revisions[-1].is_final
    finally:
        await native.close()
        await worker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("expiry", [False, True])
async def test_waiter_cancellation_or_timeout_does_not_touch_active_owner(
    monkeypatch, expiry
):
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetOwnershipTimeout
    from tldw_chatbook.Audio import voice_transcription

    worker = module._SerialSttWorker()
    model = Model()
    if expiry:
        monkeypatch.setattr(
            voice_transcription, "_PARAKEET_OWNERSHIP_TIMEOUT_SECONDS", 0.02
        )
    waiting = asyncio.Event()

    async def waiter():
        waiting.set()
        async with worker.parakeet_context(None, model.transcribe_stream):
            pytest.fail("cancelled waiter must never enter a context")

    try:
        async with worker.parakeet_context(None, model.transcribe_stream):
            task = asyncio.create_task(waiter())
            await waiting.wait()
            if not expiry:
                task.cancel()
            with pytest.raises(
                ParakeetOwnershipTimeout if expiry else asyncio.CancelledError
            ):
                await task
            assert worker._model_lock.locked()
            assert model.contexts == 1
            assert not model.exited.is_set()
        async with worker.parakeet_context(None, model.transcribe_stream):
            assert model.contexts == 2
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_cancel_after_acquisition_before_rpc_releases_once(monkeypatch):
    worker = module._SerialSttWorker()
    model = Model()
    sleep = asyncio.sleep
    owner = None

    async def cancel_at_checkpoint(delay, *args, **kwargs):
        if asyncio.current_task() is owner and delay == 0:
            owner.cancel()
        await sleep(delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", cancel_at_checkpoint)

    async def use():
        async with worker.parakeet_context(None, model.transcribe_stream):
            pytest.fail("no RPC may start after the cancelled lease checkpoint")

    try:
        owner = asyncio.create_task(use())
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert not worker._model_lock.locked()
        assert not worker._receipts
        assert model.contexts == 0
        assert not model.exited.is_set()
        async with worker.parakeet_context(None, model.transcribe_stream):
            pass
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_sentinel_after_exact_chunk_finalizes_cached_result_once():
    model, worker = Model(), module._SerialSttWorker()
    native = adapter(None, worker, SimpleNamespace(model=model))
    revisions, settled = [], []
    native.start(
        revisions.append, lambda _: pytest.fail("native failed"), settled.append
    )
    try:
        for i in range(400):
            native.submit(frame(i))
        await event(model.pushed)
        native._frames.put_nowait(None)
        await asyncio.wait_for(native._worker, 2)
        await asyncio.wait_for(native._frames.join(), 1)
        assert len(model.audio) == 1
        assert settled == list(range(400))
        assert sum(revision.is_final for revision in revisions) == 1
        assert revisions[-1].covered_through_ns == 4_000_000_000
    finally:
        await native.close()
        await worker.close()


@pytest.mark.asyncio
async def test_unavailable_native_service_does_not_attempt_rolling_fallback():
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetVoiceProcess
    from tldw_chatbook.Audio.rolling_transcript import (
        TranscriptBackendFailure,
        TranscriptEngine,
    )

    service = ParakeetVoiceProcess(model="test-model", language="en")
    service.close()
    worker = module._SerialSttWorker()

    class Fallback:
        async def transcribe_window(self, **kwargs):
            pytest.fail("closed service must not advertise a fallback")

        async def abort(self):
            pass

    engine = TranscriptEngine(
        turn_id="closed",
        live_adapter=adapter(service, worker, SimpleNamespace(model=service)),
        fallback_adapter=Fallback(),
    )
    try:
        engine.append_admitted_frame(frame(0))
        with pytest.raises(TranscriptBackendFailure):
            await asyncio.wait_for(engine.seal_through(0), 2)
        assert not engine._fallback_attempted
    finally:
        await engine.close()
        await worker.close()


@pytest.mark.asyncio
async def test_shutdown_interrupts_blocked_checked_exit_without_acquiring_lease():
    import multiprocessing
    from functools import partial
    from Tests.Audio.test_parakeet_voice_worker import gated_exit_runtime
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetVoiceProcess

    entered_exit = multiprocessing.get_context("spawn").Event()
    service = ParakeetVoiceProcess(model="test-model", language="en")
    service._runtime_factory = partial(gated_exit_runtime, entered_exit)
    worker = module._SerialSttWorker()
    candidate = await worker.run(service.create_streaming_transcriber)
    native = adapter(service, worker, candidate)
    native.start(lambda _: None, lambda _: None, lambda _: None)
    native.submit(frame(0))
    try:
        await event(entered_exit)
        native._worker.cancel()
        await asyncio.sleep(0)
        assert worker._model_lock.locked()
        assert not service.reaped
        await asyncio.wait_for(asyncio.to_thread(service.close), 4)
        await asyncio.wait_for(native.close(), 2)
        assert service.closed and service.reaped
        assert not worker._model_lock.locked()
    finally:
        await asyncio.to_thread(service.close)
        await native.close()
        await worker.close()


@pytest.mark.asyncio
async def test_failed_reap_keeps_service_fenced_and_lease_unavailable():
    from tldw_chatbook.Audio.parakeet_voice_worker import (
        ParakeetServiceUnavailable,
        ParakeetVoiceProcess,
    )

    class FailedReap(ParakeetVoiceProcess):
        def close(self):
            self._closed = True
            raise RuntimeError("parakeet_worker_not_reaped")

        @property
        def reaped(self):
            return False

    class BadContext:
        def __enter__(self):
            raise RuntimeError("uncertain entry")

    service = FailedReap(model=None, language="en")
    worker = module._SerialSttWorker()
    try:
        with pytest.raises(ParakeetServiceUnavailable):
            async with worker.parakeet_context(service, lambda **_: BadContext()):
                pytest.fail("entry must fail")
        assert worker._model_lock.locked()
        assert service.closed and not service.reaped
        with pytest.raises(ParakeetServiceUnavailable):
            service.transcribe_stream(context_size=(64, 64)).__exit__()
        with pytest.raises(ParakeetServiceUnavailable):
            async with worker.parakeet_context(service, Model().transcribe_stream):
                pytest.fail("failed reap must not permit another owner")
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_cancelled_prewarm_retains_full_context_lease():
    model = GatedModel("push")
    worker = module._SerialSttWorker()
    service = SimpleNamespace(
        create_streaming_transcriber=lambda **_: SimpleNamespace(model=model)
    )
    preparation = asyncio.create_task(
        module._prepare_streaming_candidate(
            service,
            provider="parakeet-mlx",
            model=None,
            language="en",
            serial_worker=worker,
        )
    )
    try:
        await event(model.blocked)
        for _ in range(3):
            preparation.cancel()
            await asyncio.sleep(0)
        assert worker._model_lock.locked()
        assert not model.exited.is_set()
        model.release.set()
        with pytest.raises(asyncio.CancelledError):
            await preparation
        assert model.exits == 1
        assert not worker._model_lock.locked()
    finally:
        model.release.set()
        await asyncio.gather(preparation, return_exceptions=True)
        await worker.close()


@pytest.mark.asyncio
async def test_recoverable_push_failure_allows_rolling_after_checked_exit():
    from Tests.Audio.test_parakeet_voice_worker import fake_runtime
    from tldw_chatbook.Audio.parakeet_voice_worker import ParakeetVoiceProcess
    from tldw_chatbook.Audio.rolling_transcript import TranscriptEngine

    service = ParakeetVoiceProcess(model="test-model", language="en")
    service._runtime_factory = fake_runtime
    worker = module._SerialSttWorker()
    candidate = await worker.run(service.create_streaming_transcriber)
    native = adapter(service, worker, candidate)
    native._push_parakeet_stream = lambda stream, _pcm, **_: stream.add_pcm16(b"fail")
    engine = TranscriptEngine(
        turn_id="recoverable",
        live_adapter=native,
        fallback_adapter=module._RollingWindowStt(
            service,
            provider="parakeet-mlx",
            model="test-model",
            language="en",
            serial_worker=worker,
        ),
        rolling_min_window_ns=0,
        rolling_debounce_seconds=0,
    )
    try:
        engine.append_admitted_frame(frame(0))
        revision = await asyncio.wait_for(engine.seal_through(0), 2)
        assert revision.mode == "rolling-window"
        assert "rolling:" in revision.stable_text + revision.revisable_text
        assert not service.closed
    finally:
        await engine.close()
        await asyncio.to_thread(service.close)
        await worker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["prewarm", "rolling"])
async def test_failed_reap_retains_plain_lease_for_prewarm_and_rolling(operation):
    from tldw_chatbook.Audio.parakeet_voice_worker import (
        ParakeetServiceUnavailable,
        ParakeetVoiceProcess,
    )

    cleanup_entered, release_cleanup = threading.Event(), threading.Event()

    class FailedReapService(ParakeetVoiceProcess):
        def create_streaming_transcriber(self, **_kwargs):
            return SimpleNamespace(model=self)

        def transcribe_stream(self, **_kwargs):
            service = self

            class Context:
                def __enter__(self):
                    return SimpleNamespace(add_pcm16=lambda _pcm: None)

                def __exit__(self, *_args):
                    service.close()

            return Context()

        def transcribe_buffer(self, **_kwargs):
            self.close()

        def close(self):
            self._closed = True
            cleanup_entered.set()
            assert release_cleanup.wait(2), "test did not release failed cleanup"
            raise RuntimeError("parakeet_worker_not_reaped")

    service = FailedReapService(model=None, language="en")
    # Preserve the production reaped predicate while simulating a child that
    # survives its bounded close attempt. No process/model is actually started.
    service._process = SimpleNamespace(is_alive=lambda: True)
    worker = module._SerialSttWorker()
    if operation == "prewarm":
        pending = module._prepare_streaming_candidate(
            service,
            provider="parakeet-mlx",
            model=None,
            language="en",
            serial_worker=worker,
        )
    else:
        rolling = module._RollingWindowStt(
            service,
            provider="parakeet-mlx",
            model=None,
            language="en",
            serial_worker=worker,
        )
        pending = rolling.transcribe_window(
            pcm16=frame(0).pcm16,
            started_ns=0,
            ended_ns=10_000_000,
        )
    task = asyncio.create_task(pending)
    try:
        await event(cleanup_entered)
        assert worker._model_lock.locked()
        release_cleanup.set()
        with pytest.raises(RuntimeError, match="parakeet_worker_not_reaped"):
            await task
        assert service.closed and not service.reaped
        assert worker._model_lock.locked()
        assert worker._poisoned
        with pytest.raises(ParakeetServiceUnavailable):
            async with worker.lease(service):
                pytest.fail("uncertain cleanup must not admit another model owner")
    finally:
        release_cleanup.set()
        await asyncio.gather(task, return_exceptions=True)
        await worker.close()
