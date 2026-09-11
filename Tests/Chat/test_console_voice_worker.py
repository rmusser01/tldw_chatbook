"""Owner-loop tests; blocking waits deliberately simulate a stalled UI."""

import asyncio
import threading

import pytest

from tldw_chatbook.Chat.console_voice_worker import ConsoleVoiceWorker, VoiceUiBridge
from tldw_chatbook.Chat.console_voice_controls import ControlKind
from tldw_chatbook.Audio.native_duplex_stream import AudioShutdownUnconfirmed


@pytest.mark.asyncio
async def test_worker_progresses_while_ui_is_blocked():
    worker = ConsoleVoiceWorker()
    ui_thread = threading.get_ident()
    await worker.run(lambda: asyncio.sleep(0))
    done = threading.Event()

    async def progress():
        assert threading.get_ident() != ui_thread
        done.set()

    task = asyncio.create_task(worker.run(progress))
    await asyncio.sleep(0)
    assert done.wait(1)
    await task
    await worker.aclose()
    with pytest.raises(RuntimeError):
        await worker.run(progress)


@pytest.mark.asyncio
async def test_unconfirmed_native_close_retires_facade_registration_and_state():
    from tldw_chatbook.Chat.console_voice_worker import WorkerVoiceSession

    worker = ConsoleVoiceWorker()
    bridge = VoiceUiBridge(asyncio.get_running_loop())

    class Core:
        state = "listening"

        async def enter(self, **kwargs):
            pass

        def fence_audio_admission(self):
            pass

        async def fence_and_close(self, reason):
            self.state = "audio_shutdown_unconfirmed"
            raise AudioShutdownUnconfirmed("audio_shutdown_unconfirmed")

    proxy = WorkerVoiceSession(worker, Core, bridge)
    await proxy.enter(capture_live=False)
    try:
        with pytest.raises(AudioShutdownUnconfirmed):
            await proxy.fence_and_close(ControlKind.TEARDOWN)
        assert proxy.state == "audio_shutdown_unconfirmed"
        assert (
            await worker.run(lambda: asyncio.sleep(0, result=len(worker._closers))) == 0
        )
    finally:
        # Ensure a failing assertion cannot leak this test's owner thread.
        await worker.run(lambda: asyncio.sleep(0, result=worker._closers.clear()))
        await worker.aclose()


@pytest.mark.asyncio
async def test_worker_shutdown_finishes_other_cleanup_after_native_uncertainty():
    worker = ConsoleVoiceWorker()
    seen = []

    async def uncertain():
        raise AudioShutdownUnconfirmed("audio_shutdown_unconfirmed")

    async def other():
        seen.append("other")

    async def install():
        worker.register_cleanup(uncertain)
        worker.register_cleanup(other)

    await worker.run(install)
    try:
        with pytest.raises(AudioShutdownUnconfirmed):
            await worker.aclose()
        assert seen == ["other"]
        assert not worker._thread.is_alive()
    finally:
        if worker._thread.is_alive():
            loop = worker._worker_loop
            loop.call_soon_threadsafe(loop.stop)
            await asyncio.to_thread(worker._thread.join, 2)


@pytest.mark.asyncio
async def test_facade_constructs_core_on_worker_and_fences_before_close():
    from tldw_chatbook.Chat.console_voice_worker import WorkerVoiceSession

    ui_thread = threading.get_ident()
    worker = ConsoleVoiceWorker()
    bridge = VoiceUiBridge(asyncio.get_running_loop())
    fenced = threading.Event()
    closed = threading.Event()

    class Core:
        state = "listening"

        def __init__(self):
            assert threading.get_ident() != ui_thread

        async def enter(self, **kwargs):
            assert kwargs == {"capture_live": False}

        def fence_audio_admission(self):
            fenced.set()

        async def fence_and_close(self, reason):
            assert threading.get_ident() != ui_thread
            assert reason == ControlKind.HANDS_FREE_EXIT
            closed.set()

    proxy = WorkerVoiceSession(worker, Core, bridge)
    await proxy.enter(capture_live=False)
    assert await proxy.observe(lambda core: core.state) == "listening"
    close = proxy.fence_and_close(ControlKind.HANDS_FREE_EXIT)
    assert fenced.is_set()
    assert bridge.closed
    await close
    assert closed.is_set()
    await worker.aclose()


@pytest.mark.asyncio
async def test_ui_calls_and_coalesced_projection_keep_owner_identity():
    worker = ConsoleVoiceWorker()
    ui_thread = threading.get_ident()
    bridge = VoiceUiBridge(asyncio.get_running_loop())
    seen = []

    async def work():
        assert await bridge.call(lambda: threading.get_ident()) == ui_thread
        for value in range(100):
            bridge.project(
                lambda value: seen.append((threading.get_ident(), value)), value
            )

    try:
        await worker.run(work)
        await asyncio.sleep(0)
        assert seen[-1] == (ui_thread, 99)
        bridge.fence()
        with pytest.raises(RuntimeError):
            await worker.run(lambda: bridge.call(lambda: None))
    finally:
        await worker.aclose()


@pytest.mark.asyncio
async def test_cancelled_observer_waits_for_real_owner_cleanup():
    worker = ConsoleVoiceWorker()
    bridge = VoiceUiBridge(asyncio.get_running_loop())
    entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleaned = threading.Event()

    async def on_ui():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await release_cleanup.wait()
            cleaned.set()

    task = asyncio.create_task(worker.run(lambda: bridge.call(on_ui)))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()
    assert not cleaned.is_set()
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()
    await worker.aclose()


@pytest.mark.asyncio
async def test_exit_cancels_pending_startup_without_late_capture():
    worker = ConsoleVoiceWorker()
    bridge = VoiceUiBridge(asyncio.get_running_loop())
    started = threading.Event()
    capture_opened = threading.Event()

    class Core:
        state = "idle"

        async def enter(self, **_):
            started.set()
            await asyncio.Event().wait()
            capture_opened.set()

        def fence_audio_admission(self):
            pass

        async def fence_and_close(self, _reason):
            pass

    from tldw_chatbook.Chat.console_voice_worker import WorkerVoiceSession

    proxy = WorkerVoiceSession(worker, Core, bridge)
    entering = asyncio.create_task(proxy.enter())
    while not started.is_set():
        await asyncio.sleep(0.001)
    await proxy.fence_and_close(ControlKind.HANDS_FREE_EXIT)
    await asyncio.sleep(0.01)
    try:
        assert entering.done(), "exit must cancel pending startup"
        assert not capture_opened.is_set()
    finally:
        entering.cancel()
        await asyncio.gather(entering, return_exceptions=True)
        await worker.aclose()


@pytest.mark.asyncio
async def test_ui_preparation_disposes_result_returned_after_cancel():
    worker = ConsoleVoiceWorker()
    bridge = VoiceUiBridge(asyncio.get_running_loop())
    started = asyncio.Event()
    discarded = []

    async def prepare():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return "late capability"

    task = asyncio.create_task(
        worker.run(lambda: bridge.prepare(prepare, on_discard=discarded.append))
    )
    try:
        async with asyncio.timeout(1):
            await started.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert discarded == ["late capability"]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await worker.aclose()


@pytest.mark.asyncio
async def test_diagnostic_backlog_is_bounded_and_never_writes_on_worker():
    worker = ConsoleVoiceWorker()
    bridge = VoiceUiBridge(asyncio.get_running_loop())
    ui_thread = threading.get_ident()
    emitted = threading.Event()
    records = []

    def persist(event, **fields):
        assert threading.get_ident() == ui_thread
        records.append((event, fields))

    async def burst():
        for _ in range(1000):
            bridge.diagnose(persist, "capture", {"status": "ok"})
        emitted.set()

    await worker.run(lambda: None)
    task = asyncio.create_task(worker.run(burst))
    await asyncio.sleep(0)
    try:
        assert emitted.wait(1)
        assert records == []
        await task
        await asyncio.sleep(0)
        assert 0 < len(records) <= 64
    finally:
        await asyncio.gather(task, return_exceptions=True)
        await worker.aclose()


@pytest.mark.asyncio
async def test_app_close_waits_for_orphan_then_closes_gateway_on_owner():
    worker = ConsoleVoiceWorker()
    release = threading.Event()
    settled = threading.Event()
    finalized = threading.Event()
    ui_thread = threading.get_ident()

    async def install():
        async def orphan():
            while not release.is_set():
                await asyncio.sleep(0.001)
            settled.set()

        async def finalize():
            assert settled.is_set()
            assert threading.get_ident() != ui_thread
            finalized.set()

        asyncio.create_task(orphan())
        worker.register_finalizer(finalize)

    await worker.run(install)
    closing = asyncio.create_task(worker.aclose())
    await asyncio.sleep(0.01)
    try:
        assert not closing.done(), "unfinished cleanup must keep its owner alive"
    finally:
        release.set()
        await closing
    assert finalized.is_set()
    assert not worker._thread.is_alive()


@pytest.mark.asyncio
async def test_worker_close_supports_python_311_executor_shutdown_signature(
    monkeypatch,
):
    worker = ConsoleVoiceWorker()
    await worker.run(lambda: None)
    original = worker._worker_loop.shutdown_default_executor

    async def shutdown_default_executor():
        await original()

    monkeypatch.setattr(
        worker._worker_loop, "shutdown_default_executor", shutdown_default_executor
    )
    await worker.aclose()
