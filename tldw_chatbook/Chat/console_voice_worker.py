"""One voice loop and explicit, lifetime-fenced UI handoffs."""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import Future
import inspect
import threading
from typing import Any, Callable

from tldw_chatbook.Audio.native_duplex_stream import AudioShutdownUnconfirmed


class _OwnerCall:
    """Keep owner completion independent of a cancelled caller observation."""

    def __init__(self, loop: asyncio.AbstractEventLoop, factory: Callable):
        if loop.is_closed():
            raise RuntimeError("voice_owner_loop_closed")
        self._loop = loop
        self._factory = factory
        self.receipt: Future = Future()
        self._cancelled = threading.Event()
        self._task: asyncio.Task | None = None
        self._cancel_sent = False
        loop.call_soon_threadsafe(self._start)

    def _start(self) -> None:
        async def invoke():
            if self._cancelled.is_set():
                raise asyncio.CancelledError
            result = self._factory()
            return await result if inspect.isawaitable(result) else result

        self._task = asyncio.create_task(invoke())
        self._task.add_done_callback(self._settled)

    def _settled(self, task: asyncio.Task) -> None:
        self._factory = lambda: None
        try:
            result = task.result()
        except BaseException as exc:
            self.receipt.set_exception(exc)
        else:
            self.receipt.set_result(result)
        self._task = None

    def cancel(self) -> None:
        self._cancelled.set()
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._cancel_on_owner)

    def _cancel_on_owner(self) -> None:
        if self._task is not None and not self._cancel_sent:
            self._cancel_sent = True
            self._task.cancel()

    async def wait(self) -> Any:
        observed = asyncio.wrap_future(self.receipt)
        try:
            return await asyncio.shield(observed)
        except asyncio.CancelledError:
            if not self.receipt.done():
                self.cancel()
                # A second cancellation must not turn unfinished owner cleanup
                # into a completed receipt. No blocking waits on either loop.
                while not observed.done():
                    try:
                        await asyncio.shield(observed)
                    except asyncio.CancelledError:
                        continue
                    except BaseException:
                        break
            if observed.done() and not observed.cancelled():
                observed.exception()
            raise


class ConsoleVoiceWorker:
    """Lazy app-owned loop; view sessions register their own resource cleanup."""

    def __init__(self) -> None:
        self._ready: Future = Future()
        self._thread: threading.Thread | None = None
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._closing = False
        self._close_task: asyncio.Task | None = None
        self._closers: list[Callable] = []  # accessed only on the worker loop
        self._finalizers: list[Callable] = []

    async def _loop(self) -> asyncio.AbstractEventLoop:
        if self._worker_loop is not None:
            return self._worker_loop
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._drive, name="console-voice", daemon=True
            )
            self._thread.start()
        return await asyncio.shield(asyncio.wrap_future(self._ready))

    def _drive(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except BaseException as exc:
            self._ready.set_exception(exc)
            return
        self._worker_loop = loop
        self._ready.set_result(loop)
        try:
            loop.run_forever()
        finally:
            loop.close()

    async def run(self, factory: Callable) -> Any:
        """Invoke a factory on the voice loop and await actual completion."""
        if self._closing:
            raise RuntimeError("voice_worker_closed")
        loop = await self._loop()
        if self._closing:
            raise RuntimeError("voice_worker_closed")
        return await _OwnerCall(loop, factory).wait()

    def register_cleanup(self, callback: Callable) -> None:
        """Register on the worker loop; resources must be closed before stop."""
        self._closers.append(callback)

    def unregister_cleanup(self, callback: Callable) -> None:
        if callback in self._closers:
            self._closers.remove(callback)

    def register_finalizer(self, callback: Callable) -> None:
        """Register app-owned resources, closed after every session settles."""
        if callback not in self._finalizers:
            self._finalizers.append(callback)

    async def aclose(self) -> None:
        """Close resources, then stop/join off the UI; never close a live loop."""
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        if self._thread is None:
            return
        loop = await self._loop()

        async def clean():
            shutdown_error = None
            for close in tuple(self._closers):
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except AudioShutdownUnconfirmed as error:
                    shutdown_error = error
            self._closers.clear()
            await asyncio.sleep(0)
            pending = [
                t
                for t in asyncio.all_tasks()
                if t is not asyncio.current_task() and not t.done()
            ]
            if pending:
                # Orphans remain supervised on this loop after view close.
                # The runtime's quit budget reports a pending close; it must
                # not stop their owner or claim cleanup finished prematurely.
                await asyncio.shield(asyncio.gather(*pending, return_exceptions=True))
            for finalize in self._finalizers:
                await finalize()
            self._finalizers.clear()
            await loop.shutdown_asyncgens()
            # Python 3.11 has no timeout keyword; the driver join below is
            # bounded after resource cleanup, independently of this receipt.
            await loop.shutdown_default_executor()
            return shutdown_error

        shutdown_error = await _OwnerCall(loop, clean).wait()
        loop.call_soon_threadsafe(loop.stop)
        await asyncio.to_thread(self._thread.join, 2.0)
        if self._thread.is_alive():
            raise RuntimeError("voice_worker_join_timeout")
        if shutdown_error is not None:
            raise shutdown_error


class WorkerVoiceSession:
    """UI facade; the core and its asynchronous resources belong to voice."""

    def __init__(
        self, worker: ConsoleVoiceWorker, build_core: Callable, ui_bridge: VoiceUiBridge
    ):
        self._worker = worker
        self._build_core = build_core
        self._bridge = ui_bridge
        self._core = None
        self._lock = threading.RLock()
        self._state = None
        self._publisher = None
        self._close_task = None
        self._enter_task = None
        self._core_close_task = None

    @property
    def state(self):
        with self._lock:
            return self._state

    async def enter(self, *, capture_live: bool = False) -> None:
        async def start():
            if self._bridge.closed:
                raise RuntimeError("voice_ui_generation_closed")
            core = self._build_core()
            with self._lock:
                self._core = core
                self._build_core = None
                if self._bridge.closed:
                    core.fence_audio_admission()
            self._worker.register_cleanup(self._shutdown)
            try:
                if self._bridge.closed:
                    raise RuntimeError("voice_ui_generation_closed")
                self._enter_task = asyncio.create_task(
                    core.enter(capture_live=capture_live)
                )
                await self._enter_task
                if self._bridge.closed:
                    raise RuntimeError("voice_ui_generation_closed")
                self._publisher = asyncio.create_task(self._publish_state())
            except BaseException:
                await self._shutdown()
                raise

        await self._worker.run(start)

    async def _publish_state(self):
        while True:
            with self._lock:
                self._state = self._core.state
            await asyncio.sleep(0.05)

    async def observe(self, callback: Callable) -> Any:
        """Read a diagnostic snapshot on the core's owner loop."""
        return await self._worker.run(lambda: callback(self._core))

    def fence_and_close(self, reason):
        self._bridge.fence()
        with self._lock:
            if self._core is not None:
                self._core.fence_audio_admission()
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._worker.run(lambda: self._close_core(reason))
            )
        return asyncio.shield(self._close_task)

    async def _shutdown(self):
        from tldw_chatbook.Chat.console_voice_controls import ControlKind

        self._bridge.fence()
        await self._close_core(ControlKind.TEARDOWN)

    async def _close_core(self, reason):
        if self._core_close_task is None:
            self._core_close_task = asyncio.create_task(self._close_resources(reason))
        await asyncio.shield(self._core_close_task)

    async def _close_resources(self, reason):
        if self._enter_task is not None and not self._enter_task.done():
            self._enter_task.cancel()
            await asyncio.gather(self._enter_task, return_exceptions=True)
        if self._publisher is not None:
            self._publisher.cancel()
            await asyncio.gather(self._publisher, return_exceptions=True)
            self._publisher = None
        shutdown_error = None
        if self._core is not None:
            try:
                await self._core.fence_and_close(reason)
            except AudioShutdownUnconfirmed as error:
                shutdown_error = error
            with self._lock:
                self._state = self._core.state
        self._worker.unregister_cleanup(self._shutdown)
        self._build_core = None
        if shutdown_error is not None:
            raise shutdown_error


class VoiceUiBridge:
    """Bound UI authority requests and coalesce disposable projections."""

    def __init__(self, owner_loop: asyncio.AbstractEventLoop) -> None:
        self._loop = owner_loop
        self._lock = threading.RLock()
        self._closed = False
        self._calls: set[_OwnerCall] = set()
        self._preparing = 0
        self._terminal = False
        self._projection: tuple[Callable, object] | None = None
        self._projection_queued = False
        self._diagnostics = deque(maxlen=64)
        self._diagnostics_queued = False

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    async def call(
        self, factory: Callable, *, on_discard: Callable | None = None
    ) -> Any:
        """Call UI authority with checks before dispatch and after completion."""

        def guarded():
            if self.closed:
                raise RuntimeError("voice_ui_generation_closed")
            return factory()

        with self._lock:
            if self._closed:
                raise RuntimeError("voice_ui_generation_closed")
            call = _OwnerCall(self._loop, guarded)
            self._calls.add(call)
        try:
            result = await call.wait()
            if self.closed:
                raise RuntimeError("voice_ui_generation_closed")
            return result
        except BaseException:
            # The receipt is settled even when cancellation was resisted.
            # Only explicitly thread-safe disposal belongs on this caller loop.
            if on_discard is not None and call.receipt.done():
                try:
                    abandoned = call.receipt.result()
                except BaseException:
                    pass
                else:
                    on_discard(abandoned)
            raise
        finally:
            with self._lock:
                self._calls.discard(call)

    async def prepare(
        self, factory: Callable, *, on_discard: Callable | None = None
    ) -> Any:
        with self._lock:
            if self._preparing >= 2:
                raise RuntimeError("voice_ui_preparation_capacity")
            self._preparing += 1
        try:
            return await self.call(factory, on_discard=on_discard)
        finally:
            with self._lock:
                self._preparing -= 1

    async def terminal(self, factory: Callable) -> Any:
        with self._lock:
            if self._terminal:
                raise RuntimeError("voice_ui_terminal_capacity")
            self._terminal = True
        try:
            return await self.call(factory)
        finally:
            with self._lock:
                self._terminal = False

    def project(self, callback: Callable, value: object) -> None:
        with self._lock:
            if self._closed:
                return
            self._projection = (callback, value)
            if self._projection_queued:
                return
            self._projection_queued = True
        self._loop.call_soon_threadsafe(self._deliver_projection)

    def diagnose(self, callback: Callable, event: str, fields: dict) -> None:
        """Queue discardable content-free diagnostics separately from controls."""
        with self._lock:
            if self._closed:
                return
            self._diagnostics.append((callback, event, dict(fields)))
            if self._diagnostics_queued:
                return
            self._diagnostics_queued = True
        self._loop.call_soon_threadsafe(self._deliver_diagnostics)

    def _deliver_diagnostics(self) -> None:
        with self._lock:
            records = tuple(self._diagnostics)
            self._diagnostics.clear()
            self._diagnostics_queued = False
        for callback, event, fields in records:
            try:
                callback(event, **fields)
            except Exception:
                pass

    def _deliver_projection(self) -> None:
        with self._lock:
            item, self._projection = self._projection, None
            self._projection_queued = False
            closed = self._closed
        if item is not None and not closed:
            item[0](item[1])

    def fence(self) -> None:
        with self._lock:
            self._closed = True
            self._projection = None
            self._diagnostics.clear()
            calls = tuple(self._calls)
        for call in calls:
            call.cancel()
