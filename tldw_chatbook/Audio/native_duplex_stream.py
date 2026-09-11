"""Native callback ABI and checked, loop-independent PortAudio ownership."""

from __future__ import annotations

import asyncio
import atexit
from concurrent.futures import Future
from importlib import import_module
import inspect
import math
import threading
import time
from typing import Callable

_SHUTDOWN_SECONDS = 2.0


class NativeDuplexUnavailable(RuntimeError):
    """The installed native callback capability or clock cannot be used."""


class AudioShutdownUnconfirmed(RuntimeError):
    """Callbacks may still own native storage; replacement audio is quarantined."""


class NativeStreamRegistry:
    """App-root strong ownership; only checked close releases a native stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.owner: NativeDuplexStream | None = None
        self._shutdown = False
        self._installed = False

    def claim(self, owner: NativeDuplexStream) -> None:
        with self._lock:
            if self.owner is not None or self._shutdown:
                raise AudioShutdownUnconfirmed("audio shutdown unconfirmed")
            self.owner = owner

    def release(self, owner: NativeDuplexStream) -> None:
        with self._lock:
            if self.owner is owner:
                self.owner = None

    def install_shutdown_guard(self, sounddevice: object) -> None:
        with self._lock:
            if self._installed:
                return
            original = sounddevice._exit_handler
            atexit.unregister(original)
            atexit.register(self.shutdown, original)
            self._installed = True

    def shutdown(self, original: Callable[[], None]) -> None:
        """Do not terminate PortAudio concurrently with uncertain native work."""
        with self._lock:
            self._shutdown = True
            owner = self.owner
            if owner is not None:
                owner.bridge.deactivate()
                owner.bridge.retain_for_process_lifetime()
                return
        original()


# Module lifetime is application lifetime, including replacement Console sessions.
NATIVE_STREAM_REGISTRY = NativeStreamRegistry()


class NativeDuplexStream:
    """One daemon owns open/start/stop/close; observers never own native lifetime.

    The callback, userdata, loaded module and original stream handle stay strongly
    held until checked close success. Futures are polling receipts only: they do
    not schedule callbacks on the voice loop, including after that loop closes.
    """

    def __init__(
        self,
        *,
        generation: int,
        capture_capacity: int = 64,
        render_capacity: int = 64,
        device_pair: tuple[int, int] | None = None,
        registry: NativeStreamRegistry | None = None,
        native: object | None = None,
        sounddevice: object | None = None,
        clock: Callable[[], float] = time.monotonic,
        transport_clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        try:
            self.native = native or import_module("tldw_voice_aec")
            if self.native.DUPLEX_ABI_VERSION != 1:
                raise ValueError("unsupported ABI")
            self.bridge = self.native.NativeDuplexBridge(
                generation=generation,
                capture_capacity=capture_capacity,
                render_capacity=render_capacity,
            )
            self.sounddevice = sounddevice or import_module("sounddevice")
            parameters = inspect.signature(self.sounddevice._StreamBase).parameters
            if (
                not {"kind", "wrap_callback", "callback", "userdata"}
                <= parameters.keys()
            ):
                raise ValueError("unsupported callback seam")
            ffi = self.sounddevice._ffi
            self.callback = ffi.cast("PaStreamCallback *", self.bridge.callback_address)
            self.userdata = ffi.cast("void *", self.bridge.userdata_address)
        except (
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as error:
            raise NativeDuplexUnavailable(
                "native duplex unavailable; rebuild the native audio component"
            ) from error
        self.registry = registry if registry is not None else NATIVE_STREAM_REGISTRY
        self._clock = clock
        self._transport_clock = transport_clock
        self._device_pair = device_pair
        self._closing = threading.Event()
        self._close_lock = threading.Lock()
        self._deadline: float | None = None
        self._started: Future[None] = Future()
        self._closed: Future[bool] = Future()
        self.raw_stream = None
        self.native_handle = None
        self.latency: tuple[float, float] | None = None
        self.native_to_python_offset_ns = 0
        self.stop_failed = False
        self._thread: threading.Thread | None = None
        self._dispatch_failed = False
        self._native_execution_started = False

    def begin(self) -> None:
        """Claim application ownership before dispatching the sole native worker."""
        self.registry.claim(self)
        try:
            if self.registry is NATIVE_STREAM_REGISTRY:
                self.registry.install_shutdown_guard(self.sounddevice)
            self._thread = threading.Thread(
                target=self._run, name="native-duplex-owner", daemon=True
            )
            self._thread.start()
        except BaseException:
            with self._close_lock:
                self._dispatch_failed = True
                unopened = not self._native_execution_started
            self.request_close()
            if unopened:
                # The worker entry gate also excludes a delayed thread. Neither
                # a missing Thread nor is_alive() alone proves this ownership.
                self.registry.release(self)
                self._closed.set_result(True)
            raise

    def _calibrate(self) -> None:
        samples = []
        for _ in range(8):
            before = self._transport_clock()
            native = self.bridge.monotonic_ns()
            after = self._transport_clock()
            if after >= before:
                samples.append((after - before, (before + after) // 2 - native))
        if not samples or min(samples)[0] > 1_000_000:
            raise NativeDuplexUnavailable("native duplex clock calibration unavailable")
        self.native_to_python_offset_ns = min(samples)[1]

    def _run(self) -> None:
        with self._close_lock:
            if self._dispatch_failed:
                return
            self._native_execution_started = True
        try:
            self._calibrate()
            # Retain a partially initialized wrapper too: stream-info failure can
            # occur after Pa_OpenStream has already returned a valid handle.
            self.raw_stream = self.sounddevice._StreamBase.__new__(
                self.sounddevice._StreamBase
            )
            self.raw_stream.__init__(
                kind="duplex",
                wrap_callback=None,
                callback=self.callback,
                userdata=self.userdata,
                samplerate=48000,
                channels=1,
                dtype="int16",
                blocksize=480,
                latency="low",
                device=self._device_pair,
            )
            self.native_handle = self.raw_stream._ptr
            value = self.raw_stream.latency
            if (
                not isinstance(value, (tuple, list))
                or len(value) != 2
                or any(
                    isinstance(item, bool)
                    or not math.isfinite(float(item))
                    or float(item) < 0
                    for item in value
                )
            ):
                raise ValueError("duplex stream reported invalid actual latency")
            self.latency = (float(value[0]), float(value[1]))
            if not self._closing.is_set():
                self.start()
            self._started.set_result(None)
        except Exception as error:
            # Remove worker tracebacks, which otherwise create a lifetime cycle.
            self._started.set_exception(error.with_traceback(None))
            self.request_close()
        self._closing.wait()
        confirmed = False
        try:
            if self.raw_stream is None or not hasattr(self.raw_stream, "_ptr"):
                confirmed = True  # No native open was attempted.
            else:
                ptr = self.raw_stream._ptr
                ffi = self.sounddevice._ffi
                if ffi.typeof(ptr) == ffi.typeof("PaStream **"):
                    # Pa_OpenStream failed before the wrapper dereferenced it.
                    ptr = ptr[0]
                    self.raw_stream._ptr = ptr
                self.native_handle = ptr
                if ptr == ffi.NULL:
                    confirmed = True  # Failed open produced no native handle.
                else:
                    try:
                        self.stop()
                    except Exception:
                        self.stop_failed = True
                    self.close()
                    confirmed = True
        except Exception:
            pass  # Checked close failure retains the original handle and owner.
        if confirmed:
            self.registry.release(self)
        self._closed.set_result(confirmed)

    def start(self) -> None:
        """Checked start, called exclusively by the native operation owner."""
        self.raw_stream.start()

    def stop(self) -> None:
        """Checked stop; close cannot run until this native call returns."""
        self.raw_stream.stop(ignore_errors=False)

    def close(self) -> None:
        """Checked close; a cleared wrapper pointer is not a success receipt."""
        self.raw_stream.close(ignore_errors=False)

    def request_close(self) -> None:
        """Deactivate synchronously and set the one absolute observer deadline."""
        self.bridge.deactivate()
        with self._close_lock:
            if self._deadline is None:
                self._deadline = self._clock() + _SHUTDOWN_SECONDS
            self._closing.set()

    async def wait_started(self) -> None:
        while not self._started.done():
            if self._closing.is_set():
                await self.wait_closed()
                raise NativeDuplexUnavailable("native duplex startup cancelled")
            await asyncio.sleep(0.001)
        self._started.result()

    async def wait_closed(self) -> None:
        self.request_close()
        while not self._closed.done():
            if self._clock() >= self._deadline:
                raise AudioShutdownUnconfirmed("audio shutdown unconfirmed")
            await asyncio.sleep(0.001)
        if not self._closed.result():
            raise AudioShutdownUnconfirmed("audio shutdown unconfirmed")
