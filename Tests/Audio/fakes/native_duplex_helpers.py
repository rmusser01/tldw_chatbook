"""Load the explicit local artifact and call its ABI without opening audio."""

import ctypes
import importlib
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType


class TimeInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in ("adc", "current", "dac")]


def install_fake_portaudio():
    """Install only the ABI wrapper seam; never load a PortAudio library."""
    from cffi import FFI

    ffi = FFI()
    ffi.cdef("typedef void PaStream; typedef int PaStreamCallback(void);")
    module = ModuleType("sounddevice")

    class Stream:
        def __init__(self, *, kind, wrap_callback, callback, userdata, **kwargs):
            assert kind == "duplex" and wrap_callback is None
            assert kwargs["samplerate"] == 48000 and kwargs["blocksize"] == 480
            self.callback, self.userdata = callback, userdata
            self._ptr = ffi.cast("PaStream *", 1234)
            self.latency = (0.01, 0.01)

        def start(self):
            pass

        def stop(self, *, ignore_errors):
            assert ignore_errors is False

        def close(self, *, ignore_errors):
            assert ignore_errors is False
            self._ptr = ffi.NULL

    module._ffi = ffi
    module._StreamBase = Stream
    module._exit_handler = lambda: None
    sys.modules["sounddevice"] = module
    return module


def load_native():
    root = Path(os.environ["TLDW_NATIVE_DUPLEX_ROOT"]).resolve()
    native = importlib.import_module("tldw_voice_aec")
    extension = importlib.import_module("tldw_voice_aec._native")
    assert Path(native.__file__).resolve().is_relative_to(root), native.__file__
    assert Path(extension.__file__).resolve().is_relative_to(root), extension.__file__
    assert native.DUPLEX_ABI_VERSION == 1
    return native


def build_driver(directory):
    source = Path(__file__).with_name("native_duplex_driver.cpp")
    library = Path(directory) / (
        "driver.dylib" if sys.platform == "darwin" else "driver.so"
    )
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-pthread",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return library


class CallbackObservation(ctypes.Structure):
    _fields_ = [
        ("callback_ns", ctypes.c_uint64),
        ("completed_ns", ctypes.c_uint64),
        ("output_sample", ctypes.c_int16),
    ]


def driver_library(path, *, release_gil=False):
    driver = (ctypes.CDLL if release_gil else ctypes.PyDLL)(str(path))
    driver.emit_callback.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(TimeInfo),
        ctypes.c_ulong,
    ]
    driver.emit_callback.restype = ctypes.c_int
    driver.progress_with_gil_held.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_ulong,
    ]
    driver.progress_with_gil_held.restype = ctypes.c_ulong
    driver.paced_progress_with_gil_held.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    driver.paced_progress_with_gil_held.restype = ctypes.c_ulong
    driver.observed_paced_progress.argtypes = [
        *driver.paced_progress_with_gil_held.argtypes,
        ctypes.POINTER(CallbackObservation),
    ]
    driver.observed_paced_progress.restype = ctypes.c_ulong
    driver.hold_gil_ms.argtypes = [ctypes.c_ulong, ctypes.POINTER(ctypes.c_uint64)]
    driver.hold_gil_ms.restype = None
    driver.driver_monotonic_ns.argtypes = []
    driver.driver_monotonic_ns.restype = ctypes.c_uint64
    driver.completed_observations.argtypes = [
        ctypes.POINTER(CallbackObservation),
        ctypes.c_ulong,
    ]
    driver.completed_observations.restype = ctypes.c_ulong
    return driver


def driver_clock_offset(driver):
    """Bracket the test driver's steady clock against Python's monotonic clock."""
    samples = []
    for _ in range(8):
        before = time.monotonic_ns()
        native = driver.driver_monotonic_ns()
        after = time.monotonic_ns()
        samples.append((after - before, (before + after) // 2 - native))
    driver.clock_uncertainty_ns = (min(samples)[0] + 1) // 2
    assert driver.clock_uncertainty_ns < 500_000
    return min(samples)[1]


def emit_callback(
    driver,
    bridge,
    *,
    pcm16=b"\x03\x00" * 480,
    frames=480,
    times=(10.0, 10.02, 10.04),
    status=0,
    null_input=False,
    null_output=False,
):
    incoming = ctypes.create_string_buffer(pcm16, max(len(pcm16), frames * 2))
    outgoing = ctypes.create_string_buffer(b"\x55" * (frames * 2))
    timing = None if times is None else TimeInfo(*times)
    result = driver.emit_callback(
        bridge.callback_address,
        bridge.userdata_address,
        None if null_input else incoming,
        None if null_output else outgoing,
        frames,
        None if timing is None else ctypes.byref(timing),
        status,
    )
    return outgoing.raw[: frames * 2], result
