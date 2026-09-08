"""Hardware-free tests of the actual native PortAudio callback boundary."""

import importlib
import os
import subprocess
import sys

import pytest

from Tests.Audio.fakes.native_duplex_helpers import (
    build_driver,
    driver_library,
    emit_callback,
    load_native,
)


@pytest.fixture
def native_bridge():
    return load_native().NativeDuplexBridge


@pytest.fixture(scope="module")
def driver_path(tmp_path_factory):
    return build_driver(tmp_path_factory.mktemp("native-driver"))


@pytest.fixture
def emit(driver_path):
    driver = driver_library(driver_path)
    return lambda bridge, **kwargs: emit_callback(driver, bridge, **kwargs)[0]


def test_native_duplex_capability_is_present():
    """A stale companion must not silently select a Python callback."""
    native = importlib.import_module("tldw_voice_aec")
    assert hasattr(native, "NativeDuplexBridge"), native.__file__
    assert native.DUPLEX_ABI_VERSION == 1
    if "TLDW_NATIVE_DUPLEX_ROOT" in os.environ:
        assert load_native() is native


def test_cancelled_submissions_do_not_skip_aec_reference_ordinals(native_bridge, emit):
    bridge = native_bridge(capture_capacity=64, render_capacity=64, generation=7)
    bridge.set_render_admission(True)
    assert bridge.queue_render(b"\x01\x00" * 480, 0, 0)
    assert emit(bridge) == b"\x01\x00" * 480
    first = bridge.pop_capture()
    assert (first["submission_id"], first["reference_sequence"]) == (0, 0)
    assert bridge.queue_render(bytes(960), 1, 0)
    assert bridge.queue_render(bytes(960), 2, 0)
    epoch = bridge.abort_output()
    assert bridge.queue_render(b"\x02\x00" * 480, 3, epoch)
    assert emit(bridge) == b"\x02\x00" * 480
    second = bridge.pop_capture()
    assert (second["submission_id"], second["reference_sequence"]) == (3, 1)
    assert second["output_pcm16"] == b"\x02\x00" * 480
    assert second["generation"] == 7


def test_native_progress_and_owned_records_while_gil_is_held(driver_path):
    # A process deadline makes a future accidental Python callback fail, not hang pytest.
    script = """
import struct
from Tests.Audio.fakes.native_duplex_helpers import load_native, driver_library
b = load_native().NativeDuplexBridge(generation=9)
b.set_render_admission(True)
for i in range(8):
    assert b.queue_render(struct.pack('<h', i + 20) * 480, i, 0)
driver = driver_library(DRIVER)
before = b.monotonic_ns()
assert driver.progress_with_gil_held(b.callback_address, b.userdata_address, 8) == 8
after = b.monotonic_ns()
assert b.snapshot()['callback_count'] == 8
for i in range(8):
    frame = b.pop_capture()
    assert frame['pcm16'] == struct.pack('<h', i + 1) * 480
    assert frame['output_pcm16'] == struct.pack('<h', i + 20) * 480
    assert frame['capture_sequence'] == i
    assert frame['reference_sequence'] == i
    assert frame['callback_ordinal'] == i + 1
    assert frame['input_adc_time'] == 10.0 + i * .01
    assert frame['output_dac_time'] == 10.04 + i * .01
    assert before <= frame['observed_ns'] <= after
assert b.pop_capture() is None
""".replace("DRIVER", repr(str(driver_path)))
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=10,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr


def test_ring_full_empty_and_fifo_wraparound(native_bridge, emit):
    b = native_bridge(capture_capacity=2, render_capacity=2)
    b.set_render_admission(True)
    assert b.pop_capture() is None
    assert b.latest_render() is None
    assert b.queue_render(b"\x01\x00" * 480, 0, 0)
    assert b.queue_render(b"\x02\x00" * 480, 1, 0)
    assert not b.queue_render(bytes(960), 2, 0)
    emit(b)
    assert b.pop_capture()["submission_id"] == 0
    assert b.queue_render(b"\x03\x00" * 480, 2, 0)
    emit(b)
    emit(b)
    assert [b.pop_capture()["submission_id"] for _ in range(2)] == [1, 2]
    emit(b)
    idle = b.pop_capture()
    assert idle["submission_id"] is None
    assert idle["reference_sequence"] is None
    assert idle["output_pcm16"] == bytes(960)
    assert b.latest_render()["submission_id"] == 2
    assert b.snapshot()["fatal_status_bits"] == 0


def test_capture_overflow_fences_output_and_latches_even_when_fault_record_dropped(
    native_bridge, emit
):
    b = native_bridge(capture_capacity=1)
    b.set_render_admission(True)
    emit(b)
    b.queue_render(b"\x04\x00" * 480, 0, 0)
    assert emit(b, status=2) == bytes(960)
    snap = b.snapshot()
    assert snap["capture_overflows"] == 1
    assert snap["fatal_status_bits"] == 2
    assert snap["startup_discarded"] == 0
    assert b.pop_capture()["capture_sequence"] == 0
    emit(b)
    record = b.pop_capture()
    assert record["capture_sequence"] == 2
    assert record["capture_overflows"] == 1
    assert record["fatal_status_bits"] == 2
    b.set_render_admission(True)
    assert not b.queue_render(bytes(960), 1, b.output_epoch)
    assert b.latest_render() is None


def test_startup_status_at_50_is_tolerated_but_51_latches(native_bridge, emit):
    b = native_bridge()
    b.set_render_admission(True)
    b.queue_render(b"\x05\x00" * 480, 0, 0)
    for _ in range(50):
        assert emit(b, status=1) == bytes(960)
    assert b.pop_capture() is None
    assert b.snapshot()["startup_discarded"] == 50
    assert b.snapshot()["fatal_status_bits"] == 0
    assert b.snapshot()["render_occupancy"] == 1
    assert emit(b, status=4) == bytes(960)
    record = b.pop_capture()
    assert record["capture_sequence"] == 0
    assert record["callback_ordinal"] == 51
    assert record["startup_discarded_before"] == 50
    assert record["fatal_status_bits"] == 4
    assert b.snapshot()["startup_discarded"] == 50
    emit(b)
    assert b.snapshot()["fatal_status_bits"] == 4


@pytest.mark.parametrize(
    "kwargs,counter",
    [
        ({"frames": 479}, "invalid_frames"),
        ({"frames": 481}, "invalid_frames"),
        ({"null_input": True}, "invalid_frames"),
        ({"null_output": True}, "invalid_frames"),
        ({"times": None}, "invalid_timing"),
        ({"times": (float("nan"), 1.0, 1.0)}, "invalid_timing"),
        ({"times": (1.0, float("inf"), 1.0)}, "invalid_timing"),
        ({"times": (1.0, 1.0, float("-inf"))}, "invalid_timing"),
    ],
)
def test_malformed_callback_latches_distinct_fault(
    native_bridge, emit, kwargs, counter
):
    b = native_bridge()
    b.set_render_admission(True)
    b.queue_render(b"\x01\x00" * 480, 0, 0)
    emit(b, **kwargs)
    assert b.snapshot()[counter] == 1
    assert b.latest_render() is None
    assert emit(b) == bytes(960)
    assert b.snapshot()[counter] == 1


def test_fences_preserve_actual_receipt_and_capture_time_playback_context(
    native_bridge, emit
):
    b = native_bridge(generation=4)
    b.set_render_admission(True)
    b.queue_render(b"\x06\x00" * 480, 0, 0)
    emit(b, times=(10.0, 10.02, 10.1))
    receipt = b.latest_render()
    assert receipt["output_dac_time"] == 10.1
    assert receipt["output_dac_end_time"] == pytest.approx(10.11)
    b.abort_output()
    b.set_render_admission(False)
    assert emit(b, times=(10.105, 10.12, 10.15)) == bytes(960)
    committed = b.pop_capture()
    buffered = b.pop_capture()
    assert committed["submission_id"] == 0
    assert buffered["submission_id"] is None
    assert buffered["playback_context_valid"]
    assert buffered["playback_start_dac_time"] == 10.1
    assert buffered["playback_end_dac_time"] == pytest.approx(10.11)
    assert b.latest_render() == receipt
    b.deactivate()
    b.set_render_admission(True)
    assert not b.queue_render(bytes(960), 1, b.output_epoch)
    assert emit(b, null_input=True) == bytes(960)
    assert b.pop_capture() is None
    assert not b.snapshot()["active"]
    assert not b.snapshot()["render_admission"]


def test_retained_native_storage_survives_python_owner_destruction(
    native_bridge, driver_path
):
    import gc
    from types import SimpleNamespace

    b = native_bridge()
    raw = SimpleNamespace(
        callback_address=b.callback_address, userdata_address=b.userdata_address
    )
    b.set_render_admission(True)
    assert b.queue_render(b"\x07\x00" * 480, 0, 0)
    b.retain_for_process_lifetime()
    b.retain_for_process_lifetime()
    b.set_render_admission(True)
    assert not b.snapshot()["active"]
    del b
    gc.collect()
    output, result = emit_callback(driver_library(driver_path), raw, null_input=True)
    assert output == bytes(960)
    assert result == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capture_capacity": 0},
        {"capture_capacity": 4097},
        {"render_capacity": 0},
        {"render_capacity": 4097},
        {"generation": -1},
    ],
)
def test_constructor_rejects_unbounded_or_negative_values(native_bridge, kwargs):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        native_bridge(**kwargs)


def test_render_validation_happens_before_publication(native_bridge):
    b = native_bridge()
    with pytest.raises(TypeError):
        native_bridge(64, 64, 0)
    b.set_render_admission(True)
    for pcm, submission, epoch in [
        (bytes(958), 0, 0),
        (bytearray(960), 0, 0),
        (bytes(960), -1, 0),
        (bytes(960), 0, -1),
    ]:
        with pytest.raises((TypeError, ValueError, OverflowError)):
            b.queue_render(pcm, submission, epoch)
    assert b.snapshot()["render_occupancy"] == 0
    assert not b.queue_render(bytes(960), 0, 1)
    assert b.queue_render(bytes(960), 0, 0)
    with pytest.raises(ValueError):
        b.queue_render(bytes(960), 0, 0)


def test_admission_reopen_cannot_resurrect_pre_fence_queue(native_bridge, emit):
    b = native_bridge()
    b.set_render_admission(True)
    assert b.queue_render(b"\x08\x00" * 480, 0, 0)
    b.set_render_admission(False)
    b.set_render_admission(True)
    assert emit(b) == bytes(960)
    assert b.pop_capture()["reference_sequence"] is None
    assert b.latest_render() is None
    assert b.queue_render(b"\x09\x00" * 480, 1, b.output_epoch)
    assert emit(b) == b"\x09\x00" * 480
    assert b.pop_capture()["reference_sequence"] == 0


def test_tolerated_startup_does_not_relabel_later_timing_fault(native_bridge, emit):
    b = native_bridge()
    emit(b, status=2)
    assert b.snapshot()["startup_discarded"] == 1
    assert b.snapshot()["fatal_status_bits"] == 0
    emit(b, times=(float("nan"), 10.02, 10.04))
    emit(b)
    record = b.pop_capture()
    assert record["invalid_timing"] == 1
    assert record["fatal_status_bits"] == 0
    assert record["startup_discarded_before"] == 1
    assert not record["playback_context_valid"]


def test_never_rendered_capture_has_known_empty_playback_context(native_bridge, emit):
    b = native_bridge()
    emit(b)
    record = b.pop_capture()
    assert record["playback_context_valid"]
    assert record["playback_start_dac_time"] is None
    assert record["playback_end_dac_time"] is None
    assert record["output_epoch"] is None
