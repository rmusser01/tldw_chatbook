"""Finite native-producer controls; synthetic PCM, no hardware or network."""

import asyncio
import ctypes
import json
import os
import time

import pytest

from Tests.Audio import test_native_duplex_stream as native_helpers
from Tests.Audio.fakes.native_duplex_helpers import (
    CallbackObservation,
    build_driver,
    driver_library,
    driver_clock_offset,
)
from Tests.UI.test_console_voice_native_callback import _native_core
from tldw_chatbook.Audio.voice_preprocessor import VoicePreprocessor

portaudio = native_helpers.portaudio
native_transport = native_helpers.native_transport


@pytest.mark.asyncio
@pytest.mark.parametrize("retired_first", ["parent", "child"])
async def test_real_pipes_keep_draft_credit_identity_across_asymmetric_retirement(
    retired_first,
):
    """A consumed old draft may retire independently in the two interpreters."""
    from Tests.UI.test_console_voice_process_continuity import eventually
    from tldw_chatbook.Audio.voice_process_lifetime import LifecyclePipe

    child_read, parent_write = os.pipe()
    parent_read, child_write = os.pipe()
    received, faults = [], []

    def consume(record):
        if record.header["op"] == "draft":
            received.append(record)
            return True  # Original receiver retains real consumer custody.

    parent = LifecyclePipe(
        parent_read,
        parent_write,
        generation=1,
        request_id="a" * 32,
        parent=True,
        consume=consume,
        fault=faults.append,
    )
    child = LifecyclePipe(
        child_read,
        child_write,
        generation=1,
        request_id="a" * 32,
        parent=False,
        consume=lambda _: None,
        fault=faults.append,
    )
    try:
        child.send("draft", turn_id="old", revision=1, epoch=1, payload=b"old")
        await eventually(lambda: len(received) == 1)
        assert child._windows["draft_0"].outstanding == (1, 3)
        parent.release(received.pop())
        await eventually(lambda: child._windows["draft_0"].outstanding == (0, 0))
        (parent if retired_first == "parent" else child).retire_turn("old")
        # Prepare may establish the receiver's local turn slot before its draft.
        child.send("prepare", turn_id="new", revision=1, epoch=2, payload=b"new")
        await eventually(lambda: "new" in parent.input._turns)
        child.send("draft", turn_id="new", revision=1, epoch=2, payload=b"new")
        await eventually(lambda: received or faults)
        assert not faults, [error.code for error in faults]
        window = next(
            window for window in child._windows.values() if window.key.turn_id == "new"
        )
        assert window.outstanding == (1, 3)
        parent.release(received.pop())
        await eventually(lambda: faults or window.outstanding == (0, 0))
        assert not faults, [error.code for error in faults]
        assert window.outstanding == (0, 0)
    finally:
        parent.stop()
        child.stop()
        os.close(parent_write)
        os.close(child_write)
        assert await parent.join()
        assert await child.join()


@pytest.mark.asyncio
async def test_shared_interpreter_control_loses_dsp_progress(
    monkeypatch, native_transport, tmp_path
):
    """Same interpreter cannot drain the native rings during a 720 ms hold."""
    library = build_driver(tmp_path)
    producer_driver = driver_library(library, release_gil=True)
    holder = driver_library(library)
    clock_offset = driver_clock_offset(holder)
    clock_uncertainty = holder.clock_uncertainty_ns
    observations = (CallbackObservation * 100)()
    first = ctypes.c_uint64()
    bounds = (ctypes.c_uint64 * 2)()
    acknowledgements = []
    original = VoicePreprocessor.process_capture

    async def process(self, *args, **kwargs):
        await original(self, *args, **kwargs)
        acknowledgements.append(time.monotonic_ns())

    monkeypatch.setattr(VoicePreprocessor, "process_capture", process)
    async with _native_core(native_transport):
        bridge = native_transport._stream.bridge
        producer = asyncio.create_task(
            asyncio.to_thread(
                producer_driver.observed_paced_progress,
                bridge.callback_address,
                bridge.userdata_address,
                100,
                ctypes.byref(first),
                observations,
            )
        )
        async with asyncio.timeout(3):
            while len(acknowledgements) < 3:
                await asyncio.sleep(0.001)
        holder.hold_gil_ms(720, bounds)
        bounds = [stamp + clock_offset for stamp in bounds]
        counters = bridge.snapshot()
        await producer
        drift = abs(driver_clock_offset(holder) - clock_offset)
        calibration_bound = clock_uncertainty + holder.clock_uncertainty_ns + drift
        margin = 2_000_000
        assert drift < 1_000_000 and calibration_bound < margin
        captures = sum(
            bounds[0] + margin < item.callback_ns + clock_offset < bounds[1] - margin
            for item in observations
        )
        dsp = sum(
            bounds[0] + margin < stamp < bounds[1] - margin
            for stamp in acknowledgements
        )
        print(
            "SHARED_GIL_CONTROL "
            + json.dumps(
                dict(
                    captures=captures,
                    dsp=dsp,
                    native=counters,
                    hold_ns=bounds[1] - bounds[0],
                    calibration_bound_ns=calibration_bound,
                    offset_ns=clock_offset,
                )
            )
        )
        assert captures > 0
        if os.environ.get("TLDW_TASK9_EXPECT_SHARED_CONTINUITY") == "1":
            assert dsp > 0, "shared interpreter cannot acknowledge DSP during GIL hold"
        else:
            assert dsp == 0
            assert counters["capture_overflows"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["terminal_first", "speech_first"])
async def test_native_terminal_preroll_orders_and_delayed_parent_promotion(
    monkeypatch, tmp_path, order
):
    from Tests.UI.test_console_voice_process_continuity import (
        mounted_native_process,
        eventually,
    )

    async with mounted_native_process(monkeypatch, tmp_path, interruptions=0) as h:
        h.promotion_gate.clear()
        await h.control.command("terminal_order", order=order)
        await h.control.command("produce", count=450)
        await h.control.wait(lambda result: len(result["dsp"]) >= 5)
        original = await h.control.command("speak")
        evidence = await h.control.wait(
            lambda result: result["boundary_speech"] is not None
        )
        boundary = evidence["boundary_speech"]
        assert boundary["order"] == order
        assert boundary["preroll_frames"] == 24
        assert boundary["original_turn_id"] == original["turn_id"]
        assert boundary["turn_id"] == original["turn_id"]
        assert boundary["pending_next_turn_id"] not in {None, original["turn_id"]}
        start = boundary["boundary_ns"]
        assert boundary["frames"] == [
            [start, start + 10_000_000, start + 10_000_000],
            [start + 10_000_000, start + 20_000_000, None],
        ]
        await eventually(lambda: h.promotion_requests)
        assert not h.promotions
        held = await h.control.wait(lambda result: result["terminal_receipts"])
        assert held["terminal_receipts"] == [
            ["terminal_claim", original["turn_id"], None]
        ]
        assert held["retained_attempt_epochs"]
        before = len(evidence["dsp"])
        held_later = await h.control.wait(
            lambda result: len(result["dsp"]) >= before + 10
        )
        assert held_later["terminal_receipts"] == held["terminal_receipts"]
        assert not h.promotions  # The original parent owner is still held.
        h.promotion_gate.set()
        await eventually(lambda: len(h.promotions) == 2)
        assert all(outcome.status.value == "promoted" for _, outcome in h.promotions)
        completed = await h.control.wait(
            lambda result: (
                result["turn_id"] is None and not result["retained_attempt_epochs"]
            )
        )
        assert completed["terminal_receipts"] == [
            ["terminal_claim", original["turn_id"], None],
            ["terminal_result", original["turn_id"], "promoted"],
            ["terminal_claim", boundary["pending_next_turn_id"], None],
            ["terminal_result", boundary["pending_next_turn_id"], "promoted"],
        ]
        final = await h.control.command("finish")
        assert final["turn_id"] is None
        assert (
            final["capture_overflows"]
            == final["render_overflows"]
            == final["reference_overflows"]
            == 0
        )
        assert final["discontinuities"] == final["drift_faults"] == 0
        assert h.owner._failure is None
