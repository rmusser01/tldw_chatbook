"""Windows measurement packets and their device-frame timeline."""

from dataclasses import dataclass
from itertools import pairwise
import math
from queue import Queue
import threading
import time

import numpy as np

RATE = 48000
DISCONTINUITY = 1
SILENT = 2
TIMESTAMP_ERROR = 4
BUFFER_EMPTY = 0x08890001
MAX_POLL_GAP_S = 0.15
MAX_PACKET_CLOCK_ERROR_S = 0.005
SOUND_RMS = 0.0002


@dataclass(frozen=True)
class Packet:
    frame: int
    start_s: float
    flags: int
    data: np.ndarray

    @property
    def end_s(self) -> float:
        return self.start_s + len(self.data) / RATE


def content_span(packets: list[Packet], metrics: dict):
    signal = [index for index, packet in enumerate(packets) if np.any(packet.data)]
    if not signal:
        metrics["timeline_stage"] = "no_signal_packets"
        return None
    span = packets[signal[0] : signal[-1] + 1]
    first = span[0]
    max_clock_error = 0.0
    next_frame = first.frame
    for packet in span:
        error = packet.start_s - first.start_s - (packet.frame - first.frame) / RATE
        if (
            not math.isfinite(packet.start_s)
            or packet.start_s <= 0
            or packet.flags & TIMESTAMP_ERROR
        ):
            metrics["timeline_stage"] = "packet_timestamp_error"
            return None
        if packet.flags & DISCONTINUITY:
            metrics["timeline_stage"] = "packet_discontinuity"
            return None
        if packet.frame != next_frame:
            metrics.update(
                timeline_stage="device_frame_gap", frame_gap=packet.frame - next_frame
            )
            return None
        max_clock_error = max(max_clock_error, abs(error))
        if max_clock_error > MAX_PACKET_CLOCK_ERROR_S:
            metrics.update(
                timeline_stage="packet_clock_gap", packet_clock_error_s=error
            )
            return None
        next_frame += len(packet.data)
    metrics.update(
        timeline_stage="contiguous",
        content_frames=next_frame - first.frame,
        packet_clock_error_max_s=max_clock_error,
    )
    return np.concatenate([packet.data for packet in span]), first.start_s


def quiet_window(
    packets: list[Packet],
    polls: list[float],
    start_s: float,
    end_s: float,
    *,
    max_gap_s: float = MAX_POLL_GAP_S,
    drain_delay_s: float = 0.0,
):
    result = {
        "status": "unqualified",
        "evidence": "monitor_gap",
        "rms": None,
        "packet_frames": 0,
    }
    before = [poll for poll in polls if poll <= start_s]
    after = [poll for poll in polls if poll >= end_s + drain_delay_s]
    if not before or not after or start_s >= end_s:
        return result
    relevant = [poll for poll in polls if before[-1] <= poll <= after[0]]
    if max_gap_s <= 0 or any(
        end - start > max_gap_s for start, end in pairwise(relevant)
    ):
        return result
    total = 0.0
    for packet in packets:
        if (
            not math.isfinite(packet.start_s)
            or packet.start_s <= 0
            or packet.flags & TIMESTAMP_ERROR
        ):
            result["evidence"] = "unreliable_packet_timestamp"
            return result
        if packet.end_s <= start_s or packet.start_s >= end_s:
            continue
        if packet.flags & (DISCONTINUITY | TIMESTAMP_ERROR):
            result["evidence"] = "unreliable_packet"
            return result
        first = max(0, round((start_s - packet.start_s) * RATE))
        last = min(len(packet.data), round((end_s - packet.start_s) * RATE))
        samples = packet.data[first:last]
        if len(samples):
            if not np.isfinite(samples).all():
                result["evidence"] = "nonfinite_packet_samples"
                return result
            power = float(np.sum(samples.astype(np.float64) ** 2))
            total += power
            result["packet_frames"] += len(samples)
    if not result["packet_frames"]:
        result.update(status="pass", evidence="monitored_no_packets")
    else:
        rms = math.sqrt(total / (2 * result["packet_frames"]))
        result.update(
            status="fail" if rms > SOUND_RMS else "pass",
            evidence="monitored_packets",
            rms=rms,
        )
    return result


def read_packet(ffi, check_error, client, *, expected_frames: int, capacity: int):
    api = client[0].lpVtbl
    data = ffi.new("BYTE **")
    frames = ffi.new("UINT32 *")
    flags = ffi.new("DWORD *")
    device_pos = ffi.new("UINT64 *")
    qpc = ffi.new("UINT64 *")
    status = api.GetBuffer(client, data, frames, flags, device_pos, qpc)
    if status == BUFFER_EMPTY:
        return None
    check_error(status)
    count = frames[0]
    if not count:
        return None
    try:
        if count != expected_frames or count > capacity:
            raise RuntimeError("invalid_native_packet_size")
        if flags[0] & SILENT:
            samples = np.zeros((count, 2), dtype=np.float32)
        else:
            if data[0] == ffi.NULL:
                raise RuntimeError("null_non_silent_packet")
            samples = (
                np.frombuffer(ffi.buffer(data[0], count * 8), dtype="<f4")
                .reshape(-1, 2)
                .copy()
            )
        return Packet(device_pos[0], qpc[0] / 10_000_000, flags[0], samples)
    finally:
        check_error(api.ReleaseBuffer(client, count))


def windows_qpc_offset() -> tuple[float, float]:
    """Calibrate native scaled QPC seconds to this process's perf_counter."""
    import ctypes

    frequency = ctypes.c_longlong()
    counter = ctypes.c_longlong()
    kernel = ctypes.windll.kernel32
    if (
        not kernel.QueryPerformanceFrequency(ctypes.byref(frequency))
        or frequency.value <= 0
    ):
        raise RuntimeError("qpc_frequency_unavailable")
    readings = []
    for _ in range(5):
        before = time.perf_counter()
        if not kernel.QueryPerformanceCounter(ctypes.byref(counter)):
            raise RuntimeError("qpc_unavailable")
        after = time.perf_counter()
        readings.append(
            (after - before, (before + after) / 2 - counter.value / frequency.value)
        )
    duration, offset = min(readings)
    if duration > 0.001:
        raise RuntimeError("qpc_calibration_uncertain")
    return offset, duration


def drain_recorder(
    connection, recorder, duration_s: float, requested_frames: int
) -> None:
    """Drain on the COM owner thread with bounded, separate pipe delivery."""
    from soundcard.mediafoundation import _com, _ffi

    offset, bracket = windows_qpc_offset()
    actual_frames = int(recorder.buffersize)
    device_period_s = float(recorder.deviceperiod[0])
    if actual_frames <= 0 or not math.isfinite(device_period_s) or device_period_s <= 0:
        raise RuntimeError("invalid_capture_geometry")
    client = recorder._ppCaptureClient[0]
    pending = Queue(maxsize=64)

    def send():
        try:
            while message := pending.get():
                connection.send(message)
        except (BrokenPipeError, EOFError, OSError):
            return

    sender = threading.Thread(target=send, daemon=True)
    sender.start()
    ready = time.perf_counter()
    pending.put_nowait(
        (
            "ready",
            ready,
            {
                "actual_buffer_frames": actual_frames,
                "requested_buffer_frames": requested_frames,
                "device_period_s": device_period_s,
                "qpc_calibration_bracket_s": bracket,
                "qpc_offset_s": offset,
            },
        )
    )
    previous_poll = ready
    next_poll_report = ready
    maximum_poll_gap = 0.0
    packet_count = 0
    raw_discontinuities = 0
    timestamp_errors = 0
    origin_flag = 0
    try:
        while time.perf_counter() - ready < duration_s:
            batch = []
            while available := recorder._capture_available_frames():
                packet = read_packet(
                    _ffi,
                    _com.check_error,
                    client,
                    expected_frames=available,
                    capacity=actual_frames,
                )
                if packet is None:
                    break
                flags = packet.flags
                raw_discontinuities += bool(flags & DISCONTINUITY)
                if not packet_count and flags & DISCONTINUITY:
                    # There is no preceding packet at capture-stream origin.
                    origin_flag = 1
                    flags &= ~DISCONTINUITY
                packet_count += 1
                stamp = packet.start_s + offset
                if not ready - 0.25 <= stamp <= time.perf_counter() + 0.05:
                    flags |= TIMESTAMP_ERROR
                timestamp_errors += bool(flags & TIMESTAMP_ERROR)
                batch.append(Packet(packet.frame, stamp, flags, packet.data))
                if len(batch) >= 64:
                    raise RuntimeError("packet_drain_bound_exceeded")
            now = time.perf_counter()
            maximum_poll_gap = max(maximum_poll_gap, now - previous_poll)
            previous_poll = now
            if batch or now >= next_poll_report:
                pending.put_nowait(
                    (
                        "packets",
                        now,
                        batch,
                        {
                            "packet_count": packet_count,
                            "discontinuity_raw_count": raw_discontinuities,
                            "session_origin_discontinuity_count": origin_flag,
                            "discontinuity_count": raw_discontinuities - origin_flag,
                            "timestamp_error_count": timestamp_errors,
                            "poll_gap_max_s": maximum_poll_gap,
                        },
                    )
                )
                next_poll_report = now + 0.02
            time.sleep(0.001)
        pending.put_nowait(None)
        sender.join(timeout=0.5)
    except Exception:
        # A stalled pipe cannot prove monitoring coverage. A full queue may
        # prevent delivery of this marker, so the parent also checks freshness.
        try:
            pending.put_nowait(("error", "packet_drain_failed"))
            pending.put_nowait(None)
            sender.join(timeout=0.5)
        except Exception:
            pass
