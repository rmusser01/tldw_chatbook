"""Meeting capture: energy attribution (Task 4) and the mixer (Task 5)."""
from __future__ import annotations

import math

import pytest

from tldw_chatbook.Audio.meeting_capture import (
    ABS_MIN_RMS,
    EnergyRing,
    SpeechRun,
    rms_int16,
)

pytestmark = pytest.mark.unit


def _fill(ring: EnergyRing, start_s: float, end_s: float, mic: float, sys_: float) -> None:
    t = start_s
    while t < end_s:
        ring.add(t, mic, sys_)
        t += 0.1


def test_rms_of_constant_and_empty():
    assert rms_int16(b"") == 0.0
    assert rms_int16(b"\x00\x10" * 10) == pytest.approx(4096.0)


def test_abs_min_is_minus_60_dbfs():
    assert ABS_MIN_RMS == pytest.approx(32768 * 10 ** (-60 / 20), rel=1e-6)


def test_mic_dominant_window_is_you():
    ring = EnergyRing()
    _fill(ring, 0.0, 5.0, mic=2000.0, sys_=0.0)
    assert ring.dominant_source(0.0, 5.0) == "you"


def test_system_dominant_window_is_others():
    ring = EnergyRing()
    _fill(ring, 0.0, 5.0, mic=0.0, sys_=2000.0)
    assert ring.dominant_source(0.0, 5.0) == "others"


def test_balanced_window_is_both():
    ring = EnergyRing()
    _fill(ring, 0.0, 5.0, mic=2000.0, sys_=2000.0)
    assert ring.dominant_source(0.0, 5.0) == "both"


def test_room_noise_below_adaptive_floor_does_not_flip_to_both():
    ring = EnergyRing()
    # 30 s of steady room noise on the mic (p10 == 200 -> floor 600),
    # then the remote party talks while the mic keeps its noise.
    _fill(ring, 0.0, 30.0, mic=200.0, sys_=0.0)
    _fill(ring, 30.0, 35.0, mic=200.0, sys_=3000.0)
    assert ring.floor("mic", 35.0) == pytest.approx(600.0)
    assert ring.dominant_source(30.0, 35.0) == "others"


def test_digital_silence_uses_absolute_minimum_floor():
    ring = EnergyRing()
    _fill(ring, 0.0, 30.0, mic=0.0, sys_=0.0)
    assert ring.floor("sys", 30.0) == pytest.approx(ABS_MIN_RMS)


def test_no_active_buckets_falls_back_to_higher_raw_sum():
    ring = EnergyRing()
    _fill(ring, 0.0, 30.0, mic=0.0, sys_=0.0)
    _fill(ring, 30.0, 31.0, mic=5.0, sys_=20.0)  # both under ABS_MIN
    assert ring.dominant_source(30.0, 31.0) == "others"


def test_ring_forgets_beyond_horizon():
    ring = EnergyRing(horizon_s=1.0)
    _fill(ring, 0.0, 3.0, mic=1000.0, sys_=0.0)
    assert ring.dominant_source(0.0, 0.5) == "others"  # evicted: nothing active, sums tie -> not "you"


def test_speech_run_defaults_open():
    run = SpeechRun(1.5)
    assert run.end_s is None and math.isclose(run.start_s, 1.5)


# ---------------------------------------------------------------- Task 5
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_chatbook.Audio.meeting_capture import MeetingCapture, mix_int16
from tldw_chatbook.Audio.wav_writer import PlaceholderWavWriter

FRAME_BYTES = 640
SILENT = b"\x00\x00" * 320
LOUD = b"\x00\x20" * 320  # 8192 amplitude
QUIET = b"\x10\x00" * 320  # 16 amplitude


class FakeRecorder:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.callback = None
        self.stopped = 0

    def start_recording(self, callback=None, save_to_file=None):
        self.callback = callback
        return True

    def stop_recording(self):
        self.stopped += 1
        return None

    def get_audio_devices(self):
        return [{"id": 0, "name": "fake"}]

    def set_device(self, device_id):
        return True


class FakeTap:
    def __init__(self):
        self.on_frames = None
        self.state = "stopped"
        self.stops = 0

    def start(self, on_frames):
        self.on_frames = on_frames
        self.state = "running"
        return True

    def stop(self):
        self.stops += 1
        self.state = "stopped"

    def push(self, frame: bytes):
        self.on_frames(frame)


class EnergyVad:
    """Stand-in for webrtcvad: speech == RMS above 100."""

    def is_speech(self, frame: bytes, rate: int) -> bool:
        return rms_int16(frame) > 100.0


def _capture(tmp_path, *, call_mode=True, silence=2.0, preroll=12):
    writers = {"mixed": PlaceholderWavWriter(tmp_path / "mixed.wav")}
    tap = None
    if call_mode:
        writers["you"] = PlaceholderWavWriter(tmp_path / "you.wav")
        writers["others"] = PlaceholderWavWriter(tmp_path / "others.wav")
        tap = FakeTap()
    recorders: list[FakeRecorder] = []

    def factory(**kwargs):
        recorders.append(FakeRecorder(**kwargs))
        return recorders[-1]

    cap = MeetingCapture(
        mic_recorder_factory=factory,
        tap=tap,
        writers=writers,
        vad_factory=EnergyVad,
        silence_threshold_s=silence,
        preroll_frames=preroll,
    )
    return cap, recorders, tap, writers


def test_mix_saturates_and_keeps_length():
    a = np.full(320, 30000, dtype=np.int16).tobytes()
    b = np.full(320, 30000, dtype=np.int16).tobytes()
    mixed = np.frombuffer(mix_int16(a, b), dtype=np.int16)
    assert mixed.size == 320 and int(mixed[0]) == 32767


@settings(max_examples=100, deadline=None)
@given(
    st.lists(st.integers(-32768, 32767), min_size=1, max_size=64),
    st.lists(st.integers(-32768, 32767), min_size=1, max_size=64),
)
def test_mix_equals_clipped_sum(a_vals, b_vals):
    n = min(len(a_vals), len(b_vals))
    a = np.asarray(a_vals[:n], dtype=np.int16)
    b = np.asarray(b_vals[:n], dtype=np.int16)
    expected = np.clip(a.astype(np.int32) + b.astype(np.int32), -32768, 32767)
    got = np.frombuffer(mix_int16(a.tobytes(), b.tobytes()), dtype=np.int16)
    assert np.array_equal(got.astype(np.int32), expected)


def test_start_builds_mic_with_retain_off_and_starts_tap(tmp_path):
    cap, recorders, tap, _ = _capture(tmp_path)
    assert cap.mode == "call"
    assert cap.start_recording(callback=lambda b: None) is True
    assert recorders[0].init_kwargs["retain_audio"] is False
    assert recorders[0].init_kwargs["use_vad"] is False
    assert recorders[0].init_kwargs["chunk_size"] == 320
    assert tap.state == "running"


def test_mic_frame_pulls_one_tap_frame_and_zero_fills(tmp_path):
    cap, recorders, tap, writers = _capture(tmp_path)
    cap.start_recording(callback=lambda b: None)
    tap.push(QUIET)
    recorders[0].callback(QUIET)   # pairs with the pushed frame
    recorders[0].callback(QUIET)   # nothing queued -> zeros
    assert writers["you"].bytes_written == 2 * FRAME_BYTES
    assert writers["others"].bytes_written == 2 * FRAME_BYTES
    assert writers["mixed"].bytes_written == 2 * FRAME_BYTES
    assert cap.audio_position_s == pytest.approx(0.04)
    mixed_second = np.frombuffer(QUIET, dtype=np.int16)
    assert cap.levels()[1] == 0.0 or cap.levels()[1] < cap.levels()[0]


def test_backlog_over_200ms_drops_one_extra_frame_per_tick(tmp_path):
    cap, recorders, tap, _ = _capture(tmp_path)
    cap.start_recording(callback=lambda b: None)
    for _ in range(20):          # 400 ms queued
        tap.push(QUIET)
    recorders[0].callback(QUIET)  # takes one, drops one extra
    assert cap._tap_backlog_bytes() == 18 * FRAME_BYTES
    for _ in range(8):
        recorders[0].callback(QUIET)
    assert cap._tap_backlog_bytes() <= 10 * FRAME_BYTES


def test_tap_buffer_is_bounded_to_one_second(tmp_path):
    cap, _, tap, _ = _capture(tmp_path)
    cap.start_recording(callback=lambda b: None)
    for _ in range(80):
        tap.push(QUIET)
    assert cap._tap_backlog_bytes() == 50 * FRAME_BYTES


def test_vad_runs_open_with_preroll_close_after_silence(tmp_path):
    cap, recorders, tap, _ = _capture(tmp_path, silence=0.1, preroll=2)
    got: list[bytes] = []
    cap.start_recording(callback=got.append)
    mic = recorders[0].callback
    for _ in range(5):
        mic(SILENT)              # 0.00-0.10 s, pre-roll keeps last 2
    mic(LOUD)                    # 0.10-0.12 s speech
    mic(LOUD)                    # 0.12-0.14 s
    assert cap.closed_runs_after(0.0) == []
    assert cap.last_speech_position_s == pytest.approx(0.14)
    for _ in range(6):
        mic(SILENT)              # gap of 0.12 s >= 0.1 -> run closes
    runs = cap.closed_runs_after(0.0)
    assert len(runs) == 1
    assert runs[0].start_s == pytest.approx(0.06)   # 0.10 - 2 pre-roll frames
    assert runs[0].end_s == pytest.approx(0.14)
    assert b"".join(got) == SILENT + SILENT + LOUD + LOUD


def test_closed_runs_after_filters_by_end(tmp_path):
    cap, recorders, _, _ = _capture(tmp_path, silence=0.02, preroll=0)
    cap.start_recording(callback=lambda b: None)
    mic = recorders[0].callback
    mic(LOUD); mic(SILENT); mic(SILENT)   # run 1 ends at 0.02
    mic(LOUD); mic(SILENT); mic(SILENT)   # run 2 ends at 0.08
    assert [r.end_s for r in cap.closed_runs_after(0.05)] == [pytest.approx(0.08)]


def test_pause_discards_tap_frames_and_writes_nothing(tmp_path):
    cap, recorders, tap, writers = _capture(tmp_path)
    cap.start_recording(callback=lambda b: None)
    cap.pause()
    assert cap.paused
    tap.push(QUIET)
    recorders[0].callback(QUIET)
    assert writers["mixed"].bytes_written == 0
    assert cap._tap_backlog_bytes() == 0
    cap.resume()
    recorders[0].callback(QUIET)
    assert writers["mixed"].bytes_written == FRAME_BYTES


def test_writer_error_is_recorded_as_fault_not_raised(tmp_path):
    cap, recorders, _, writers = _capture(tmp_path)
    cap.start_recording(callback=lambda b: None)
    writers["mixed"].close()   # next write raises ValueError
    recorders[0].callback(QUIET)
    assert isinstance(cap.fault, ValueError)


def test_stop_closes_writers_open_run_and_tap(tmp_path):
    cap, recorders, tap, writers = _capture(tmp_path, silence=5.0, preroll=0)
    cap.start_recording(callback=lambda b: None)
    recorders[0].callback(LOUD)
    cap.stop_recording()
    assert all(w.closed for w in writers.values())
    assert recorders[0].stopped == 1 and tap.stops == 1
    assert cap.closed_runs_after(0.0)[0].end_s == pytest.approx(0.02)


def test_room_mode_has_no_tap_and_only_mixed_writer(tmp_path):
    cap, recorders, tap, writers = _capture(tmp_path, call_mode=False)
    assert cap.mode == "room" and tap is None
    cap.start_recording(callback=lambda b: None)
    recorders[0].callback(QUIET)
    assert set(writers) == {"mixed"} and writers["mixed"].bytes_written == FRAME_BYTES
    assert cap.dominant_source(0.0, 0.02) in {"you", "others", "both"}


def test_recorder_surface_forwards_to_mic(tmp_path):
    cap, recorders, _, _ = _capture(tmp_path)
    cap.start_recording()
    assert cap.get_audio_devices() == [{"id": 0, "name": "fake"}]
    assert cap.set_device(0) is True
    assert cap.is_available() is True
    assert cap.sample_rate == 16000 and cap.channels == 1
    assert 0.0 <= cap.get_audio_level() <= 1.0


def test_partial_chunks_are_carried_not_dropped(tmp_path):
    cap, recorders, _, _ = _capture(tmp_path, silence=5.0, preroll=0)
    got: list[bytes] = []
    cap.start_recording(callback=got.append)
    mic = recorders[0].callback
    loud = b"\x00\x20" * 350          # 700 bytes: one slice + 60-byte remainder
    mic(loud)
    assert b"".join(got) == loud[:640]
    mic(b"\x00\x20" * 290)             # 580 bytes: remainder 60 + 580 = 640 -> one more slice
    assert b"".join(got) == loud[:640] + (b"\x00\x20" * 320)
    assert cap.last_speech_position_s == pytest.approx(0.04)
