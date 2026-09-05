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
