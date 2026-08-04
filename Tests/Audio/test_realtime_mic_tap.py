# test_realtime_mic_tap.py
"""Tests for `RealtimeMicTap` (V4 task 3). See
`.superpowers/sdd/2026-08-04-realtime-voice-engine/task-3-brief.md`.

All tests inject a fake recorder via `recorder_factory` -- never a real
`AudioRecordingService` -- so this module never opens real audio hardware
and does not depend on `Tests/conftest.py`'s `_no_real_audio_device` guard
(which patches a different module, `Audio/streaming_sink.py`, anyway).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from tldw_chatbook.Audio.realtime_mic_tap import RealtimeMicTap

pytestmark = pytest.mark.unit


class FakeRecorder:
    """Stand-in for `AudioRecordingService`: records the constructor
    kwargs it was built with, captures the callback passed to
    `start_recording`, and lets a test push frames directly by calling
    `.callback(frame)` -- simulating the recorder's own background
    thread invoking it.
    """

    def __init__(self, *, start_result: bool = True, **kwargs):
        """Record constructor kwargs and the desired `start_recording`
        result.

        Args:
            start_result: Value `start_recording` returns.
            **kwargs: Captured verbatim as `init_kwargs` for assertions.
        """
        self.init_kwargs = kwargs
        self.callback = None
        self.start_calls = 0
        self.stop_calls = 0
        self._start_result = start_result

    def start_recording(self, callback):
        """Fake `start_recording`: capture the callback, return the
        configured result.

        Args:
            callback: The frame callback `RealtimeMicTap` passes in.

        Returns:
            The `start_result` this fake was configured with.
        """
        self.start_calls += 1
        self.callback = callback
        return self._start_result

    def stop_recording(self):
        """Fake `stop_recording`: just count the call.

        Returns:
            None.
        """
        self.stop_calls += 1
        return None


def make_factory(*, start_result: bool = True):
    """Build a `recorder_factory` callable that constructs one
    `FakeRecorder` and stashes it on `factory.instance` for the test to
    reach into (push frames, inspect kwargs, count calls).

    Args:
        start_result: Value the fake's `start_recording` will return.

    Returns:
        A callable matching `recorder_factory`'s expected signature
        (`Callable[..., Any]`), with an `.instance` attribute set to the
        most recently constructed `FakeRecorder` (or None before the tap
        calls it).
    """

    def factory(**kwargs):
        recorder = FakeRecorder(start_result=start_result, **kwargs)
        factory.instance = recorder
        return recorder

    factory.instance = None
    return factory


def test_constructor_kwargs_pinned_and_start_returns_true():
    """`start()` must build the recorder with exactly `backend=None,
    sample_rate=24000, channels=1, use_vad=False` and return True when
    the recorder starts successfully.
    """
    factory = make_factory(start_result=True)
    tap = RealtimeMicTap(lambda frame: None, recorder_factory=factory)

    assert tap.start() is True
    assert factory.instance.init_kwargs == {
        "backend": None,
        "sample_rate": 24000,
        "channels": 1,
        "use_vad": False,
    }
    assert factory.instance.start_calls == 1


def test_start_returns_false_on_recorder_failure():
    """`start()` returns False (device failure) when the recorder's own
    `start_recording` reports failure, without raising.
    """
    factory = make_factory(start_result=False)
    tap = RealtimeMicTap(lambda frame: None, recorder_factory=factory)

    assert tap.start() is False


def test_frames_buffer_before_ready_and_flush_in_order_on_mark_ready():
    """Frames pushed before `mark_ready()` must not reach `on_frames`
    immediately; `mark_ready()` flushes them in the order received, then
    subsequent frames stream straight through.
    """
    received: list[bytes] = []
    factory = make_factory()
    tap = RealtimeMicTap(received.append, recorder_factory=factory)
    tap.start()

    factory.instance.callback(b"frame1")
    factory.instance.callback(b"frame2")
    assert received == []

    tap.mark_ready()
    assert received == [b"frame1", b"frame2"]

    factory.instance.callback(b"frame3")
    assert received == [b"frame1", b"frame2", b"frame3"]


def test_pre_ready_buffer_bound_evicts_oldest_keeps_newest():
    """Once the pre-ready buffer exceeds `max_buffer_seconds *
    sample_rate * 2` bytes, the oldest buffered frame(s) are dropped so
    the newest is kept.
    """
    received: list[bytes] = []
    factory = make_factory()
    # max_buffer_bytes = 0.01 * 100 * 2 = 2 bytes -- exactly one 2-byte
    # frame fits; a second push must evict the first.
    tap = RealtimeMicTap(
        received.append,
        sample_rate=100,
        recorder_factory=factory,
        max_buffer_seconds=0.01,
    )
    tap.start()

    factory.instance.callback(b"AA")
    factory.instance.callback(b"BB")

    tap.mark_ready()
    assert received == [b"BB"]


def test_set_gated_true_drops_frames_device_stays_open():
    """`set_gated(True)` drops incoming frames without forwarding or
    buffering them; `set_gated(False)` resumes normal flow.
    """
    received: list[bytes] = []
    factory = make_factory()
    tap = RealtimeMicTap(received.append, recorder_factory=factory)
    tap.start()
    tap.mark_ready()

    tap.set_gated(True)
    factory.instance.callback(b"dropped")
    assert received == []

    tap.set_gated(False)
    factory.instance.callback(b"kept")
    assert received == [b"kept"]

    # Device/recorder was never stopped by gating.
    assert factory.instance.stop_calls == 0


def test_gating_before_ready_also_drops_without_buffering():
    """Gating applies before `mark_ready()` too: gated frames are dropped
    outright, not buffered for a later flush.
    """
    received: list[bytes] = []
    factory = make_factory()
    tap = RealtimeMicTap(received.append, recorder_factory=factory)
    tap.start()

    tap.set_gated(True)
    factory.instance.callback(b"dropped")
    tap.set_gated(False)

    tap.mark_ready()
    assert received == []


def test_stop_prevents_any_further_callbacks():
    """After `stop()`, pushing more frames through the captured callback
    must not invoke `on_frames`, and the underlying recorder's
    `stop_recording` must have been called exactly once.
    """
    received: list[bytes] = []
    factory = make_factory()
    tap = RealtimeMicTap(received.append, recorder_factory=factory)
    tap.start()
    tap.mark_ready()

    tap.stop()
    factory.instance.callback(b"frame-after-stop")

    assert received == []
    assert factory.instance.stop_calls == 1


def test_stop_before_mark_ready_discards_buffered_frames():
    """Buffered pre-ready frames are discarded by `stop()`, not flushed;
    a `mark_ready()` call after `stop()` must not replay them.
    """
    received: list[bytes] = []
    factory = make_factory()
    tap = RealtimeMicTap(received.append, recorder_factory=factory)
    tap.start()

    factory.instance.callback(b"buffered")
    tap.stop()
    tap.mark_ready()

    assert received == []


def test_import_pulls_no_heavy_transcription_dependencies():
    """Import-lightness pin: importing this module alone must never pull
    `faster_whisper`, `torch`, or `nemo` into `sys.modules` -- those are
    heavy, optional transcription-stack dependencies pulled by other
    parts of `tldw_chatbook.Audio` when imported carelessly at module
    scope. Run in a fresh subprocess (the same venv as pytest, via
    `sys.executable`) since the pytest process itself may have already
    imported plenty by the time this test runs.
    """
    script = (
        "import sys, tldw_chatbook.Audio.realtime_mic_tap; "
        "assert 'faster_whisper' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'nemo' not in str(sys.modules.keys())"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"import-lightness probe failed (exit={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
