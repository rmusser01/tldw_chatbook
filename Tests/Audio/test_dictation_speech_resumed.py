"""`VoiceSpeechResumed`: the silence-to-speech transition, service side.

`_audio_callback` (see `dictation_service_lazy.py`) receives only VAD-positive
frames -- `AudioRecordingService._process_audio_chunk` splits capture into
20 ms frames and hands only speech-positive ones to the callback, so "a frame
arrived" already means "speech arrived" (see
`test_dictation_vad_finalization.py`'s module docstring for the full
rationale). `last_speech_time` starts at 0 (`__init__`/`start_dictation()`)
and is zeroed again by `_processing_loop`'s silence gate once a pause exceeds
`silence_threshold_seconds` -- see `SILENCE_THRESHOLD_SECONDS` and the check
just above where it sets `self.last_speech_time = 0`.

The chosen emission rule distinguishes "first frame of a fresh capture"
(never a resume -- capture start is not a resume) from "a frame arriving
after the silence gate zeroed `last_speech_time`" (a genuine resume) with a
single per-capture flag, `_capture_saw_first_frame`:

    resumed = self.last_speech_time == 0 and self._capture_saw_first_frame

Deliberately NOT derived from a delivery-gap time delta: without `webrtcvad`
the recorder forwards every chunk unconditionally, so `last_speech_time` only
ever advances and a gap alone (with no finalize in between) can never make it
0 -- see `test_a_delivery_gap_without_a_finalize_does_not_emit_a_resume`
below, which is the test a delta-based rule would fail.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import List

import pytest

pytestmark = pytest.mark.unit


def _service():
    """Build a `LazyLiveDictationService` without touching hardware.

    Mirrors `Tests/Audio/test_dictation_vad_finalization.py`'s `_service()`:
    built via `__new__` (skipping the device-opening constructor), with every
    attribute `_audio_callback` reads populated by hand. `on_speech_resumed`
    and `_capture_saw_first_frame` are deliberately left UNSET here so a bare
    `__new__` build exercises the same class-level `__new__`-safety defaults
    every other caller of this service relies on (see
    `LazyLiveDictationService`'s class docstring comments next to
    `on_segment_transcribing`).
    """
    from tldw_chatbook.Audio.dictation_service_lazy import (
        DictationState,
        LazyLiveDictationService,
    )

    service = LazyLiveDictationService.__new__(LazyLiveDictationService)

    service.state = DictationState.LISTENING
    service.state_lock = threading.Lock()

    service.audio_buffer = []
    service.buffer_lock = threading.Lock()
    service.captured_bytes = 0
    service._current_audio_level = 0.0
    service.last_speech_time = 0

    service.transcript_segments = []
    service.current_transcript = ""
    service.transcript_lock = threading.Lock()

    service.processing_queue = queue.Queue()

    return service


def _frame() -> bytes:
    """One production-shaped 640-byte (20 ms @ 16 kHz) frame.

    Content is irrelevant to `_audio_callback`'s speech-resumed logic -- it
    only ever inspects `last_speech_time` and the per-capture flag -- so an
    all-zero frame is fine here (unlike the VAD-facing tests in
    `test_dictation_vad_finalization.py`, which need genuinely noise-like
    content to satisfy a real `Vad`).
    """
    return bytes(640)


# --------------------------------------------------------------------------
# (a) capture start is never a resume
# --------------------------------------------------------------------------


def test_first_frame_of_a_capture_does_not_emit_resume():
    """`last_speech_time == 0` on the very first frame too -- must not fire."""
    service = _service()
    resumed: List[None] = []
    service.on_speech_resumed = lambda: resumed.append(None)

    service._audio_callback(_frame())

    assert resumed == []
    # The flag must flip so a LATER zero (post-silence-gate) is recognized.
    assert service._capture_saw_first_frame is True


# --------------------------------------------------------------------------
# (b) a continuous run of frames emits nothing
# --------------------------------------------------------------------------


def test_a_continuous_run_of_frames_emits_no_resume():
    service = _service()
    resumed: List[None] = []
    service.on_speech_resumed = lambda: resumed.append(None)

    for _ in range(5):
        service._audio_callback(_frame())

    assert resumed == []


# --------------------------------------------------------------------------
# (c) a frame after the silence gate zeroed last_speech_time: exactly ONE
# --------------------------------------------------------------------------


def test_a_frame_after_the_silence_gate_emits_exactly_one_resume():
    """Mirrors how `test_dictation_vad_finalization.py` isolates
    `_audio_callback` from `_processing_loop`: the silence gate's own
    `self.last_speech_time = 0` is reproduced directly rather than run
    through the real processing thread, since that line is the entire
    contract this event rests on.
    """
    service = _service()
    resumed: List[None] = []
    service.on_speech_resumed = lambda: resumed.append(None)

    service._audio_callback(_frame())  # capture start, not a resume
    assert resumed == []

    service.last_speech_time = 0  # the silence gate firing mid-capture

    service._audio_callback(_frame())  # the resume
    assert resumed == [None]

    service._audio_callback(_frame())  # back to a continuous run
    assert resumed == [None]  # still exactly one


# --------------------------------------------------------------------------
# (d) a delivery gap alone (no finalize) must not fire spuriously
# --------------------------------------------------------------------------


def test_a_delivery_gap_without_a_finalize_does_not_emit_a_resume():
    """The degraded/no-VAD case: `last_speech_time` only ever advances.

    Without `webrtcvad` the recorder forwards every chunk unconditionally, so
    a "gap" in delivery is modeled directly on `last_speech_time`: set it to a
    timestamp far in the past (stale, but deliberately NOT 0) between the two
    calls, standing in for whatever real wall-clock gap produced that
    staleness -- no finalize, no `== 0`, ever touched it. A time-delta-based
    rule (`(time.time() - last_speech_time) > threshold`, the exact
    alternative the brief ruled out) WOULD fire here; the shipped `== 0` rule
    does not. Verified: this assertion fails under the delta rule and passes
    under the shipped one (a plain back-to-back `_audio_callback()` pair with
    no intervening staleness, tried first, does NOT discriminate the two
    rules -- both leave `last_speech_time` non-zero and neither fires -- so
    it is not a valid test of this case; it is behaviourally a duplicate of
    test (b) above).
    """
    service = _service()
    resumed: List[None] = []
    service.on_speech_resumed = lambda: resumed.append(None)

    service._audio_callback(_frame())
    assert service.last_speech_time != 0

    # A real gap: stale, but non-zero. No finalize -- nothing set this to 0.
    service.last_speech_time = time.time() - 60.0

    service._audio_callback(_frame())

    assert resumed == []


# --------------------------------------------------------------------------
# (e) callback exceptions are swallowed, like every sibling callback.
#
# `_notify_speech_resumed()` (review finding F7: routed through a helper
# matching every sibling's shape, e.g. `_notify_segment_transcribing`) runs
# the callback strictly AFTER `buffer_lock` is released and after
# `last_speech_time`/`_capture_saw_first_frame` are already updated (review
# finding F6: those two are now read-check-written under `buffer_lock`,
# shared with `_processing_loop`'s silence check, to close a missed-resume
# race -- see `_audio_callback`'s inline comment). That reordering means a
# raise from the callback no longer sits *ahead of* any state this class still
# needs to update -- it is the last thing `_audio_callback` does -- so
# `_audio_callback`'s own outer `try/except` would swallow an unguarded raise
# just as effectively as an inner guard, and "last_speech_time still updates"
# is no longer a test that distinguishes guarded from unguarded (it is true
# either way, since the update already happened before the callback runs).
# The guard's own behaviour has to be pinned directly on `_notify_speech_resumed`
# instead: call it in isolation and assert no exception escapes.
# --------------------------------------------------------------------------


def test_notify_speech_resumed_swallows_a_raising_callback():
    """Guarded like every sibling `_notify_*` (`_notify_segment_transcribing`,
    `_notify_state_change`, `_notify_error`): a raising `on_speech_resumed`
    must not escape `_notify_speech_resumed()` itself. Mutation check:
    dropping the `try/except` inside `_notify_speech_resumed` makes this
    raise, and only this test catches it -- see the module-level note above
    for why the previous version of this test (asserting on
    `last_speech_time` after `_audio_callback`) stopped being able to.
    """
    service = _service()

    def _boom():
        raise RuntimeError("boom")

    service.on_speech_resumed = _boom

    service._notify_speech_resumed()  # must not raise


def test_state_updates_before_the_callback_runs_not_after():
    """Ordering invariant the F6 lock fix depends on: `last_speech_time` and
    `_capture_saw_first_frame` are already updated by the time
    `on_speech_resumed` is invoked, not the other way around -- otherwise a
    slow or blocking callback would delay state `_processing_loop`'s silence
    check reads on its own thread. Observed from inside the callback itself,
    the only vantage point that can see the ordering rather than just the
    end state.
    """
    service = _service()
    observed: List[tuple] = []

    def _observe():
        observed.append((service.last_speech_time, service._capture_saw_first_frame))

    service.on_speech_resumed = _observe

    service._audio_callback(_frame())  # capture start: not a resume, no call
    service.last_speech_time = 0  # simulate the silence gate firing

    service._audio_callback(_frame())  # the resume

    assert len(observed) == 1
    seen_last_speech_time, seen_saw_first_frame = observed[0]
    assert seen_last_speech_time != 0, (
        "on_speech_resumed ran before last_speech_time was refreshed"
    )
    assert seen_saw_first_frame is True


# --------------------------------------------------------------------------
# `start_dictation()`'s per-capture reset (`_capture_saw_first_frame = False`)
#
# Every test above builds a service via `__new__` and drives `_audio_callback`
# directly -- none of them go through `start_dictation()`, so none can catch
# a regression to its reset of `_capture_saw_first_frame`. That reset is what
# keeps a REUSED service instance honest across captures:
# `UI/Dictation_Window_Improved.py` keeps one `LazyLiveDictationService` alive
# across start/stop cycles (`_initialize_service()` returns early once one
# exists) rather than building a fresh one per capture the way the Console's
# `default_service_factory` does. Without the reset, a service whose PREVIOUS
# capture ended via the silence gate (`_capture_saw_first_frame` left `True`,
# `last_speech_time` left at 0 -- `_cleanup()` touches neither) would report
# its NEXT capture's own first frame as a resume: exactly the false-positive
# test (a) above exists to rule out, just reached through a different door.
#
# This needs the real `start_dictation()`/`stop_dictation()` machinery (a
# fake recorder + fake transcription service, no `__new__` shortcut), mirrored
# from `Tests/Audio/test_dictation_lazy_transcription.py`'s `_build_service`
# pattern rather than imported from it -- kept local and minimal since this
# file only needs enough of that machinery to drive two real captures.
# --------------------------------------------------------------------------


class _FakeTranscriptionServiceForReset:
    """Just enough of `TranscriptionService` for `start_dictation()` to run."""

    def transcribe_buffer(self, **kwargs) -> dict:
        return {"text": ""}


class _FakeRecorderForReset:
    """Stands in for `AudioRecordingService`; never opens a device.

    `feed()` calls the recorder callback synchronously (on the calling
    thread), matching how `_FakeRecorder.feed()` works in
    `test_dictation_lazy_transcription.py` -- there is no separate recorder
    thread to synchronize with here.
    """

    def __init__(self) -> None:
        self.callback = None
        self.is_recording = False

    def start_recording(self, callback) -> bool:
        self.callback = callback
        self.is_recording = True
        return True

    def stop_recording(self) -> bytes:
        self.is_recording = False
        return b""

    def feed(self, chunk: bytes) -> None:
        assert self.callback is not None, "recording was never started"
        self.callback(chunk)


def _stub_short_silence_threshold(monkeypatch, threshold: float = 0.15) -> None:
    """Make the real silence gate fire quickly, and keep other reads hermetic."""
    from tldw_chatbook.Audio import dictation_service_lazy

    values = {
        "dictation.silence_threshold_seconds": threshold,
        "dictation.buffer_duration_ms": 10,
        "dictation.privacy.save_history": False,
        "dictation.privacy.encrypt_history": True,
        "dictation.privacy.local_only": False,
        "dictation.privacy.auto_clear_buffer": True,
    }

    def _get(section: str, key=None, default=None):
        if key is not None and not isinstance(key, str):
            key, default = None, key
        path = section if key is None else f"{section}.{key}"
        return values.get(path, default)

    monkeypatch.setattr(dictation_service_lazy, "get_cli_setting", _get)


def _wait_until(predicate, timeout: float = 1.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_a_reused_service_instances_second_capture_first_frame_is_not_a_resume(
    monkeypatch,
):
    """The real two-capture scenario `start_dictation()`'s reset exists for.

    Capture 1 runs on the real processing thread until the real silence gate
    fires (short `silence_threshold_seconds`, real elapsed time -- not a
    hand-set `last_speech_time`), leaving `_capture_saw_first_frame == True`
    and `last_speech_time == 0` on the instance. Capture 2 reuses that SAME
    instance (never rebuilt) and its own very first frame must not be
    reported as a resume, even though `last_speech_time` is still 0 from
    capture 1's gate. Mutation check: deleting
    `self._capture_saw_first_frame = False` from `start_dictation()`'s reset
    block makes this test fail (verified below, then reverted).
    """
    _stub_short_silence_threshold(monkeypatch)
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    service = LazyLiveDictationService(
        transcription_provider="faster-whisper",
        enable_commands=False,
    )
    service._transcription_service = _FakeTranscriptionServiceForReset()
    recorder = _FakeRecorderForReset()
    service._audio_service = recorder

    # -- Capture 1: ends via the real silence gate -----------------------
    resumed_1: List[None] = []
    started_1 = service.start_dictation(
        on_final_transcript=lambda _text: None,
        on_speech_resumed=lambda: resumed_1.append(None),
    )
    assert started_1 is True

    recorder.feed(_frame())  # capture 1's own first frame: not a resume
    assert resumed_1 == []

    assert _wait_until(lambda: service.last_speech_time == 0), (
        "the real silence gate never zeroed last_speech_time"
    )
    # The state the reset exists to fix, confirmed present before capture 2:
    assert service._capture_saw_first_frame is True

    service.stop_dictation()

    # -- Capture 2: reuses the SAME instance, per `Dictation_Window_Improved` --
    resumed_2: List[None] = []
    started_2 = service.start_dictation(
        on_final_transcript=lambda _text: None,
        on_speech_resumed=lambda: resumed_2.append(None),
    )
    assert started_2 is True

    recorder.feed(_frame())  # capture 2's own first frame: must not resume

    assert resumed_2 == [], (
        "a reused service instance reported its second capture's first "
        "frame as a resume -- start_dictation()'s _capture_saw_first_frame "
        "reset was not applied"
    )

    service.stop_dictation()
