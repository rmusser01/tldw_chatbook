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
    a "gap" in delivery is simulated here purely by NOT calling
    `_audio_callback` for a while -- `last_speech_time` stays at whatever the
    last real call set it to, never 0, so the `== 0` rule cannot fire from a
    gap alone. A time-delta-based rule would need its own threshold (and its
    own false-positive surface on every ordinary inter-frame gap); this test
    is what such a rule would fail.
    """
    service = _service()
    resumed: List[None] = []
    service.on_speech_resumed = lambda: resumed.append(None)

    service._audio_callback(_frame())
    assert service.last_speech_time != 0

    # No finalize in between -- just another delivery, standing in for one
    # arriving after an arbitrarily long gap.
    service._audio_callback(_frame())

    assert resumed == []


# --------------------------------------------------------------------------
# (e) callback exceptions are swallowed, like every sibling callback --
# and, specifically, must not abort the rest of `_audio_callback`.
# --------------------------------------------------------------------------


def test_a_raising_callback_does_not_prevent_last_speech_time_from_updating():
    """Guarded like `on_final_transcript`/`on_partial_transcript`'s siblings.

    `_audio_callback` already has its own outer `try/except` around the whole
    method body, which would swallow ANY exception raised inside it -- so "no
    exception escapes `_audio_callback`" is not, by itself, proof this
    callback is guarded correctly; it would be true even completely
    unguarded. What the outer catch-all does NOT protect is the code that
    runs *after* the raise point inside that same `try` block: an unguarded
    `on_speech_resumed()` call would abort the rest of `_audio_callback` the
    instant it raises, silently skipping the `last_speech_time` refresh right
    after it -- which would make the microphone look like it had gone silent.
    That is the concrete, observable thing a dropped guard breaks, and what
    this test actually pins.
    """
    service = _service()

    def _boom():
        raise RuntimeError("boom")

    service.on_speech_resumed = _boom

    service._audio_callback(_frame())  # capture start
    service.last_speech_time = 0  # simulate the silence gate firing

    before = service.last_speech_time
    service._audio_callback(_frame())  # would resume; callback raises

    assert service.last_speech_time > before
