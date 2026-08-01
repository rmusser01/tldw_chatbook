"""VAD onset pre-roll: the recorder must not clip the start of speech.

Live-diagnosed defect: `AudioRecordingService._process_audio_chunk` walks a
captured chunk in 20 ms frames and hands only `vad.is_speech()`-accepted
frames to `_handle_audio_chunk` (and from there, to the callback that feeds
the transcriber). Low-energy speech onsets -- word-initial fricatives
especially -- are classified as non-speech at `vad_aggressiveness=3`, so the
first frame(s) of an utterance were silently dropped before transcription
ever saw them. Live consequence on real hardware with parakeet-mlx: "stop"
came back as "dot"/"top"-like forms, "send" as "and" -- the leading
consonant gone. Voice commands garbled; mid-stream prose was fine because
the VAD gate was already open by then.

The fix is a small pre-roll ring buffer: the recorder now remembers the last
`VAD_PREROLL_MS` (240 ms / 12 frames by default -- see the constant's
docstring in `recording_service.py`) of *rejected* frames, and replays them
through the exact same `_handle_audio_chunk` path the instant a frame is
accepted after a silence run, before the accepted frame itself. This
recovers the clipped onset without touching the no-VAD path or the
mid-speech (already-open-gate) behavior.

These are unit-level tests: no real microphone. A fake `vad` object with a
scripted accept/reject sequence stands in for `webrtcvad.Vad`, and a plain
list captures whatever `AudioRecordingService.callback` receives -- exactly
what `_handle_audio_chunk` forwards downstream.
"""

from __future__ import annotations

from typing import List
from unittest.mock import patch

import pytest

from tldw_chatbook.Audio.recording_service import AudioRecordingService

pytestmark = pytest.mark.unit

# 20 ms at 16 kHz, 16-bit mono -- the frame size `_process_audio_chunk` always
# uses for VAD, matching `AudioRecordingService.VAD_FRAME_DURATION_MS`.
FRAME_SIZE = 640


class _ScriptedVad:
    """Fake VAD: returns a pre-scripted accept/reject verdict, one per call.

    Raises if asked for more verdicts than were scripted, so a test that
    mis-sizes its chunk against its script fails loudly instead of silently
    padding with a default.
    """

    def __init__(self, script: List[bool]) -> None:
        self._script = list(script)
        self.calls = 0

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        self.calls += 1
        if not self._script:
            raise AssertionError("_ScriptedVad ran out of scripted verdicts")
        return self._script.pop(0)


def _frame(tag: int) -> bytes:
    """A `FRAME_SIZE`-byte frame whose content identifies it in assertions."""
    return bytes([tag % 256]) * FRAME_SIZE


def _make_service(**kwargs) -> AudioRecordingService:
    """A minimal, unstarted `AudioRecordingService` with VAD wired to a fake.

    Mirrors the construction pattern in `test_recording_service.py`: patch
    the backend-availability flag so `__init__` picks a backend without
    touching real hardware, then swap in a fake `vad` directly rather than
    depending on `webrtcvad` being installed.
    """
    with patch("tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE", True):
        with patch("tldw_chatbook.Audio.recording_service.pyaudio"):
            service = AudioRecordingService(backend="pyaudio", **kwargs)
    service.use_vad = True
    return service


class TestVadPreroll:
    def test_preroll_delivers_buffered_reject_frames_before_the_accepted_frame(self):
        """RED A: [reject r1, reject r2, accept a1] -> callback sees r1, r2, a1."""
        service = _make_service()
        delivered: List[bytes] = []
        service.callback = delivered.append

        r1, r2, a1 = _frame(1), _frame(2), _frame(3)
        service.vad = _ScriptedVad([False, False, True])

        service._process_audio_chunk(r1 + r2 + a1)

        assert delivered == [r1, r2, a1]

    def test_preroll_ring_buffer_keeps_only_the_last_12_rejected_frames(self):
        """20 rejects then an accept: only the last 12 rejects precede it."""
        service = _make_service()
        assert service._preroll_frames.maxlen == 12

        delivered: List[bytes] = []
        service.callback = delivered.append

        reject_frames = [_frame(i) for i in range(20)]
        accept_frame = _frame(99)
        service.vad = _ScriptedVad([False] * 20 + [True])

        service._process_audio_chunk(b"".join(reject_frames) + accept_frame)

        assert delivered == reject_frames[-12:] + [accept_frame]

    def test_no_double_delivery_across_accept_then_accept(self):
        """[reject r1, accept a1, accept a2] -> r1, a1, a2 -- r1 exactly once,
        nothing extra delivered between a1 and a2."""
        service = _make_service()
        delivered: List[bytes] = []
        service.callback = delivered.append

        r1, a1, a2 = _frame(1), _frame(2), _frame(3)
        service.vad = _ScriptedVad([False, True, True])

        service._process_audio_chunk(r1 + a1 + a2)

        assert delivered == [r1, a1, a2]

    def test_mid_speech_consecutive_accepts_pass_through_unchanged(self):
        """[accept, accept] with no rejects between -> exactly those two."""
        service = _make_service()
        delivered: List[bytes] = []
        service.callback = delivered.append

        a1, a2 = _frame(1), _frame(2)
        service.vad = _ScriptedVad([True, True])

        service._process_audio_chunk(a1 + a2)

        assert delivered == [a1, a2]

    def test_preroll_buffer_resets_after_each_flush(self):
        """[reject r1, accept a1, reject r2, accept a2] -> r1 a1 r2 a2; the
        second flush must contain only r2, never a stale r1 replay."""
        service = _make_service()
        delivered: List[bytes] = []
        service.callback = delivered.append

        r1, a1, r2, a2 = _frame(1), _frame(2), _frame(3), _frame(4)
        service.vad = _ScriptedVad([False, True, False, True])

        service._process_audio_chunk(r1 + a1 + r2 + a2)

        assert delivered == [r1, a1, r2, a2]

    def test_preroll_frames_count_toward_max_buffer_bytes_like_accepted_frames(self):
        """A pre-rolled (originally-rejected) frame consumes the byte budget
        exactly like any frame `_handle_audio_chunk` delivers -- because it
        goes through that same function. With the limit set to exactly one
        frame, the rejected frame alone exhausts it before the accepted
        frame that triggered its replay is even considered."""
        limit_calls: List[bool] = []
        service = _make_service(
            max_buffer_bytes=FRAME_SIZE,
            on_buffer_limit=lambda: limit_calls.append(True),
        )
        service.is_recording = True
        delivered: List[bytes] = []
        service.callback = delivered.append

        r1, a1 = _frame(1), _frame(2)
        service.vad = _ScriptedVad([False, True])

        service._process_audio_chunk(r1 + a1)

        # Only r1 fits the one-frame budget; a1 is truncated to nothing once
        # the pre-roll flush has already spent it, exactly as it would if r1
        # had simply been an ordinary accepted frame ahead of a1.
        assert service.audio_buffer == [r1]
        assert service._audio_buffer_bytes == FRAME_SIZE
        assert delivered == [r1]
        assert service.is_recording is False
        assert limit_calls == [True]
