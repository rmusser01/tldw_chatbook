"""TASK-38612(a): the VAD loop must carry sub-frame chunk tails forward.

`AudioRecordingService._process_audio_chunk` walks each captured chunk in
20 ms frames (640 bytes at 16 kHz / 16-bit mono). The old loop
`range(0, len(chunk) - frame_size + 1, frame_size)` silently dropped the
`len(chunk) % frame_size` tail bytes of EVERY chunk, with no cross-chunk
carry. The default 500 ms capture buffer happens to be frame-aligned, but
`buffer_duration_ms` is user-configurable (1-2000 ms) and any non-aligned
duration meant continuous, permanent audio loss at the microphone-capture
stage -- every chunk lost the same fraction of a frame forever.

The fix assembles frames the same way `speech_presentation.PcmEnvelope`
windows PCM: prepend the carried partial frame, slice off whole frames,
keep the new tail for the next chunk. The pre-roll deque semantics are
unchanged. Tests below are pure logic: a scripted/fake VAD classifier, no
microphone, no real backend.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tldw_chatbook.Audio.recording_service import AudioRecordingService

pytestmark = pytest.mark.unit

# 20 ms at 16 kHz, 16-bit mono -- the frame size `_process_audio_chunk`
# always uses for VAD (`AudioRecordingService.VAD_FRAME_DURATION_MS`).
FRAME_SIZE = 640


class _ScriptedVad:
    """Fake VAD returning a pre-scripted verdict per call (as in
    test_recording_vad_preroll.py)."""

    def __init__(self, script: list[bool]) -> None:
        self._script = list(script)
        self.calls = 0

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        self.calls += 1
        if not self._script:
            raise AssertionError("_ScriptedVad ran out of scripted verdicts")
        return self._script.pop(0)


class _AlwaysAcceptVad:
    """Fake VAD that accepts every frame and records each frame's bytes."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        self.frames.append(frame)
        return True


def _frame(tag: int) -> bytes:
    """A `FRAME_SIZE`-byte frame whose content identifies it in assertions."""
    return bytes([tag % 256]) * FRAME_SIZE


def _make_service(**kwargs) -> AudioRecordingService:
    """Minimal unstarted service with VAD forced on and a fake backend probe.

    Same construction pattern as `test_recording_vad_preroll.py`: patch the
    backend-availability flag so `__init__` never touches real hardware,
    then install a fake VAD per test.
    """
    with (
        patch("tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE", True),
        patch("tldw_chatbook.Audio.recording_service.pyaudio"),
    ):
        service = AudioRecordingService(backend="pyaudio", **kwargs)
    service.use_vad = True
    return service


class TestAssembleVadFramesHelper:
    """Direct unit tests of the pure frame-assembly helper.

    The helper import lives inside each test so the behavioral tests in
    this file still COLLECT (and fail on behavior) against the pre-fix
    module, where no helper exists yet.
    """

    def test_whole_frames_plus_tail(self):
        from tldw_chatbook.Audio.recording_service import _assemble_vad_frames

        data = bytes(range(256)) * 3  # 768 bytes: 1 frame + 128 tail
        frames, carry = _assemble_vad_frames(b"", data[:768], FRAME_SIZE)
        assert frames == [data[:FRAME_SIZE]]
        assert carry == data[FRAME_SIZE:]

    def test_pending_prefix_completes_a_frame(self):
        from tldw_chatbook.Audio.recording_service import _assemble_vad_frames

        pending = bytes([7]) * 300
        rest = bytes([9]) * (FRAME_SIZE - 300 + 40)
        frames, carry = _assemble_vad_frames(pending, rest, FRAME_SIZE)
        assert frames == [pending + rest[: FRAME_SIZE - 300]]
        assert carry == rest[FRAME_SIZE - 300 :]

    def test_chunk_smaller_than_one_frame_is_all_carry(self):
        from tldw_chatbook.Audio.recording_service import _assemble_vad_frames

        frames, carry = _assemble_vad_frames(b"", b"abc", FRAME_SIZE)
        assert frames == []
        assert carry == b"abc"

    def test_no_samples_ever_dropped_or_duplicated(self):
        from tldw_chatbook.Audio.recording_service import _assemble_vad_frames

        pending = b""
        stream = b""
        seen = b""
        for size in (1000, 96, 640, 1, 639, 2000, 123, 4567):
            chunk = bytes(((i * 37) % 256) for i in range(size))
            stream += chunk
            frames, pending = _assemble_vad_frames(pending, chunk, FRAME_SIZE)
            for frame in frames:
                assert len(frame) == FRAME_SIZE
            seen += b"".join(frames)
        # Every byte that ever entered is either in a frame or still carried.
        assert seen + pending == stream


class TestVadFrameCarry:
    def test_sub_frame_tail_carries_into_next_chunk(self):
        """96-byte remainder (the audit's example) must not be dropped."""
        service = _make_service()
        delivered: list[bytes] = []
        service.callback = delivered.append

        f1, f2, f3, f4 = _frame(1), _frame(2), _frame(3), _frame(4)
        service.vad = _ScriptedVad([True, True, True, True])

        # chunk 1: two whole frames + the first 96 bytes of f3
        service._process_audio_chunk(f1 + f2 + f3[:96])
        # chunk 2: the remaining 544 bytes of f3 + all of f4
        service._process_audio_chunk(f3[96:] + f4)

        assert delivered == [f1, f2, f3, f4]

    def test_chunks_smaller_than_one_frame_accumulate(self):
        """300 + 300 + 40 bytes must assemble into exactly one frame."""
        service = _make_service()
        delivered: list[bytes] = []
        service.callback = delivered.append
        vad = _ScriptedVad([True])
        service.vad = vad

        whole = _frame(42)
        for part in (whole[:300], whole[300:600], whole[600:]):
            service._process_audio_chunk(part)

        assert delivered == [whole]
        assert vad.calls == 1

    def test_no_samples_lost_across_non_aligned_chunk_stream(self):
        """A long user-configurable-duration stream loses zero samples."""
        service = _make_service()
        delivered: list[bytes] = []
        service.callback = delivered.append
        vad = _AlwaysAcceptVad()
        service.vad = vad

        stream = b"".join(
            bytes(((i * 31 + chunk_index) % 256) for i in range(1000))
            for chunk_index in range(7)
        )
        # 7 x 1000 = 7000 bytes = 10 frames + 240 tail; the final 400 bytes
        # complete an 11th frame only if the carry works, leaving a 360-byte
        # tail legitimately still carried (a partial frame awaits more audio).
        stream += bytes(((i * 17) % 256) for i in range(400))
        for start in range(0, len(stream), 1000):
            service._process_audio_chunk(stream[start : start + 1000])

        delivered_bytes = b"".join(delivered)
        # Zero loss, zero duplication, order preserved: everything captured
        # is either inside a delivered whole frame or still carried.
        assert delivered_bytes + service._vad_pending_frame == stream
        assert len(delivered_bytes) % FRAME_SIZE == 0
        assert all(len(frame) == FRAME_SIZE for frame in vad.frames)

    def test_carried_completion_of_a_rejected_frame_feeds_preroll(self):
        """Carry must not disturb pre-roll: a cross-chunk-completed rejected
        frame goes to the pre-roll deque, and the next accepted frame
        replays it first."""
        service = _make_service()
        delivered: list[bytes] = []
        service.callback = delivered.append

        rejected = _frame(5)
        accepted = _frame(6)
        service.vad = _ScriptedVad([False, True])

        service._process_audio_chunk(rejected[300:])  # partial frame
        service._process_audio_chunk(rejected[:300] + accepted)

        assert delivered == [rejected, accepted]
