"""Pause-driven segment finalization, and the recorder gate it rests on.

`VoiceFinal` fires mid-capture because `_processing_loop` finalizes the
segment in progress once `last_speech_time` has gone stale for longer than
`silence_threshold_seconds`. What makes that timer mean *silence* rather than
merely *no chunk* is upstream, in the recorder:
`AudioRecordingService._process_audio_chunk` splits capture into 20 ms frames
and hands only VAD-positive ones to `_audio_callback`. So the dictation
service queues and speech-stamps every chunk it is given, unconditionally --
"a chunk arrived" already means "speech arrived".

An earlier version of this branch re-gated inside `_audio_callback` on a
locally built `Vad`, scanning 960-byte (30 ms) windows. The recorder's frames
are 640 bytes at 16 kHz, so `range(0, 640 - 960 + 1, 960)` was empty, every
frame was judged silent, and dictation produced nothing at all. The
production-frame-shape tests below exist so that class of mismatch fails
here: they feed `_audio_callback` exactly what the recorder delivers.

The "Recorder gate" section drives the real `webrtcvad` library through the
recorder's own frame loop, since that loop is now the load-bearing filter.
Skipped, not xfailed, when `webrtcvad` truly is not installed -- that is the
documented degrade path (finals at stop only), not a test failure.
"""

from __future__ import annotations

import queue
import random
import struct
import threading
import time
from typing import Any, Dict, List, Optional

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _SpyVad:
    """Wraps a real `webrtcvad.Vad`, recording every call and any exception.

    The recorder's frame loop has no error handling around `is_speech()`, so
    a frame-contract violation (wrong frame size, wrong sample rate) would
    surface as a raised exception from the recording thread rather than as a
    quiet misclassification. This spy records the arguments it was handed so
    a test can assert the *contract* -- frame length, sample rate -- and not
    just the verdict.
    """

    def __init__(self, real_vad: Any) -> None:
        self._real = real_vad
        self.frame_lengths: List[int] = []
        self.sample_rates: List[int] = []
        self.raised: List[Exception] = []

    def is_speech(self, frame_bytes: bytes, sample_rate: int) -> bool:
        self.frame_lengths.append(len(frame_bytes))
        self.sample_rates.append(sample_rate)
        try:
            return self._real.is_speech(frame_bytes, sample_rate)
        except Exception as exc:
            self.raised.append(exc)
            raise


class _FakeTranscriptionService:
    """Real `TranscriptionService.transcribe_buffer` signature, one canned reply."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.buffer_calls: List[Dict[str, Any]] = []

    def transcribe_buffer(
        self,
        audio_data: bytes,
        sample_rate: int,
        channels: int = 1,
        sample_width: int = 2,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.buffer_calls.append({"audio_data": audio_data})
        return {"text": self._text}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

#: Bytes in one frame as the recorder actually emits them: 20 ms of 16-bit
#: mono PCM at 16 kHz. Derived the same way `_process_audio_chunk` derives
#: it, so a change to the recorder's frame duration shows up here as a
#: failure rather than as a stale literal.
PRODUCTION_FRAME_BYTES = int(16000 * 20 / 1000) * 2  # 640


def _speech_frame() -> bytes:
    """One production-shaped frame of speech-like content.

    Non-zero so it is not mistaken for the silence fixture; the dictation
    service does not inspect content at all, but a test reading as "silence
    is queued" would mislead about the intended production case.
    """
    return _noise_pcm(PRODUCTION_FRAME_BYTES)


def _noise_pcm(n_bytes: int) -> bytes:
    """Broadband, alternating-sign noise from a fixed seed.

    A real `Vad(2)` classifies this as speech in every 20 ms frame (verified
    directly against `webrtcvad`, not asserted on faith). Deterministic and
    reproducible across runs and machines -- never drawn from OS entropy at
    test time.
    """
    rng = random.Random(1234)
    samples = [
        rng.randint(8000, 32000) * (1 if rng.random() < 0.5 else -1)
        for _ in range(n_bytes // 2)
    ]
    return struct.pack(f"<{len(samples)}h", *samples)


def _service():
    """Build a `LazyLiveDictationService` without touching hardware.

    Mirrors `Tests/Audio/test_dictation_capture_release.py`'s `_service_with`:
    construct via `__new__` (skipping the real, device-opening constructor)
    and set every attribute the exercised code paths (`_audio_callback`,
    `_processing_loop`) actually read.
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
    service.stop_processing = threading.Event()
    service.processing_thread = None

    service.buffer_duration_ms = 500
    service.silence_threshold_seconds = (
        LazyLiveDictationService.SILENCE_THRESHOLD_SECONDS
    )
    service.privacy_settings = {"auto_clear_buffer": True, "save_history": False}

    service.streaming_transcriber = None
    service._transcription_service = None
    service._transcription_init_error = None
    service._audio_service = None
    service.transcription_provider = "faster-whisper"
    service.transcription_model = None
    service.language = "en"
    service.enable_commands = False

    service.on_partial_transcript = None
    service.on_final_transcript = None
    service.on_state_change = None
    service.on_error = None
    service.on_command = None

    return service


def _recorder(use_vad: bool, vad: Any = None, sample_rate: int = 16000):
    """Build an `AudioRecordingService` without a backend or a real device.

    `__new__` skips the constructor's backend probe (which would raise
    `NoAudioBackendError` in a headless run and open a device otherwise).
    The attributes `_process_audio_chunk` and `_handle_audio_chunk` read are
    populated, plus the ones `stop_recording` reads -- `__del__` calls it, so
    an incomplete double turns garbage collection into a
    `PytestUnraisableExceptionWarning` in whichever test happens to trigger
    the collection.
    """
    from tldw_chatbook.Audio.recording_service import AudioRecordingService

    recorder = AudioRecordingService.__new__(AudioRecordingService)
    recorder.sample_rate = sample_rate
    recorder.channels = 1
    recorder.use_vad = use_vad
    recorder.vad = vad
    recorder.max_buffer_bytes = None
    recorder.audio_buffer = []
    recorder._audio_buffer_bytes = 0
    recorder._buffer_limit_reached = False
    recorder.audio_queue = queue.Queue()
    recorder.is_recording = False
    recorder.on_buffer_limit = None
    recorder.callback = None
    recorder.recording_thread = None
    recorder.save_file = None
    recorder.backend = None
    recorder.pyaudio_instance = None
    recorder.stream = None
    return recorder


def _wait_until(predicate, timeout: float = 1.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# --------------------------------------------------------------------------
# `_audio_callback` on production-shaped frames
#
# These are the tests a 960-byte (or any >640-byte) re-gate inside
# `_audio_callback` must fail.
# --------------------------------------------------------------------------


def test_production_frame_refreshes_last_speech_time():
    """A 640-byte frame -- what the recorder actually delivers -- must stamp speech."""
    service = _service()
    service.last_speech_time = 0
    service._audio_callback(_speech_frame())
    assert service.last_speech_time > 0


def test_production_frame_is_queued_for_transcription():
    """The recorder already dropped the silence; nothing here may drop the speech."""
    service = _service()
    service._audio_callback(_speech_frame())
    assert not service.processing_queue.empty()
    item_type, data = service.processing_queue.get_nowait()
    assert item_type == "audio"
    assert data == _speech_frame()


def test_every_production_frame_is_queued_and_counted():
    """A run of frames, as a real capture delivers them, is queued in full."""
    service = _service()
    frame = _speech_frame()
    for _ in range(5):
        service._audio_callback(frame)

    assert service.processing_queue.qsize() == 5
    assert service.captured_bytes == 5 * PRODUCTION_FRAME_BYTES


def test_captured_bytes_counts_every_delivered_frame():
    """The capture-outcome logic distinguishes "no mic" from "no transcript"."""
    service = _service()
    before = service.captured_bytes
    service._audio_callback(_speech_frame())
    assert service.captured_bytes == before + PRODUCTION_FRAME_BYTES


# --------------------------------------------------------------------------
# The whole point: a pause in delivery finalizes a segment mid-capture
# --------------------------------------------------------------------------


def test_pause_in_frame_delivery_finalizes_a_segment_mid_capture():
    """Frames stop arriving for >threshold; a final fires before `stop_dictation()`.

    The pause is modelled the way production produces it -- the recorder's
    VAD simply stops calling the callback -- so the loop-level staleness
    check is the only thing that can fire the final.
    """
    service = _service()
    # A short threshold (not the 2.0s production default) keeps this test
    # fast without sleeping past a hard-coded production constant.
    service.silence_threshold_seconds = 0.2
    # A short buffer window so the periodic flush transcribes the queued
    # frames (populating `current_transcript`) well before the silence
    # threshold elapses -- otherwise `_finalize_current_segment` would find
    # nothing to finalize and never fire `on_final_transcript`.
    service.buffer_duration_ms = 10
    service._transcription_service = _FakeTranscriptionService(text="hello")

    finals: List[str] = []
    service.on_final_transcript = finals.append

    service.stop_processing.clear()
    service.processing_thread = threading.Thread(
        target=service._processing_loop, daemon=True
    )
    service.processing_thread.start()
    try:
        # A short burst of speech frames, then nothing at all.
        for _ in range(3):
            service._audio_callback(_speech_frame())
        assert _wait_until(lambda: bool(finals)), (
            "the pause in frame delivery never finalized a segment"
        )
    finally:
        service.stop_processing.set()
        service.processing_thread.join(timeout=2.0)

    assert finals == ["hello"]
    # The segment was committed and cleared -- not left dangling for
    # `stop_dictation()` to finalize a second time.
    assert service.current_transcript == ""


# --------------------------------------------------------------------------
# Recorder gate
#
# `AudioRecordingService._process_audio_chunk` is the filter that makes "no
# chunk delivered" equivalent to "no speech". These tests drive the real
# `webrtcvad` library through the recorder's own frame loop (never a
# reimplementation of it), so a genuine frame-contract violation -- wrong
# frame size, wrong sample-rate assumption -- fails here instead of only on
# a live capture.
# --------------------------------------------------------------------------


def test_recorder_delivers_speech_frames_and_withholds_silence():
    """The real library, through the real loop, must distinguish the two.

    Each half gets its own freshly constructed `Vad(2)`. `webrtcvad` keeps
    hangover state across calls on one instance (verified directly: running
    a noise chunk and then a silent chunk through the *same* `Vad` leaked
    speech-positive frames into the silent chunk's result), so sharing one
    instance between the two halves would contaminate the second.
    """
    webrtcvad = pytest.importorskip("webrtcvad")

    # 25 frames' worth of speech-like audio.
    speech_chunk = _noise_pcm(PRODUCTION_FRAME_BYTES * 25)
    delivered: List[bytes] = []
    recorder = _recorder(use_vad=True, vad=webrtcvad.Vad(2))
    recorder.callback = delivered.append
    recorder._process_audio_chunk(speech_chunk)

    assert len(delivered) == 25, (
        f"the recorder withheld speech frames: delivered {len(delivered)} of 25"
    )
    assert {len(frame) for frame in delivered} == {PRODUCTION_FRAME_BYTES}

    # All-zero, not a constant-DC pattern: a real `Vad` needs a genuinely
    # flat signal for a definitive negative.
    silent_delivered: List[bytes] = []
    silent_recorder = _recorder(use_vad=True, vad=webrtcvad.Vad(2))
    silent_recorder.callback = silent_delivered.append
    silent_recorder._process_audio_chunk(bytes(PRODUCTION_FRAME_BYTES * 25))

    assert silent_delivered == [], (
        f"the recorder delivered silence to the transcriber: "
        f"{len(silent_delivered)} frames"
    )


def test_recorder_frame_contract_matches_the_vad_api():
    """Every frame handed to `is_speech` is 20 ms at the recorder's own rate.

    The sample rate is read off the recorder rather than hard-coded, because
    hard-coding 16 kHz in a caller of `is_speech` is exactly the bug this
    branch removed from the dictation service.
    """
    webrtcvad = pytest.importorskip("webrtcvad")

    spy = _SpyVad(webrtcvad.Vad(2))
    recorder = _recorder(use_vad=True, vad=spy)
    recorder.callback = lambda _frame: None

    # Deliberately not a whole multiple of the frame size: the remainder
    # must be dropped, never truncated into a short frame the VAD rejects.
    recorder._process_audio_chunk(_noise_pcm(PRODUCTION_FRAME_BYTES * 3 + 17 * 2))

    assert spy.raised == [], (
        f"the real VAD raised on a frame the recorder handed it: {spy.raised!r}"
    )
    assert spy.frame_lengths == [PRODUCTION_FRAME_BYTES] * 3
    assert set(spy.sample_rates) == {recorder.sample_rate}


def test_recorder_without_vad_delivers_everything():
    """The degrade path: no `webrtcvad`, so nothing is withheld.

    Finals then fire only at `stop_dictation()`, exactly as before this
    branch. What must never happen is a crash or a silently empty capture.
    """
    delivered: List[bytes] = []
    recorder = _recorder(use_vad=False, vad=None)
    recorder.callback = delivered.append

    silence = bytes(PRODUCTION_FRAME_BYTES * 3)
    recorder._process_audio_chunk(silence)

    assert delivered == [silence]


# --------------------------------------------------------------------------
# `silence_threshold_seconds` config
#
# The threshold is the only knob on the mechanism above, and a bad value is
# silent: too small finalizes plain dictation into fragments, too large (or
# `inf`/`nan`) means a mid-capture final never fires at all and spoken
# commands stop working. Mirrors the `stop_join_timeout_seconds` coverage in
# `test_dictation_stop_join.py`, including the `nan`/`inf` cases -- both are
# valid TOML floats that survive `float()`, and `nan <= 0` is False, so a
# bare positivity check waves them straight through.
# --------------------------------------------------------------------------


def _stub_threshold_setting(monkeypatch, value) -> None:
    """Make `get_cli_setting` report `value` for the threshold key only."""
    from tldw_chatbook.Audio import dictation_service_lazy

    def _get(section: str, key: Any = None, default: Any = None) -> Any:
        if key is not None and not isinstance(key, str):
            key, default = None, key
        path = section if key is None else f"{section}.{key}"
        if path == "dictation.silence_threshold_seconds":
            return value
        return default

    monkeypatch.setattr(dictation_service_lazy, "get_cli_setting", _get)


def test_a_configured_threshold_is_honored(monkeypatch):
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    _stub_threshold_setting(monkeypatch, 0.75)
    assert LazyLiveDictationService._resolve_silence_threshold() == 0.75


@pytest.mark.parametrize(
    "bad",
    [
        float("nan"),
        float("inf"),
        "nan",
        "inf",
        0,
        -1.0,
        "not a number",
        None,
        [],
    ],
)
def test_an_unusable_threshold_falls_back_to_the_default(monkeypatch, bad):
    """A typo must not make finalization instantaneous, infinite, or NaN."""
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    _stub_threshold_setting(monkeypatch, bad)
    assert (
        LazyLiveDictationService._resolve_silence_threshold()
        == LazyLiveDictationService.SILENCE_THRESHOLD_SECONDS
    )
