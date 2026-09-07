"""Shared fixtures for Tests/Audio.

`meeting_session_with_fake_capture` builds a `MeetingSession` wired to a
fake capture/dictation pair (mirroring `test_meeting_session.py`'s own
`FakeCapture`/`FakeDictation`, plus `pcm_window` for the diarizer seam),
so diarizer-wiring tests don't need real audio devices.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Audio.meeting_session import MeetingMeta, MeetingSession


class FakeCapture:
    def __init__(self, mode: str = "call") -> None:
        self.mode = mode
        self.audio_position_s = 2.0
        self.last_speech_position_s = 2.0
        self.runs: list[Any] = []
        self.labels: dict[tuple[float, float], str] = {}
        self.default_label = "you"
        self.stops = 0
        self.paused = False

    def closed_runs_after(self, t: float) -> list[Any]:
        return [r for r in self.runs if getattr(r, "end_s", None) is not None and r.end_s > t]

    def dominant_source(self, a: float, b: float) -> str:
        return self.labels.get((round(a, 2), round(b, 2)), self.default_label)

    def pcm_window(self, source: str, start_s: float, end_s: float) -> bytes:
        # Content is irrelevant to these tests -- only non-emptiness gates
        # whether the session calls the diarizer's `assign`.
        return b"\x00\x01" * 160

    def stop_recording(self) -> None:
        self.stops += 1

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False


class FakeDictation:
    MAX_NON_STREAMING_SEGMENT_SECONDS = 30.0

    def __init__(self, capture: Any) -> None:
        self.capture = capture
        self.privacy_settings = {"auto_clear_buffer": False, "local_only": True}
        self.callbacks: dict[str, Any] = {}
        self.stopped = 0
        self.complete = True

    def start_dictation(self, **callbacks: Any) -> bool:
        self.callbacks = callbacks
        return True

    def stop_dictation(self) -> SimpleNamespace:
        self.stopped += 1
        return SimpleNamespace(transcription_complete=self.complete)


def _meta(tmp_path, mode: str = "call", diarize_mic_channel: bool = False) -> MeetingMeta:
    return MeetingMeta(
        folder=tmp_path, mode=mode, started_at="2026-09-04T14:30:00",
        mic_device="MacBook Pro Microphone", system_source="Native (macOS tap)",
        provider="faster-whisper", model="base.en",
        diarize_mic_channel=diarize_mic_channel,
    )


@pytest.fixture
def meeting_session_with_fake_capture(tmp_path):
    """Factory fixture: `meeting_session_with_fake_capture(mode=..., diarizer=..., diarize_mic_channel=...)`."""

    def _build(
        *, mode: str = "call", diarizer: Any = None, sinks: Any = None, diarize_mic_channel: bool = False,
    ) -> MeetingSession:
        capture = FakeCapture(mode)
        return MeetingSession(
            meta=_meta(tmp_path, mode, diarize_mic_channel),
            capture=capture,
            dictation_factory=lambda cap: FakeDictation(cap),
            sinks=sinks or [],
            diarizer=diarizer,
        )

    return _build
