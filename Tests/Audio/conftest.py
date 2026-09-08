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


def _meta(
    tmp_path,
    mode: str = "call",
    diarize_mic_channel: bool = False,
    user_display_name: str = "You",
) -> MeetingMeta:
    return MeetingMeta(
        folder=tmp_path, mode=mode, started_at="2026-09-04T14:30:00",
        mic_device="MacBook Pro Microphone", system_source="Native (macOS tap)",
        provider="faster-whisper", model="base.en",
        user_display_name=user_display_name,
        diarize_mic_channel=diarize_mic_channel,
    )


@pytest.fixture
def meeting_session_with_fake_capture(tmp_path):
    """Factory fixture: `meeting_session_with_fake_capture(mode=..., diarizer=..., diarize_mic_channel=...)`."""

    def _build(
        *, mode: str = "call", diarizer: Any = None, sinks: Any = None, diarize_mic_channel: bool = False,
        user_display_name: str = "You", close_diarizer_on_stop: bool = True,
    ) -> MeetingSession:
        capture = FakeCapture(mode)
        return MeetingSession(
            meta=_meta(tmp_path, mode, diarize_mic_channel, user_display_name),
            capture=capture,
            dictation_factory=lambda cap: FakeDictation(cap),
            sinks=sinks or [],
            diarizer=diarizer,
            close_diarizer_on_stop=close_diarizer_on_stop,
        )

    return _build


# --- opt-in real-worker helpers (task 8: 31827) -----------------------------
#: Engine name -> the packages its worker needs to be importable.
REAL_ENGINE_PACKAGES = {
    "speechbrain": ("torch", "torchaudio", "speechbrain", "sklearn"),
    "onnx": ("sherpa_onnx", "numpy"),
}


def real_engine_available(engine: str) -> bool:
    """True when every package `engine`'s worker imports is importable here."""
    import importlib.util

    for name in REAL_ENGINE_PACKAGES[engine]:
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def real_engine_kwargs(engine: str, models_dir) -> dict:
    """`LocalDiarizer` kwargs for an opt-in real-worker run of `engine`.

    The ONNX engine needs its models on disk. `$TLDW_DIARIZER_MODELS_DIR`
    points at an already-populated directory (how these are run repeatedly);
    with it unset they are downloaded into the test's own `tmp_path`. Never
    the user's real data dir -- an opt-in test must not write there.
    """
    import os
    from pathlib import Path

    if engine != "onnx":
        return {}
    env_dir = os.environ.get("TLDW_DIARIZER_MODELS_DIR")
    return {"models_dir_override": Path(env_dir) if env_dir else Path(models_dir)}
