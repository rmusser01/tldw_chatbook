"""Shared fixtures for the first-run wizard suites."""

import pytest


class _SilentAudioPlayer:
    async def play(self, _path):
        return True

    async def stop(self):
        return None

    async def cleanup(self):
        return None


@pytest.fixture(autouse=True)
def _silent_audio_player(monkeypatch):
    """Voice "Test and Hear" creates the app's audio player on first use;
    keep wizard tests from spawning a real OS player. A test that needs a
    specific player sets app.audio_player or re-patches this class."""
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.AsyncAudioPlayer", _SilentAudioPlayer
    )
