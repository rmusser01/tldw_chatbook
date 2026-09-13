"""`TTS/playback_capability.py`: can this machine actually play reply audio?

Combines the two independent playback paths -- the in-process streaming
sink (`sounddevice`/PortAudio, PCM and WAV only) and the external player
binaries (`TTS/audio_player.py`'s format-aware catalogue) -- into the one
predicate the Console speech pipeline needs: "is this response format
playable here?", plus the adaptive fallback decision ("if not, would WAV
be playable?") that keeps reply speech working on machines whose player
binaries cannot decode the configured format (stock Fedora: no mpv/
mplayer/ffplay, and paplay/pw-play/wav-only by the conservative catalogue).
"""
from __future__ import annotations

import pytest

from tldw_chatbook.TTS import playback_capability as pc


def _sink(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    monkeypatch.setattr(pc, "sink_available", lambda: available)


def _players(monkeypatch: pytest.MonkeyPatch, formats_by_player: dict[str, set[str]]):
    """Fake a machine whose available players decode exactly these formats."""
    monkeypatch.setattr(
        pc,
        "find_player_for_format",
        lambda fmt: next(
            (
                name
                for name, formats in formats_by_player.items()
                if fmt in formats
            ),
            None,
        ),
    )


class TestLocallyPlayableFormats:
    def test_sink_available_plays_pcm_and_wav(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _sink(monkeypatch, True)
        _players(monkeypatch, {})
        assert pc.locally_playable_formats() == frozenset({"pcm", "wav"})

    def test_sink_unavailable_players_cover_their_formats(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _sink(monkeypatch, False)
        _players(monkeypatch, {"mpv": {"mp3", "opus", "aac", "flac", "wav"}})
        assert pc.locally_playable_formats() == frozenset(
            {"mp3", "opus", "aac", "flac", "wav"}
        )

    def test_sink_and_players_union(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _sink(monkeypatch, True)
        _players(monkeypatch, {"mpv": {"mp3", "flac", "wav"}})
        assert pc.locally_playable_formats() == frozenset(
            {"pcm", "mp3", "flac", "wav"}
        )

    def test_stock_linux_no_sink_no_players_plays_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _sink(monkeypatch, False)
        _players(monkeypatch, {})
        assert pc.locally_playable_formats() == frozenset()

    def test_format_playable_locally_membership(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _sink(monkeypatch, True)
        _players(monkeypatch, {})
        assert pc.format_playable_locally("wav") is True
        assert pc.format_playable_locally("pcm") is True
        assert pc.format_playable_locally("mp3") is False


class TestAdaptConsoleSpeechFormat:
    def test_unplayable_mp3_with_sink_adapts_to_wav(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _sink(monkeypatch, True)
        _players(monkeypatch, {})
        assert pc.adapt_console_speech_format("mp3") == "wav"

    def test_playable_mp3_keeps_mp3(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _sink(monkeypatch, False)
        _players(monkeypatch, {"mpv": {"mp3", "wav"}})
        assert pc.adapt_console_speech_format("mp3") == "mp3"

    def test_nothing_playable_returns_format_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No sink and no players at all: there is no better format to ask
        # for -- the failure-surfacing path owns telling the user.
        _sink(monkeypatch, False)
        _players(monkeypatch, {})
        assert pc.adapt_console_speech_format("mp3") == "mp3"

    def test_wav_stays_wav(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _sink(monkeypatch, True)
        _players(monkeypatch, {})
        assert pc.adapt_console_speech_format("wav") == "wav"

    def test_sinkless_wav_with_wav_player_stays_wav(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _sink(monkeypatch, False)
        _players(monkeypatch, {"aplay": {"wav"}})
        assert pc.adapt_console_speech_format("wav") == "wav"

    def test_sinkless_flac_with_only_aplay_adapts_to_wav(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _sink(monkeypatch, False)
        _players(monkeypatch, {"aplay": {"wav"}})
        assert pc.adapt_console_speech_format("flac") == "wav"
