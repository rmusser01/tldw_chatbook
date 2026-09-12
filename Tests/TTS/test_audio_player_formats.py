"""Format-aware Linux player selection for `TTS/audio_player.py`.

The historical probe picked the FIRST available player binary regardless of
what format it can decode (`aplay` is WAV-only, `paplay`/`pw-play` decode via
libsndfile whose MP3 support is version/build-dependent) and never probed
`pw-play` at all -- so an MP3 artifact on a box with only `aplay` present was
handed to a player that cannot decode it, and a stock PipeWire Fedora box
(found no player at all) fell to a silent `play() -> False`. These tests pin
the format-aware selection contract.

All tests are deterministic on any host OS: `platform.system` and
`shutil.which` are monkeypatched inside `tldw_chatbook.TTS.audio_player`.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tldw_chatbook.TTS import audio_player
from tldw_chatbook.TTS.audio_player import SimpleAudioPlayer


def _linux(monkeypatch: pytest.MonkeyPatch, available: set[str]) -> None:
    monkeypatch.setattr(audio_player.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        audio_player.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in available else None,
    )


def _darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_player.platform, "system", lambda: "Darwin")


class TestFindPlayerForFormat:
    def test_linux_only_aplay_wav_yes_mp3_no(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _linux(monkeypatch, {"aplay"})
        assert audio_player.find_player_for_format("wav") == "aplay"
        assert audio_player.find_player_for_format("mp3") is None

    def test_linux_pw_play_serves_wav_and_flac_not_mp3(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Conservative: pw-play's MP3 support depends on the distro's
        # libsndfile build, so it must not be selected for MP3.
        _linux(monkeypatch, {"pw-play"})
        assert audio_player.find_player_for_format("wav") == "pw-play"
        assert audio_player.find_player_for_format("flac") == "pw-play"
        assert audio_player.find_player_for_format("mp3") is None

    def test_linux_paplay_serves_wav_and_flac_not_mp3(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _linux(monkeypatch, {"paplay"})
        assert audio_player.find_player_for_format("wav") == "paplay"
        assert audio_player.find_player_for_format("mp3") is None

    def test_linux_mpv_serves_every_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _linux(monkeypatch, {"mpv"})
        for fmt in ("mp3", "opus", "aac", "flac", "wav"):
            assert audio_player.find_player_for_format(fmt) == "mpv"

    def test_linux_no_players_means_none_for_any_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _linux(monkeypatch, set())
        assert audio_player.find_player_for_format("wav") is None
        assert audio_player.find_player_for_format("mp3") is None

    def test_darwin_afplay_serves_os_decodable_formats_not_ogg_opus(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # dev observed afplay exiting successfully WITHOUT decoding an
        # Ogg/Opus body -- worse than a clean failure -- so those
        # containers route to ffplay when installed and refuse otherwise.
        _darwin(monkeypatch)
        monkeypatch.setattr(audio_player.shutil, "which", lambda name: None)
        for fmt in ("mp3", "aac", "flac", "wav"):
            assert audio_player.find_player_for_format(fmt) == "afplay"
        assert audio_player.find_player_for_format("opus") is None
        assert audio_player.find_player_for_format("ogg") is None

    def test_darwin_ogg_opus_routes_to_ffplay_when_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _darwin(monkeypatch)
        monkeypatch.setattr(
            audio_player.shutil, "which", lambda name: "/opt/homebrew/bin/ffplay"
        )
        assert audio_player.find_player_for_format("opus") == "ffplay"
        assert audio_player.find_player_for_format("ogg") == "ffplay"

    def test_linux_capability_beats_list_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # aplay is probed before pw-play in the historical order; a FLAC
        # request must skip the incapable earlier player and select pw-play.
        _linux(monkeypatch, {"aplay", "pw-play"})
        assert audio_player.find_player_for_format("flac") == "pw-play"


class TestPlayerSupportedFormats:
    def test_per_player_format_sets_are_frozen_and_complete(self) -> None:
        for name in ("mpv", "mplayer", "ffplay", "pw-play", "paplay", "aplay"):
            formats = audio_player.player_supported_formats(name)
            assert isinstance(formats, frozenset)
            assert "wav" in formats

    def test_unknown_player_has_no_formats(self) -> None:
        assert audio_player.player_supported_formats("no-such-player") == frozenset()


class TestPlayUsesFormatAwareSelection:
    def test_play_refuses_format_the_selected_player_cannot_decode(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _linux(monkeypatch, {"aplay"})
        player = SimpleAudioPlayer()
        audio_file = tmp_path / "utterance.mp3"
        audio_file.write_bytes(b"fake mp3 body")

        def _no_spawn(*args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("no player process may be spawned for mp3")

        monkeypatch.setattr(audio_player.subprocess, "Popen", _no_spawn)
        assert player.play(audio_file) is False
        assert player.get_current_file() is None

    def test_play_selects_capable_player_for_format(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _linux(monkeypatch, {"pw-play"})
        player = SimpleAudioPlayer()
        audio_file = tmp_path / "utterance.wav"
        audio_file.write_bytes(b"fake wav body")

        fake_process = MagicMock()
        fake_process.poll.return_value = None
        monkeypatch.setattr(
            audio_player.subprocess,
            "Popen",
            lambda *args, **kwargs: fake_process,
        )
        assert player.play(audio_file) is True
        assert player._player_name == "pw-play"
