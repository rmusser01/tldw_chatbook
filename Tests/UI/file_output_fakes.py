"""Native test double for UI playback, keeping the real coordinator and file run."""

import threading

from Tests.TTS.test_audio_output_runs import FilePlayer
from tldw_chatbook.TTS import audio_output_runs


def capture_file_playback(monkeypatch, paths):
    players = []

    class Player(FilePlayer):
        def __init__(self):
            super().__init__()
            players.append(self)

        def play_prepared(self, plan, **kwargs):
            result = super().play_prepared(plan, **kwargs)
            paths.append(plan.file_path)
            return result

    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", Player)
    return players


class GatedStopPlayer(FilePlayer):
    def __init__(self):
        super().__init__()
        self.stop_entered = threading.Event()
        self.release_stop = threading.Event()

    def stop(self, *, expected_token):
        self.stop_entered.set()
        assert self.release_stop.wait(5)
        return super().stop(expected_token=expected_token)
