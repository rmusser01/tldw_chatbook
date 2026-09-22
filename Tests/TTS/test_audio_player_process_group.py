"""TASK-32892 P0-3: TTS subprocesses must not share the app's process group.

`SimpleAudioPlayer.stop()`'s last-resort branch does
`os.killpg(os.getpgid(process.pid), SIGKILL)`. A `subprocess.Popen` child
started without `start_new_session` INHERITS the parent's process group, so
that pgid IS the app's -- the "force kill a stuck player" path SIGKILLs the
TUI itself (and everything else in the terminal's foreground group) with no
shutdown, no `close_tts_resources()`, and no DB quiesce.

Two guards, matching `Event_Handlers/LLM_Management_Events/server_lifecycle.py`
`_signal_server_process_group` (TASK-32806.5, the nearest in-repo precedent):
the spawn side makes the child a session leader so its pid IS its group id,
and the signal side still refuses a group that resolves to our own.
"""

from __future__ import annotations

import ast
import os
import signal
import subprocess
from pathlib import Path

import pytest

from tldw_chatbook.TTS import audio_player
from tldw_chatbook.TTS.audio_player import PlaybackState, SimpleAudioPlayer

_TTS_ROOT = Path(audio_player.__file__).resolve().parent

_SPAWN_CALLS = {"Popen", "create_subprocess_exec", "create_subprocess_shell"}


def _spawn_sites_missing_start_new_session() -> list[str]:
    """Every TTS subprocess spawn that does not detach into its own session."""
    offenders: list[str] = []
    for path in sorted(_TTS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else (
                func.id if isinstance(func, ast.Name) else None
            )
            if name not in _SPAWN_CALLS:
                continue
            if any(kw.arg == "start_new_session" for kw in node.keywords):
                continue
            if any(kw.arg is None for kw in node.keywords):
                continue  # **kwargs: the flag may be supplied there
            offenders.append(
                f"{path.relative_to(_TTS_ROOT.parent)}:{node.lineno} {name}"
            )
    return offenders


def test_every_tts_subprocess_spawn_detaches_into_its_own_session():
    """The defect class, not just the three named lines."""
    assert _spawn_sites_missing_start_new_session() == []


class _StuckProcess:
    """A player wedged past both waits -- `stop()`'s last-resort branch."""

    pid = 4242

    def poll(self):
        return None

    def kill(self):
        return None

    def terminate(self):
        return None

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired(cmd="player", timeout=timeout or 0)


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
def test_stop_never_signals_the_apps_own_process_group(monkeypatch):
    """The guard that makes the fix independent of the spawn site.

    Reproduces the catastrophic shape directly: a process whose pgid
    resolves to OURS. Unfixed, `stop()` hands that straight to
    `os.killpg(..., SIGKILL)` -- suicide.
    """
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "getpgid", lambda _pid: os.getpgrp())
    monkeypatch.setattr(
        os, "killpg", lambda pgid, sig: signalled.append((pgid, sig))
    )

    player = SimpleAudioPlayer()
    player._system = "Darwin"
    player._player_name = "afplay"
    player._current.process = _StuckProcess()
    player._current.state = PlaybackState.PLAYING

    assert player.stop() is True
    assert signalled == [], (
        f"stop() SIGKILLed the app's own process group: {signalled}"
    )
    assert signal.SIGKILL not in {sig for _pgid, sig in signalled}
