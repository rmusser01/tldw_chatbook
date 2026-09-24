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


def _calls_killpg(tree: ast.Module) -> bool:
    """Whether this module actually signals a process GROUP.

    AST, not a text search, precisely so the two modules that explain in a
    comment why they do *not* killpg are not matched by their own
    explanation.
    """
    return any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "killpg")
            or (isinstance(node.func, ast.Name) and node.func.id == "killpg")
        )
        for node in ast.walk(tree)
    )


def _spawn_sites_missing_start_new_session() -> list[str]:
    """TTS spawns that need `start_new_session` and do not declare it.

    "Need" is the point. `start_new_session` is not free: it removes the
    child from the terminal's process group, and so removes the SIGHUP/SIGINT
    that is otherwise the only thing reaping a grandchild. It earns its keep
    exactly where teardown does a group kill -- there, without it, `killpg`
    resolves to the app's OWN group and SIGKILLs the TUI.

    So the invariant is a pairing, not a blanket: a module that calls
    `os.killpg` must detach every spawn; a module that tears down its direct
    child only (`terminate`/`kill`) must not. `audio_cpp_supervisor.py` and
    `backends/chatterbox.py` are the second kind, which is why commit
    2e63022 took the flag back off them -- and why asserting it everywhere
    made this check contradict the code it guards.
    """
    offenders: list[str] = []
    for path in sorted(_TTS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not _calls_killpg(tree):
            continue
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


def test_every_group_killing_tts_module_detaches_its_spawns():
    """The defect class, not just the three named lines."""
    assert _spawn_sites_missing_start_new_session() == []


def test_the_pairing_rule_still_has_teeth():
    """Negative control: the check must fail for a module that killpgs a
    child it never detached -- which is the original P0, verbatim."""
    source = (
        "import os, signal, subprocess\n"
        "def run():\n"
        "    p = subprocess.Popen(['player'])\n"
        "    os.killpg(os.getpgid(p.pid), signal.SIGKILL)\n"
    )
    tree = ast.parse(source)
    assert _calls_killpg(tree)
    spawns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _SPAWN_CALLS
        and not any(kw.arg == "start_new_session" for kw in node.keywords)
    ]
    assert len(spawns) == 1, "the offending spawn shape stopped matching"


def test_a_module_that_only_kills_its_direct_child_is_not_required_to_detach():
    """The other half of the pairing: no killpg, no requirement. Without
    this the check would demand a flag that is a net negative there."""
    tree = ast.parse(
        "import subprocess\n"
        "def run():\n"
        "    p = subprocess.Popen(['worker'])\n"
        "    p.terminate()\n"
        "    p.kill()\n"
    )
    assert not _calls_killpg(tree)


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


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
def test_a_real_spawn_lands_the_player_in_its_own_process_group(
    tmp_path, monkeypatch
):
    """The platform half, unmocked (Qodo review of #2799).

    The AST census above proves the keyword is WRITTEN; the guard test
    mocks `os.getpgid` outright. Neither proves `start_new_session=True`
    actually detaches the child -- if that expression ever evaluated to
    False (it is `os.name == "posix"`, not a literal), both would stay
    green while `stop()` went back to SIGKILLing the app's own group.

    Player *selection* is stubbed because it probes the box for real
    audio binaries; the spawn is not -- this goes through the production
    `subprocess.Popen(..., start_new_session=...)` call.
    """
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

    player = SimpleAudioPlayer()
    player._system = "Darwin"
    player._player_name = "ffplay"  # not mpv/mplayer: the plain spawn branch
    # `sh -c <script> <arg>` puts the trailing file path in $0, so the
    # production `self._player_cmd + [str(file_path)]` stays intact.
    player._player_cmd = ["/bin/sh", "-c", "sleep 30"]
    monkeypatch.setattr(player, "_select_player_for_artifact", lambda _p: None)

    assert player.play(audio) is True
    process = player._current.process
    assert process is not None and process.poll() is None, "the player exited"

    try:
        pgid = os.getpgid(process.pid)
        assert pgid != os.getpgrp(), (
            "the player shares the app's process group; stop()'s last-resort "
            "killpg would SIGKILL the TUI itself"
        )
        assert pgid == process.pid, "the child is not its own session leader"
    finally:
        # Never signal a group that resolved to ours -- the very hazard
        # under test would otherwise take the test runner down with it.
        if os.getpgid(process.pid) == os.getpgrp():
            process.kill()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=5)
