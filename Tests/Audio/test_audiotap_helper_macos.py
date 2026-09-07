"""Opt-in: compiles and runs the real macOS tap helper. Never in CI.

Run: TLDW_RUN_AUDIOTAP_HELPER_TEST=1 pytest Tests/Audio/test_audiotap_helper_macos.py -p no:cacheprovider
"""
from __future__ import annotations

import contextlib
import os
import platform
import shutil
import subprocess
import sys
import threading

import pytest

from tldw_chatbook.Audio import system_audio_tap as sat

pytestmark = [pytest.mark.real_audio_device, pytest.mark.integration]

_SYSTEM_SOUND = "/System/Library/Sounds/Submarine.aiff"


@contextlib.contextmanager
def _system_audio_playing():
    """Play a system sound on a loop for the body's duration.

    A Core Audio process tap emits frames only while something is actually
    playing, so a silent machine yields zero frames from a perfectly working
    helper. Driving real output makes this a genuine capture check rather
    than a room-noise lottery. Falls back to ambient audio if the tools are
    absent.
    """
    afplay = shutil.which("afplay")
    if afplay is None or not os.path.exists(_SYSTEM_SOUND):
        yield
        return
    player = subprocess.Popen(
        ["bash", "-c", f'while true; do "{afplay}" "{_SYSTEM_SOUND}"; done'],
        preexec_fn=os.setsid,
    )
    try:
        yield
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(player.pid), 15)
        with contextlib.suppress(Exception):
            player.wait(timeout=5)


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("swiftc") is None or os.environ.get("TLDW_RUN_AUDIOTAP_HELPER_TEST") != "1",
    reason="opt-in: set TLDW_RUN_AUDIOTAP_HELPER_TEST=1 on a Mac with the System Audio Recording grant",
)
def test_helper_compiles_and_emits_frames(tmp_path):
    assert sat.macos_version_ok(platform.mac_ver()[0])
    helper = sat.ensure_helper(tmp_path, executable=str(tmp_path / "nowhere"))
    assert helper is not None and helper.exists()
    tap = sat.SubprocessTap((str(helper),))
    frames: list[bytes] = []
    got = threading.Event()

    def on_frames(frame: bytes) -> None:
        frames.append(frame)
        if len(frames) >= 50:
            got.set()

    with _system_audio_playing():
        assert tap.start(on_frames)
        try:
            assert got.wait(10.0), f"no frames; stderr: {tap.last_stderr}"
        finally:
            tap.stop()
    assert tap.state == "stopped" and all(len(f) == 640 for f in frames)
