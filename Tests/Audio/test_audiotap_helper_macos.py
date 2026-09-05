"""Opt-in: compiles and runs the real macOS tap helper. Never in CI.

Run: TLDW_RUN_AUDIOTAP_HELPER_TEST=1 pytest Tests/Audio/test_audiotap_helper_macos.py -p no:cacheprovider
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
import threading

import pytest

from tldw_chatbook.Audio import system_audio_tap as sat

pytestmark = [pytest.mark.real_audio_device, pytest.mark.integration]


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

    assert tap.start(on_frames)
    assert got.wait(10.0), f"no frames; stderr: {tap.last_stderr}"
    tap.stop()
    assert tap.state == "stopped" and all(len(f) == 640 for f in frames)
