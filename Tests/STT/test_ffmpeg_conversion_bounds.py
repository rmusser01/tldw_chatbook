"""Both STT ffmpeg conversions must be bounded and must not inherit stdin.

Tier-2 code review S07: an AST census of every ``subprocess.run`` under
``tldw_chatbook/`` found 26 call sites, 14 of which pass ``timeout=``. These
two -- the ordinary transcription staging paths -- did not. ffmpeg on a
malformed or truncated container can block indefinitely, and it reads stdin by
default, so an unbounded call inherits the parent's stdin and can both wedge
and steal input. Neither conversion feeds ffmpeg anything on stdin: the input
is always a file path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _capture(monkeypatch, module) -> list[dict]:
    seen: list[dict] = []

    def fake_run(args, **kwargs):
        seen.append(kwargs)
        # Both call sites write their output to the last positional argument.
        Path(args[-1]).write_bytes(b"")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return seen


def test_parakeet_staging_conversion_is_bounded(tmp_path, monkeypatch):
    from tldw_chatbook.STT import parakeet_onnx

    seen = _capture(monkeypatch, parakeet_onnx)
    source = tmp_path / "input.m4a"
    source.write_bytes(b"not really audio")

    with parakeet_onnx._prepared_wav(source, "/usr/bin/ffmpeg"):
        pass

    assert seen, "ffmpeg was never invoked"
    assert seen[0].get("timeout"), "ffmpeg conversion ran with no timeout"
    assert seen[0].get("stdin") is subprocess.DEVNULL


def test_transcribe_cpp_conversion_is_bounded(tmp_path, monkeypatch):
    from tldw_chatbook.STT import transcribe_cpp

    seen = _capture(monkeypatch, transcribe_cpp)
    source = tmp_path / "input.m4a"
    source.write_bytes(b"not really audio")

    # The staged WAV is empty, so the post-conversion read fails -- that is
    # after the call under test and is not what this pins.
    with pytest.raises(Exception):
        transcribe_cpp._pcm_16k_mono(source, ffmpeg_path="/usr/bin/ffmpeg")

    assert seen, "ffmpeg was never invoked"
    assert seen[0].get("timeout"), "ffmpeg conversion ran with no timeout"
    assert seen[0].get("stdin") is subprocess.DEVNULL
