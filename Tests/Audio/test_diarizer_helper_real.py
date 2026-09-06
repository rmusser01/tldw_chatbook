"""Opt-in: spawns the REAL SpeechBrain worker. Needs torch; never in CI.

Run: TLDW_RUN_DIARIZER_TEST=1 pytest Tests/Audio/test_diarizer_helper_real.py -p no:cacheprovider

Feeds two distinct synthetic tones through the real subprocess and asserts the
worker returns two stable, distinct cluster ids -- a genuine end-to-end check
of spawn, READY, embedding, and clustering.
"""
from __future__ import annotations

import importlib.util
import math
import os
import struct

import pytest

pytestmark = [pytest.mark.integration]


def _diarization_importable() -> bool:
    for name in ("torch", "torchaudio", "speechbrain", "sklearn"):
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def _tone(freq: float, seconds: float = 2.0, sr: int = 16000) -> bytes:
    """A mono PCM16 sine tone of the given frequency."""
    frames = int(seconds * sr)
    return b"".join(
        struct.pack("<h", int(0.5 * 32767 * math.sin(2 * math.pi * freq * i / sr)))
        for i in range(frames)
    )


@pytest.mark.skipif(
    os.environ.get("TLDW_RUN_DIARIZER_TEST") != "1" or not _diarization_importable(),
    reason="opt-in: set TLDW_RUN_DIARIZER_TEST=1 with the `diarization` extra installed",
)
def test_real_worker_assigns_two_stable_ids():
    from tldw_chatbook.Audio.diarizer_local import SpeechBrainDiarizer

    d = SpeechBrainDiarizer(max_speakers=8)
    try:
        assert d._degraded is False, "worker failed to reach READY"
        low_a = d.assign(_tone(180.0), 16000, 0)
        high = d.assign(_tone(600.0), 16000, 1)
        low_b = d.assign(_tone(180.0), 16000, 2)
        assert low_a is not None and high is not None and low_b is not None
        assert low_a == low_b, "same tone must map to the same speaker id"
        assert low_a != high, "distinct tones must map to distinct ids"
    finally:
        d.close()
