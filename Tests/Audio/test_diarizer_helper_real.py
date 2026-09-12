"""Opt-in: spawns the REAL diarizer worker, per engine. Never in CI.

Run: TLDW_RUN_DIARIZER_TEST=1 pytest Tests/Audio/test_diarizer_helper_real.py -p no:cacheprovider

Feeds two distinct synthetic tones through the real subprocess and asserts the
worker returns two stable, distinct cluster ids -- a genuine end-to-end check
of spawn, READY, embedding, and clustering.

Both engines run (task 8: 31827): `speechbrain` needs torch, `onnx` needs
sherpa-onnx; each skips itself when its packages are missing. The ONNX run
uses `$TLDW_DIARIZER_MODELS_DIR` when set and otherwise downloads the models
into `tmp_path` -- never the user's data dir. Measured on the default
embedder (titanet_small): the two tones separate at the shipped 0.25 live
threshold, as they do under ECAPA. They do NOT separate under `eres2net_en`
or `campplus_en`, which is why this test does not sweep embedders -- tones
are not speech, and only the bake-off corpus can rank them.
"""
from __future__ import annotations

import math
import os
import struct

import pytest

from Tests.Audio.conftest import real_engine_available, real_engine_kwargs

pytestmark = [pytest.mark.integration]


def _tone(freq: float, seconds: float = 2.0, sr: int = 16000) -> bytes:
    """A mono PCM16 sine tone of the given frequency."""
    frames = int(seconds * sr)
    return b"".join(
        struct.pack("<h", int(0.5 * 32767 * math.sin(2 * math.pi * freq * i / sr)))
        for i in range(frames)
    )


@pytest.mark.parametrize("engine", ["speechbrain", "onnx"])
def test_real_worker_assigns_two_stable_ids(engine, tmp_path):
    if os.environ.get("TLDW_RUN_DIARIZER_TEST") != "1":
        pytest.skip("opt-in: set TLDW_RUN_DIARIZER_TEST=1")
    if not real_engine_available(engine):
        pytest.skip(f"opt-in: {engine} packages not installed")

    from tldw_chatbook.Audio.diarizer_local import READY_TIMEOUT_S, LocalDiarizer

    d = LocalDiarizer(engine, max_speakers=8, **real_engine_kwargs(engine, tmp_path))
    try:
        assert d.wait_ready(READY_TIMEOUT_S), f"worker failed to reach READY ({d.warmup_status})"
        assert d._degraded is False, "worker failed to reach READY"
        low_a = d.assign(_tone(180.0), 16000, 0)
        high = d.assign(_tone(600.0), 16000, 1)
        low_b = d.assign(_tone(180.0), 16000, 2)
        assert low_a is not None and high is not None and low_b is not None
        assert low_a == low_b, "same tone must map to the same speaker id"
        assert low_a != high, "distinct tones must map to distinct ids"
    finally:
        d.close()
