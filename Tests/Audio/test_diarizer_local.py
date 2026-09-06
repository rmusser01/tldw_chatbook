"""App-side diarizer backend: protocol parse + crash/degradation rule.

No torch here -- a FAKE subprocess is injected via `spawn`, so these run
everywhere. The real worker is exercised only by the opt-in
`test_diarizer_helper_real.py`.
"""
from __future__ import annotations

import json

from tldw_chatbook.Audio.diarizer_local import SpeechBrainDiarizer


class _Pipe:
    """A stdin double: swallows the control line + PCM the backend writes."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> int:
        self.chunks.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _Lines:
    """A byte stream whose ``readline`` yields the given lines then EOF."""

    def __init__(self, lines) -> None:
        self._lines = [l.encode() if isinstance(l, str) else l for l in lines]
        self._i = 0

    def readline(self) -> bytes:
        if self._i >= len(self._lines):
            return b""
        line = self._lines[self._i]
        self._i += 1
        return line


class FakeProc:
    """A subprocess double: one JSON line on stdout per ``assign``.

    ``READY`` is emitted once on stderr so the backend's warm-up handshake
    completes; further stderr reads hit EOF.
    """

    def __init__(self, replies, *, ready: bool = True) -> None:
        self.stdin = _Pipe()
        self.stdout = _Lines(replies)
        self.stderr = _Lines(["READY\n"] if ready else [])
        self._alive = True
        self.terminated = False

    def poll(self):
        return None if self._alive else 0

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def wait(self, timeout=None) -> int:
        self._alive = False
        return 0


# --- the two required cases (verbatim from the brief) ----------------------

def test_assign_parses_worker_reply():
    d = SpeechBrainDiarizer(spawn=lambda *a, **k: FakeProc(['{"id": "S1"}\n']))
    assert d.assign(b"\x00\x00" * 1600, 16000, 0) == "S1"


def test_crash_then_coarse_returns_none_for_the_rest():
    proc = FakeProc([])  # dies immediately
    d = SpeechBrainDiarizer(spawn=lambda *a, **k: proc)
    proc._alive = False
    assert d.assign(b"\x00\x00" * 1600, 16000, 0) is None  # crash -> coarse
    assert d.assign(b"\x00\x00" * 1600, 16000, 1) is None  # stays coarse


# --- the state machine the crash rule turns on -----------------------------

def test_null_id_is_coarse_for_the_window_but_worker_stays_up():
    # A healthy worker that could not place one window returns {"id": null};
    # that is coarse for THAT window only, not a permanent degrade.
    d = SpeechBrainDiarizer(
        spawn=lambda *a, **k: FakeProc(['{"id": null}\n', '{"id": "S1"}\n'])
    )
    assert d.assign(b"\x00\x00" * 1600, 16000, 0) is None
    assert d.assign(b"\x00\x00" * 1600, 16000, 1) == "S1"
    assert d._degraded is False


def test_restart_happens_exactly_once():
    procs = iter([
        FakeProc([]),                       # dies after start
        FakeProc(['{"id": "S1"}\n']),       # the single restart, healthy
    ])
    made: list[FakeProc] = []

    def _spawn(*a, **k):
        p = next(procs)
        made.append(p)
        return p

    d = SpeechBrainDiarizer(spawn=_spawn)
    made[0]._alive = False                  # first worker crashes
    assert d.assign(b"\x00\x00" * 1600, 16000, 0) is None   # detects, restarts
    assert d.assign(b"\x00\x00" * 1600, 16000, 1) == "S1"   # restart worked
    made[1]._alive = False                  # second worker crashes too
    assert d.assign(b"\x00\x00" * 1600, 16000, 2) is None   # no 2nd restart
    assert d._degraded is True
    assert len(made) == 2                   # exactly one restart spawned


def test_read_timeout_returns_none():
    class _Blocks:
        def readline(self):
            import threading
            threading.Event().wait()  # never returns
            return b""

    proc = FakeProc([])
    proc.stdout = _Blocks()
    # A tiny budget so the test does not actually wait; restart yields the
    # same blocked proc, which then degrades.
    d = SpeechBrainDiarizer(spawn=lambda *a, **k: proc, assign_budget_s=0.05)
    assert d.assign(b"\x00\x00" * 1600, 16000, 0) is None


def test_diarize_parses_segments():
    reply = json.dumps({"segments": [
        {"start_s": 0.0, "end_s": 1.5, "speaker": "S1"},
        {"start_s": 1.5, "end_s": 3.0, "speaker": "S2"},
    ]}) + "\n"
    d = SpeechBrainDiarizer(spawn=lambda *a, **k: FakeProc([reply]))
    from pathlib import Path
    segs = d.diarize(Path("/tmp/mixed.wav"), 0.0, 3.0)
    assert [s.speaker for s in segs] == ["S1", "S2"]
    assert segs[0].start_s == 0.0 and segs[1].end_s == 3.0


def test_batch_uses_agglomerative_pass_with_live_count_and_reconciles():
    # The torch-free Stop-pass seam: cluster window embeddings with the
    # injected agglomerative pass, then reconcile final labels -> live ids.
    import numpy as np

    from tldw_chatbook.Audio.diarizer_worker import _reconcile_windows

    live = {"S1": np.array([1.0, 0.0], np.float32), "S2": np.array([0.0, 1.0], np.float32)}
    embs = [
        np.array([1.0, 0.05], np.float32), np.array([0.05, 1.0], np.float32),
        np.array([1.0, 0.0], np.float32), np.array([0.0, 1.0], np.float32),
    ]
    spans = [(0.0, 1.5), (1.5, 3.0), (3.0, 4.5), (4.5, 6.0)]
    seen: dict = {}

    def fake_cluster(x, n):
        seen["n"] = n
        seen["rows"] = x.shape[0]
        return np.array([0, 1, 0, 1])  # two final clusters

    out = _reconcile_windows(spans, embs, live, 8, fake_cluster)
    assert seen["n"] == 2       # min(len(live)=2, max_speakers=8, len(embs)=4)
    assert seen["rows"] == 4
    # final label 0 (near S1) -> S1; final label 1 (near S2) -> S2
    assert [s["speaker"] for s in out] == ["S1", "S2", "S1", "S2"]


def test_batch_skips_clustering_for_a_single_live_speaker():
    import numpy as np

    from tldw_chatbook.Audio.diarizer_worker import _reconcile_windows

    live = {"S1": np.array([1.0, 0.0], np.float32)}
    embs = [np.array([1.0, 0.0], np.float32), np.array([0.9, 0.1], np.float32)]
    spans = [(0.0, 1.5), (1.5, 3.0)]
    called = {"v": False}

    def fake_cluster(x, n):
        called["v"] = True
        return np.zeros(len(x))

    out = _reconcile_windows(spans, embs, live, 8, fake_cluster)
    assert called["v"] is False                       # 1 speaker -> no clustering
    assert [s["speaker"] for s in out] == ["S1", "S1"]


def test_close_is_best_effort_and_idempotent():
    proc = FakeProc(['{"id": "S1"}\n'])
    d = SpeechBrainDiarizer(spawn=lambda *a, **k: proc)
    d.close()
    d.close()  # second call must not raise
    assert d.assign(b"\x00\x00" * 1600, 16000, 0) is None  # closed -> coarse
