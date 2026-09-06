"""App-side diarizer backend: protocol parse + crash/degradation rule.

No torch here -- a FAKE subprocess is injected via `spawn`, so these run
everywhere. The real worker is exercised only by the opt-in
`test_diarizer_helper_real.py`.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from tldw_chatbook.Audio import diarizer_local
from tldw_chatbook.Audio.diarizer_local import (
    COARSE_UNAVAILABLE,
    DIARIZE_BUDGET_CEILING_S,
    DIARIZE_BUDGET_FLOOR_S,
    SpeechBrainDiarizer,
    diarize_budget_s,
)

_PCM = b"\x00\x00" * 1600
_SEGMENTS_REPLY = json.dumps({"segments": [{"start_s": 0.0, "end_s": 1.5, "speaker": "S1"}]}) + "\n"


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


class _GatedStderr:
    """A stderr double whose ``READY`` only arrives once `gate` is set."""

    def __init__(self, gate: threading.Event) -> None:
        self._gate = gate
        self._sent = False

    def readline(self) -> bytes:
        if self._sent:
            return b""
        self._gate.wait()
        self._sent = True
        return b"READY\n"


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
        self.killed = False

    def poll(self):
        return None if self._alive else 0

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False

    def wait(self, timeout=None) -> int:
        self._alive = False
        return 0


def _ready(diarizer: SpeechBrainDiarizer) -> SpeechBrainDiarizer:
    """Warm-up is asynchronous now (C1); tests that need a warm worker wait."""
    assert diarizer.wait_ready(2.0) is True
    return diarizer


# --- the two required cases (verbatim from the brief) ----------------------

def test_assign_parses_worker_reply():
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: FakeProc(['{"id": "S1"}\n'])))
    assert d.assign(_PCM, 16000, 0) == "S1"


def test_crash_then_coarse_returns_none_for_the_rest():
    proc = FakeProc([])  # dies immediately
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: proc))
    proc._alive = False
    assert d.assign(_PCM, 16000, 0) is None  # crash -> coarse
    assert d.assign(_PCM, 16000, 1) is None  # stays coarse


# --- C1: construction must never block on the cold model download ----------

def test_construction_is_non_blocking_and_assign_is_coarse_until_ready():
    """Fix C1: `build_diarizer` runs inside the owner lock just before
    `session.start()`, so a blocking warm-up meant Start recorded nothing for
    up to READY_TIMEOUT_S. Construction returns at once; the first windows are
    coarse; once READY lands, ids flow."""
    gate = threading.Event()
    proc = FakeProc(['{"id": "S1"}\n'])
    proc.stderr = _GatedStderr(gate)

    t0 = time.monotonic()
    d = SpeechBrainDiarizer(spawn=lambda *a, **k: proc)
    assert time.monotonic() - t0 < 0.5          # returned without waiting
    assert d.wait_ready(0.05) is False          # ... and it really is not warm
    assert d.assign(_PCM, 16000, 0) is None     # warming -> coarse, no wait

    gate.set()
    assert d.wait_ready(2.0) is True
    assert d.assign(_PCM, 16000, 1) == "S1"


def test_diarize_gives_up_on_a_worker_that_never_warms_and_records_why(monkeypatch):
    """Re-review item 1: the warm-up wait is bounded by READY_TIMEOUT_S, NOT by
    the clamped batch budget (which would let the Stop pass stall for 2x it),
    and a worker that spawned but never reached READY is the spec §7 "failed to
    become ready" case -- it has to reach the footer, not fail silently."""
    monkeypatch.setattr(diarizer_local, "READY_TIMEOUT_S", 0.05)
    proc = FakeProc([_SEGMENTS_REPLY])
    proc.stderr = _GatedStderr(threading.Event())   # a gate nobody ever opens

    d = SpeechBrainDiarizer(spawn=lambda *a, **k: proc)
    t0 = time.monotonic()
    assert d.diarize(Path("mixed.wav"), 0.0, 3.0) == []
    elapsed = time.monotonic() - t0

    assert elapsed < 5.0                            # bounded by READY_TIMEOUT_S...
    assert elapsed < DIARIZE_BUDGET_FLOOR_S         # ... not by the 60 s budget
    assert d.coarse_reason == COARSE_UNAVAILABLE    # ... and the footer can say so


def test_diarize_waits_for_a_late_ready():
    """The Stop pass is the only caller allowed to wait for warm-up."""
    gate = threading.Event()
    proc = FakeProc([_SEGMENTS_REPLY])
    proc.stderr = _GatedStderr(gate)
    d = SpeechBrainDiarizer(spawn=lambda *a, **k: proc)
    threading.Timer(0.05, gate.set).start()
    segs = d.diarize(Path("mixed.wav"), 0.0, 3.0)
    assert [s.speaker for s in segs] == ["S1"]


# --- the state machine the crash rule turns on -----------------------------

def test_null_id_is_coarse_for_the_window_but_worker_stays_up():
    # A healthy worker that could not place one window returns {"id": null};
    # that is coarse for THAT window only, not a permanent degrade.
    d = _ready(SpeechBrainDiarizer(
        spawn=lambda *a, **k: FakeProc(['{"id": null}\n', '{"id": "S1"}\n'])
    ))
    assert d.assign(_PCM, 16000, 0) is None
    assert d.assign(_PCM, 16000, 1) == "S1"
    assert d._degraded is False


def test_restart_happens_once_and_live_labels_stay_coarse_after_it():
    """Qodo Q10: the restarted worker's clusterer starts at `_n=0`, so its
    "S1" is a DIFFERENT person than the first worker's "S1" -- and would
    inherit that speaker's user-assigned name. Spec §7: the rest of the
    meeting is coarse; the restart exists only so the Stop pass survives."""
    procs = iter([
        FakeProc([]),                  # dies after start
        FakeProc([_SEGMENTS_REPLY]),   # the single restart, healthy
    ])
    made: list[FakeProc] = []

    def _spawn(*a, **k):
        p = next(procs)
        made.append(p)
        return p

    d = _ready(SpeechBrainDiarizer(spawn=_spawn))
    made[0]._alive = False                                  # first worker crashes
    assert d.assign(_PCM, 16000, 0) is None                 # detects, restarts
    assert len(made) == 2                                   # exactly one restart
    assert d.wait_ready(2.0) is True
    assert d.assign(_PCM, 16000, 1) is None                 # coarse for the rest
    assert d.coarse_reason == "backend crashed"
    # ... but the authoritative Stop pass still runs on the restarted worker.
    segs = d.diarize(Path("mixed.wav"), 0.0, 3.0)
    assert [s.speaker for s in segs] == ["S1"]


def test_assign_timeout_is_a_skip_not_a_crash():
    """Fix I1: one slow reply used to call `_fail()`, burning the restart
    budget and blocking the transcript thread behind a fresh warm-up. Spec
    §6.3 makes it backpressure: coarse window, worker untouched."""
    slow = threading.Event()

    class _SlowThenAnswers:
        def __init__(self):
            self._i = 0

        def readline(self) -> bytes:
            self._i += 1
            if self._i == 1:
                slow.wait(2.0)              # the first window's reply is late
                return b'{"id": "S1", "seq": 0}\n'
            if self._i == 2:
                return b'{"id": "S2", "seq": 1}\n'
            return b""

    made: list[FakeProc] = []

    def _spawn(*a, **k):
        p = FakeProc([])
        p.stdout = _SlowThenAnswers()
        made.append(p)
        return p

    d = _ready(SpeechBrainDiarizer(spawn=_spawn, assign_budget_s=0.05))
    assert d.assign(_PCM, 16000, 0) is None     # over budget -> coarse window
    assert len(made) == 1                       # no restart
    assert d._degraded is False and d._coarse_only is False
    slow.set()
    # The late reply for seq 0 must not be handed to seq 1 (it is discarded
    # by seq), and the worker keeps serving.
    assert d.assign(_PCM, 16000, 1) == "S2"


def test_kill_escalates_to_sigkill(monkeypatch):
    """Qodo Q14: a worker that ignores terminate() must not survive with its
    model (and accelerator memory) while `_fail` spawns a replacement."""

    class _Stubborn(FakeProc):
        def terminate(self) -> None:
            self.terminated = True          # ... and keeps running

        def wait(self, timeout=None):
            if not self.killed:
                raise TimeoutError("still alive")
            return 0

    proc = _Stubborn([])
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: proc))
    d._kill(proc)
    assert proc.terminated is True and proc.killed is True


def test_diarize_budget_scales_with_the_recording():
    """Qodo Q13: a fixed 60 s silently lost the Stop pass on long meetings."""
    assert diarize_budget_s(5.0) == DIARIZE_BUDGET_FLOOR_S       # short -> floor
    assert diarize_budget_s(300.0) == 300.0                      # ~1s per second
    assert diarize_budget_s(99999.0) == DIARIZE_BUDGET_CEILING_S  # bounded
    assert diarize_budget_s(None) == DIARIZE_BUDGET_FLOOR_S       # junk -> floor


def test_diarize_parses_segments():
    reply = json.dumps({"segments": [
        {"start_s": 0.0, "end_s": 1.5, "speaker": "S1"},
        {"start_s": 1.5, "end_s": 3.0, "speaker": "S2"},
    ]}) + "\n"
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: FakeProc([reply])))
    segs = d.diarize(Path("/tmp/mixed.wav"), 0.0, 3.0)
    assert [s.speaker for s in segs] == ["S1", "S2"]
    assert segs[0].start_s == 0.0 and segs[1].end_s == 3.0


def test_batch_lets_the_clusterer_choose_the_speaker_count():
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

    out = _reconcile_windows(spans, embs, live, fake_cluster)
    assert seen["n"] is None    # the service estimates it, bounded by its config
    assert seen["rows"] == 4
    # final label 0 (near S1) -> S1; final label 1 (near S2) -> S2
    assert [s["speaker"] for s in out] == ["S1", "S2", "S1", "S2"]


def test_batch_can_find_speakers_the_live_pass_missed():
    """Qodo Q11: the batch count used to be capped at `len(live_centroids)`,
    so a live pass that clustered one speaker (or was backpressured into
    finding none) forced the WHOLE recording into one cluster -- exactly the
    under-clustering the authoritative Stop pass exists to correct."""
    import numpy as np

    from tldw_chatbook.Audio.diarizer_worker import _reconcile_windows

    live = {"S1": np.array([1.0, 0.0], np.float32)}          # only one live cluster
    embs = [
        np.array([1.0, 0.0], np.float32), np.array([0.0, 1.0], np.float32),
        np.array([0.95, 0.05], np.float32), np.array([0.05, 0.95], np.float32),
    ]
    spans = [(0.0, 1.5), (1.5, 3.0), (3.0, 4.5), (4.5, 6.0)]
    called = {"n": "unset"}

    def fake_cluster(x, n):
        called["n"] = n
        return np.array([0, 1, 0, 1])  # the batch really finds two speakers

    out = _reconcile_windows(spans, embs, live, fake_cluster)
    assert called["n"] is None                       # clustering was NOT skipped
    speakers = [s["speaker"] for s in out]
    assert speakers[0] == "S1"                       # matched to the live cluster
    assert len(set(speakers)) == 2                   # ... and the missed one is kept
    assert "S2" in speakers                          # minted as a live-style id


def test_batch_mints_a_live_style_id_when_there_are_no_live_clusters():
    # Backpressure degraded near-live labelling the whole meeting -> no live
    # centroids -> the Stop pass is the only labeller. Its cluster must NOT
    # surface as "Speaker F0" (final whole-branch review I2): mint an "S" id.
    import numpy as np

    from tldw_chatbook.Audio.diarizer_worker import _reconcile_windows

    embs = [np.array([1.0, 0.0], np.float32), np.array([0.9, 0.1], np.float32)]
    spans = [(0.0, 1.5), (1.5, 3.0)]
    out = _reconcile_windows(spans, embs, {}, lambda x, n: np.zeros(len(x)))
    assert [s["speaker"] for s in out] == ["S1", "S1"]
    assert not any(s["speaker"].startswith("F") for s in out)


def test_close_is_best_effort_and_idempotent():
    proc = FakeProc(['{"id": "S1"}\n'])
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: proc))
    d.close()
    d.close()  # second call must not raise
    assert d.assign(_PCM, 16000, 0) is None  # closed -> coarse
