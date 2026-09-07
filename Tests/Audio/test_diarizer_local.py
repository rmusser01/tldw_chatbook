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
from typing import Iterator, List

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.Audio import diarizer_local
from tldw_chatbook.Audio.diarizer_local import (
    COARSE_CRASHED,
    COARSE_UNAVAILABLE,
    DIARIZE_BUDGET_CEILING_S,
    DIARIZE_BUDGET_FLOOR_S,
    SpeechBrainDiarizer,
    diarize_budget_s,
)


@pytest.fixture
def captured_lines() -> Iterator[List[str]]:
    """Collect every loguru message emitted during the test.

    `caplog` does not see loguru's own sink -- mirrors the fixture of the
    same name in `test_meeting_diarization_session.py`.
    """
    lines: List[str] = []
    sink_id = loguru_logger.add(
        lambda message: lines.append(message.record["message"]),
        level="TRACE",
        format="{message}",
        diagnose=False,
    )
    try:
        yield lines
    finally:
        loguru_logger.remove(sink_id)

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


def test_restart_spawns_with_start_id_past_max_seen_and_records_crash_seq():
    """31749: the restarted worker must not re-mint ids the user may already
    have NAMED. It inherits the pre-crash high-water mark on its argv, and the
    backend remembers WHICH segment was in flight when the worker died so the
    Stop pass can leave everything before it alone."""
    cmds: list[list[str]] = []
    procs: list[FakeProc] = []

    def _spawn(cmd, *a, **k):
        cmds.append(list(cmd))
        proc = FakeProc(['{"id": "S2", "seq": 0}\n']) if not procs else FakeProc([_SEGMENTS_REPLY])
        procs.append(proc)
        return proc

    d = _ready(SpeechBrainDiarizer(spawn=_spawn))
    assert d.assign(_PCM, 16000, 0) == "S2"       # live id S2 -- may be named
    procs[0]._alive = False                        # ... then the worker dies
    assert d.assign(_PCM, 16000, 1) is None        # detected here -> one restart

    assert d.crashed_at_seq == 1
    assert d.max_id_seen == 2
    assert "--start-id" not in cmds[0]              # the first worker starts at 0
    assert cmds[1][cmds[1].index("--start-id") + 1] == "2"


def test_crash_seq_keeps_the_first_crash(monkeypatch):
    """The restart budget is one, but `crashed_at_seq` must stay pinned to the
    FIRST death -- that is where the pre-crash id space ends."""
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: FakeProc([])))
    d._fail(seq=3)
    d._fail(seq=9)
    assert d.crashed_at_seq == 3


def test_no_crash_leaves_crash_seq_none():
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: FakeProc(['{"id": "S1", "seq": 0}\n'])))
    assert d.assign(_PCM, 16000, 0) == "S1"
    assert d.crashed_at_seq is None and d.max_id_seen == 1


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


def test_batch_mint_starts_past_the_post_crash_start_id():
    """31749, second half: after a crash the restarted worker holds NO live
    centroids (the rest of the meeting is coarse), so the Stop pass mints from
    scratch -- straight onto the pre-crash "S1" the user may have named. The
    mint has to continue past the inherited start id too."""
    import numpy as np

    from tldw_chatbook.Audio.diarizer_worker import _reconcile_windows

    embs = [np.array([1.0, 0.0], np.float32), np.array([0.9, 0.1], np.float32)]
    spans = [(0.0, 1.5), (1.5, 3.0)]
    out = _reconcile_windows(spans, embs, {}, lambda x, n: np.zeros(len(x)), start_id=4)
    assert [s["speaker"] for s in out] == ["S5", "S5"]


def test_reconcile_windows_folds_a_surplus_final_onto_one_live_id_as_a_weighted_mean():
    """Review round 1, Important 2: `reconcile()`'s surplus rule (a leftover
    final cluster within threshold of an ALREADY-matched live id is the batch
    pass over-splitting one person, not a second person -- diarizer_cluster.py)
    means two final clusters reconciling onto the same live id is the designed
    common case, not a corner case. `out_centroids[live_id]` used to keep
    whichever centroid came last in file order and threw the other away; it
    must be the seconds-weighted mean of both, unit-normalised."""
    import numpy as np

    from tldw_chatbook.Audio.diarizer_worker import _reconcile_windows

    live = {"S1": np.array([1.0, 0.0, 0.0], np.float32)}
    embs = [np.array([1.0, 0.0, 0.0], np.float32), np.array([0.99, 0.14, 0.0], np.float32)]
    spans = [(0.0, 1.5), (1.5, 3.0)]  # equal-length windows -> equal weight
    out_centroids: dict = {}

    _reconcile_windows(spans, embs, live, lambda x, n: np.array([0, 1]), out_centroids=out_centroids)

    expected = np.array([0.995, 0.07, 0.0], np.float32)
    expected = (expected / np.linalg.norm(expected)).tolist()
    assert set(out_centroids) == {"S1"}
    assert out_centroids["S1"] == pytest.approx(expected, abs=1e-3)


def test_close_is_best_effort_and_idempotent():
    proc = FakeProc(['{"id": "S1"}\n'])
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: proc))
    d.close()
    d.close()  # second call must not raise
    assert d.assign(_PCM, 16000, 0) is None  # closed -> coarse


# --- 31744: forward pin() to the worker's live clusterer --------------------

def test_pin_sends_a_pin_command_to_the_worker():
    proc = FakeProc(['{"id": "S1", "seq": 0}\n'])
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: proc))
    assert d.assign(_PCM, 16000, 0) == "S1"
    d.pin("S1")
    assert any(b'"cmd": "pin"' in chunk and b'"S1"' in chunk for chunk in proc.stdin.chunks)


def test_pin_never_waits_for_an_in_flight_assign():
    """Final review I1: `assign` holds the backend lock across `_await_reply`
    (up to the assign budget) and `pin` runs on the APP thread, from the
    Meetings screen's `Input.Submitted` handler -- so a blocking acquire froze
    the whole TUI for as long as the window in flight took to give up."""
    release = threading.Event()

    class _Silent:
        """A stdout that never answers and never EOFs (so no crash sentinel)."""

        def readline(self) -> bytes:
            release.wait(5.0)
            return b""

    proc = FakeProc([])
    proc.stdout = _Silent()
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: proc, assign_budget_s=1.0))

    assigning = threading.Thread(target=d.assign, args=(_PCM, 16000, 0), daemon=True)
    assigning.start()
    deadline = time.monotonic() + 2.0
    while not d._lock.locked() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert d._lock.locked(), "the assign under test never took the backend lock"

    t0 = time.monotonic()
    d.pin("S1")  # must not raise, must not wait out the assign budget
    elapsed = time.monotonic() - t0

    assert elapsed < 0.5, f"pin blocked the app thread for {elapsed:.2f}s"
    assigning.join(3.0)
    release.set()


def test_pin_is_a_noop_when_coarse_only():
    proc = FakeProc([])
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: proc))
    d._mark_coarse("backend crashed")
    d.pin("S1")  # must not raise, must not write
    assert not any(b'"cmd": "pin"' in c for c in proc.stdin.chunks)


# --- 31826 task 3: backend enrolls after READY, fixes the first self match --

def test_enroll_is_sent_after_ready_and_after_restart():
    """Controller ruling 1 / fix-round-1 ruling 5: pin the ORDERING at the
    wire level, not just "the bytes eventually showed up". The stdin double
    records `wait_ready(0)` at the instant it sees the enroll line -- a
    mutation that sent enroll AFTER `ready.set()` (the exact ruling-1
    violation, review finding I4) makes this False -> True and fails."""
    made: list[FakeProc] = []
    order: list[bool] = []
    holder: dict = {}

    class _OrderingPipe(_Pipe):
        def write(self, data: bytes) -> int:
            if b'"cmd": "enroll"' in data:
                order.append(holder["d"].wait_ready(0))
            return super().write(data)

    gates = [threading.Event(), threading.Event()]

    def _spawn(*a, **k):
        p = FakeProc(['{"id": "S1", "seq": 0, "self": false}\n'])
        p.stdin = _OrderingPipe()
        p.stderr = _GatedStderr(gates[len(made)])
        made.append(p)
        return p

    d = SpeechBrainDiarizer(spawn=_spawn, voiceprint=[1.0, 0.0], match_threshold=0.2)
    holder["d"] = d
    gates[0].set()                                       # let the first worker report READY
    assert d.wait_ready(2.0) is True

    assert d.assign(_PCM, 16000, 0) == "S1"
    made[0]._alive = False                               # worker dies
    assert d.assign(_PCM, 16000, 1) is None              # detected -> restart
    assert len(made) == 2
    gates[1].set()                                       # let the restarted worker report READY
    assert d.wait_ready(2.0) is True

    assert order == [False, False]                       # enrolled BEFORE the gate opened, both workers


def test_failed_enroll_send_goes_through_the_existing_failure_path():
    """Fix-round-1 ruling 6 (overrides the review's `enroll_ok` attribute
    suggestion): a broken pipe on the enroll send is a worker failure, not a
    silently-ignored one. It must mark coarse and spend the one restart
    budget, same as any other dead-worker detection -- never leave a "ready
    but self-matching is silently off" state."""
    made: list[FakeProc] = []

    class _RaisingPipe(_Pipe):
        def write(self, data: bytes) -> int:
            if b'"cmd": "enroll"' in data:
                raise OSError("broken pipe")
            return super().write(data)

    def _spawn(*a, **k):
        p = FakeProc([])
        p.stdin = _RaisingPipe()
        made.append(p)
        return p

    d = SpeechBrainDiarizer(spawn=_spawn, voiceprint=[1.0, 0.0])
    # Poll for the terminal state directly rather than through `wait_ready`:
    # the restart reassigns `self._ready` to a fresh Event mid-flight (on the
    # first watcher thread, inside its own `_send_enroll`/`_fail` call), so
    # which generation's Event `wait_ready` happens to observe is a race --
    # it must not gate this assertion.
    deadline = time.monotonic() + 2.0
    while d._degraded is False and time.monotonic() < deadline:
        time.sleep(0.005)

    assert d.coarse_reason == COARSE_CRASHED  # the failure was recorded
    assert len(made) == 2                     # exactly one restart was spent
    assert d._degraded is True                # the second worker's enroll also failed (same pipe)


def test_first_self_flag_is_fixed_and_later_ones_counted():
    proc = FakeProc(['{"id": "S1", "seq": 0, "self": true}\n', '{"id": "S2", "seq": 1, "self": true}\n'])
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: proc, voiceprint=[1.0, 0.0]))
    d.assign(_PCM, 16000, 0)
    d.assign(_PCM, 16000, 1)
    assert d.self_cluster_id == "S1" and d.self_candidates_seen == 2


def test_self_candidates_seen_and_cluster_id_default_to_none_and_zero():
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: FakeProc(['{"id": "S1", "seq": 0, "self": false}\n'])))
    d.assign(_PCM, 16000, 0)
    assert d.self_cluster_id is None and d.self_candidates_seen == 0


def test_voiceprint_property_returns_a_copy_or_none():
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: FakeProc([]), voiceprint=[1.0, 0.0]))
    vp = d.voiceprint
    assert vp == [1.0, 0.0]
    vp.append(9.0)
    assert d.voiceprint == [1.0, 0.0]  # mutating the returned list must not leak back

    d2 = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: FakeProc([])))
    assert d2.voiceprint is None


def test_vector_never_reaches_logs(captured_lines):
    """Must actually REACH `_send_enroll`'s except branch -- the only log
    statement in the file that touches the vector (review finding I4's test
    quality note): the old version never made the enroll write fail, so it
    stayed green even if that line interpolated the vector itself."""
    class _RaisingPipe(_Pipe):
        def write(self, data: bytes) -> int:
            if b'"cmd": "enroll"' in data:
                raise OSError("broken pipe")
            return super().write(data)

    proc = FakeProc([])
    proc.stdin = _RaisingPipe()
    d = SpeechBrainDiarizer(spawn=lambda *a, **k: proc, voiceprint=[0.123456, 0.654321])
    assert d.wait_ready(2.0) is True
    joined = "\n".join(captured_lines)
    assert any("enroll send failed" in line for line in captured_lines)  # the log line ran
    assert "0.123456" not in joined
    assert "0.654321" not in joined


# --- 31826 task 3: stop_self from the diarize reply -------------------------

def test_stop_self_is_set_from_a_diarize_reply():
    reply = json.dumps({"segments": [], "self": "S1"}) + "\n"
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: FakeProc([reply])))
    d.diarize(Path("mixed.wav"), 0.0, 3.0)
    assert d.stop_self == "S1"


def test_stop_self_defaults_to_none_and_stays_none_when_reply_lacks_it():
    reply = json.dumps({"segments": []}) + "\n"
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: FakeProc([reply])))
    assert d.stop_self is None
    d.diarize(Path("mixed.wav"), 0.0, 3.0)
    assert d.stop_self is None


# --- 31826 task 3: export_centroid / enroll_from_pcm -------------------------
# Every canned reply below carries the "op_id" the backend will actually
# send (1 for the first centroid op on a fresh instance, 2 for the second,
# ...) -- fix-round-1 Critical 1's request correlation, verified rather than
# bypassed: a reply missing/mismatching op_id is now dropped as stale, so a
# reply without one would time the call out instead of answering it.

def test_export_centroid_returns_the_tuple_on_a_good_reply():
    proc = FakeProc(['{"centroid": [0.6, 0.8], "seconds": 12.5, "op_id": 1}\n'])
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: proc))
    assert d.export_centroid("S1") == ([0.6, 0.8], 12.5)
    sent = b"".join(proc.stdin.chunks)
    assert b'"cmd": "export_centroid"' in sent and b'"S1"' in sent


def test_export_centroid_returns_none_on_a_null_centroid():
    proc = FakeProc(['{"centroid": null, "op_id": 1}\n'])
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: proc))
    assert d.export_centroid("S1") is None


def test_export_centroid_returns_none_on_timeout(monkeypatch):
    """Fix-round-1 ruling 4 (closes review finding I3): the double must
    never EOF, or the call returns None via the crash path (`_fail`) rather
    than the budget -- vacuous against a dropped deadline. Release the
    reader at teardown (Minor 8) so no thread is left sleeping."""
    monkeypatch.setattr(diarizer_local, "CENTROID_BUDGET_S", 0.1)
    release = threading.Event()

    class _NeverAnswers:
        def readline(self) -> bytes:
            release.wait(5.0)
            return b""

    proc = FakeProc([])
    proc.stdout = _NeverAnswers()
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: proc))

    t0 = time.monotonic()
    assert d.export_centroid("S1") is None
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0                                     # bounded by the budget, not a sentinel
    assert d._degraded is False and d._coarse_only is False   # backpressure, not a crash
    release.set()


def test_export_centroid_is_none_when_degraded_or_coarse_only():
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: FakeProc([])))
    d._mark_coarse("backend crashed")
    assert d.export_centroid("S1") is None


def test_enroll_from_pcm_sends_n_prefixed_control_then_pcm_and_parses_reply():
    proc = FakeProc(['{"centroid": [1.0, 0.0], "seconds": 3.0, "op_id": 1}\n'])
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: proc))
    pcm = b"\x00\x00" * 1600

    assert d.enroll_from_pcm(pcm, 16000) == ([1.0, 0.0], 3.0)

    control = json.loads(proc.stdin.chunks[-2])
    assert control == {"cmd": "enroll_from_pcm", "sr": 16000, "n": len(pcm), "op_id": 1}
    assert proc.stdin.chunks[-1] == pcm


def test_enroll_from_pcm_returns_none_on_null_centroid():
    proc = FakeProc(['{"centroid": null, "op_id": 1}\n'])
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: proc))
    assert d.enroll_from_pcm(b"\x00\x00" * 1600, 16000) is None


def test_a_timed_out_export_centroid_does_not_leak_into_the_next_centroid_op(monkeypatch):
    """Critical 1, reproducing the review's PROBE E: a late reply for an
    abandoned op must not be delivered as the answer to the NEXT, unrelated
    centroid op -- in §3.4's flows that would persist another speaker's
    voice as the user's."""
    monkeypatch.setattr(diarizer_local, "CENTROID_BUDGET_S", 0.15)
    release = threading.Event()

    class _SlowThenAnswers:
        def __init__(self) -> None:
            self._i = 0

        def readline(self) -> bytes:
            self._i += 1
            if self._i == 1:
                release.wait(5.0)
                # The late reply for the FIRST (abandoned) op -- tagged with
                # its own op_id, which is no longer the one anyone is
                # waiting for by the time it arrives.
                return b'{"centroid": [1.0, 0.0], "seconds": 111.0, "op_id": 1}\n'
            if self._i == 2:
                return b'{"centroid": [0.0, 1.0], "seconds": 5.0, "op_id": 2}\n'
            return b""

    proc = FakeProc([])
    proc.stdout = _SlowThenAnswers()
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: proc))

    assert d.export_centroid("S1") is None      # times out at 0.15s; op_id 1 abandoned
    release.set()                                # now the stale op_id=1 reply lands on the queue
    # NOT S1's stale centroid -- the correct, correlated reply for THIS call.
    assert d.enroll_from_pcm(b"\x00\x00" * 1600, 16000) == ([0.0, 1.0], 5.0)


def test_assign_does_not_block_behind_an_in_flight_centroid_op():
    """Important 2 (I2): `_centroid_op` can hold the single backend lock for
    up to CENTROID_BUDGET_S (5x assign's own budget); assign must not block
    behind it -- a busy backend is backpressure (coarse window), not a hang
    on the transcript thread. Mirrors `test_pin_never_waits_for_an_in_flight_assign`."""
    release = threading.Event()

    class _NeverAnswers:
        def readline(self) -> bytes:
            release.wait(5.0)
            return b""

    proc = FakeProc([])
    proc.stdout = _NeverAnswers()
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: proc, assign_budget_s=0.1))

    op_thread = threading.Thread(target=d.export_centroid, args=("S1",), daemon=True)
    op_thread.start()
    deadline = time.monotonic() + 2.0
    while not d._lock.locked() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert d._lock.locked(), "the centroid op under test never took the backend lock"

    t0 = time.monotonic()
    assert d.assign(_PCM, 16000, 0) is None
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"assign blocked the transcript thread for {elapsed:.2f}s"

    release.set()
    op_thread.join(3.0)


def test_centroid_op_is_ready_gated_without_taking_the_lock():
    """Ruling 2 (closes review finding I1): a pre-READY centroid op must
    never take `self._lock` -- that lock is also what `_send_enroll` takes,
    so a pre-READY caller blocking on it would delay `ready.set()` behind an
    unrelated op (up to CENTROID_BUDGET_S)."""
    gate = threading.Event()
    proc = FakeProc([])
    proc.stderr = _GatedStderr(gate)  # never reports READY during this test
    d = SpeechBrainDiarizer(spawn=lambda *a, **k: proc, voiceprint=[1.0, 0.0])

    assert d.wait_ready(0.05) is False           # confirm: genuinely not ready yet
    assert d.export_centroid("S1") is None       # must return at once, not hang
    assert not d._lock.locked()                  # ... and never touched the lock

    gate.set()  # let the watcher (and its pending `_send_enroll`) proceed


# --- Qodo Q2: the WORKER's own command loop, torch-free --------------------
# The tests above prove the app side puts a `pin` line on the pipe. Nothing
# proved the worker on the other end acts on it: the dispatch used to sit
# inside `main()`, behind the ECAPA load, so only a real subprocess could
# reach it. `serve()` is that loop split out (`main()` calls it with the
# encoder-bound callables), so the real protocol -- readline framing,
# length-prefixed PCM, command dispatch -- runs here against a fake encoder.


def _serve_script(*lines: bytes) -> "io.BytesIO":
    """A worker stdin holding `lines` verbatim, then a `close`."""
    import io

    return io.BytesIO(b"".join(lines) + b'{"cmd": "close"}\n')


def _drive_worker(script, live):
    """Run `serve` over `script` with a fake encoder; return the JSON replies."""
    import io

    import numpy as np

    from tldw_chatbook.Audio.diarizer_worker import serve

    vectors = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ]
    handed: list[bytes] = []

    def fake_embed(pcm: bytes):
        handed.append(pcm)
        return vectors[min(len(handed), len(vectors)) - 1]

    stdout = io.BytesIO()
    assert serve(script, stdout, live, fake_embed, lambda *a, **k: ([], {})) == 0
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()], handed


def _serve_lines(lines, embed, batch=None):
    """Drive `serve()` over raw wire lines; return the parsed JSON replies."""
    import io

    from tldw_chatbook.Audio.diarizer_worker import serve
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer

    stdin = io.BytesIO(b"".join(lines)); stdout = io.BytesIO()
    serve(stdin, stdout, OnlineClusterer(max_speakers=8), embed, batch or (lambda *a, **k: ([], {})))
    return [json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()]


def test_self_flag_requires_threshold_and_min_seconds():
    pcm = b"\x00\x00" * 16000  # 1 s
    def ctl(d): return (json.dumps(d) + "\n").encode()
    lines = [ctl({"cmd": "enroll", "vector": [1, 0, 0], "threshold": 0.2, "min_seconds": 2.0})]
    for seq in range(3):
        lines += [ctl({"cmd": "assign", "sr": 16000, "seq": seq, "n": len(pcm)}), pcm]
    out = _serve_lines(lines, embed=lambda pcm: [0.99, 0.01, 0.0])
    assert [o["self"] for o in out] == [False, True, True]   # 1 s < 2 s, then 2 s, 3 s

    # Review round 1, Important 5: the threshold arm above was never
    # exercised (the stub returns a vector inside the gate on every call) --
    # deleting the distance check from `serve()` would still pass it. An
    # orthogonal embedding, well past `min_seconds`, must read False too.
    far_lines = [ctl({"cmd": "enroll", "vector": [1, 0, 0], "threshold": 0.2, "min_seconds": 2.0}),
                 ctl({"cmd": "assign", "sr": 16000, "seq": 0, "n": len(pcm)}), pcm,
                 ctl({"cmd": "assign", "sr": 16000, "seq": 1, "n": len(pcm)}), pcm]
    far_out = _serve_lines(far_lines, embed=lambda pcm: [0.0, 1.0, 0.0])
    assert far_out[-1]["self"] is False   # 2 s >= min_seconds, but outside the threshold


def test_diarize_self_matches_nearest_batch_centroid_and_export_centroid_prefers_it():
    """Review round 1, Important 6: the whole `diarize` -> `self` path,
    including the `(segments, {live_id: centroid})` batch contract and
    `export_centroid`'s post-diarize preference, had no test at all."""
    def ctl(d): return (json.dumps(d) + "\n").encode()
    segs = [{"start_s": 0.0, "end_s": 2.0, "speaker": "S1"},
            {"start_s": 2.0, "end_s": 5.0, "speaker": "S2"}]

    def fake_batch(*_a, **_k):
        return segs, {"S1": [1.0, 0.0, 0.0], "S2": [0.0, 1.0, 0.0]}

    lines = [
        ctl({"cmd": "enroll", "vector": [1, 0, 0], "threshold": 0.2, "min_seconds": 0.0}),
        ctl({"cmd": "diarize", "wav": "mixed.wav", "start": 0.0, "end": 5.0}),
        ctl({"cmd": "export_centroid", "id": "S1"}),
    ]
    out = _serve_lines(lines, embed=lambda pcm: [0.0, 0.0, 0.0], batch=fake_batch)

    assert out[0]["self"] == "S1"                                   # nearest within threshold
    assert out[1]["centroid"] == pytest.approx([1.0, 0.0, 0.0])      # the BATCH centroid ...
    assert out[1]["seconds"] == pytest.approx(2.0)                  # ... with seconds summed from its segments


def test_diarize_self_is_null_when_no_batch_centroid_is_within_threshold():
    def ctl(d): return (json.dumps(d) + "\n").encode()
    segs = [{"start_s": 0.0, "end_s": 2.0, "speaker": "S1"}]

    def fake_batch(*_a, **_k):
        return segs, {"S1": [0.0, 1.0, 0.0]}   # orthogonal to the enrolled vector

    lines = [
        ctl({"cmd": "enroll", "vector": [1, 0, 0], "threshold": 0.2, "min_seconds": 0.0}),
        ctl({"cmd": "diarize", "wav": "mixed.wav", "start": 0.0, "end": 2.0}),
    ]
    out = _serve_lines(lines, embed=lambda pcm: [0.0, 0.0, 0.0], batch=fake_batch)
    assert out[0]["self"] is None


def test_a_malformed_enroll_does_not_kill_the_worker():
    """Important 4 (review round 1): every op shares one framed error path,
    so a malformed `enroll` (e.g. from a corrupt or wrongly-decrypted
    voiceprint file) reports `ERROR enroll <type>` on stderr and the loop
    keeps serving -- it used to crash `serve()` unhandled (PROBE3)."""
    def ctl(d): return (json.dumps(d) + "\n").encode()
    lines = [ctl({"cmd": "enroll", "vector": "not-a-list", "threshold": 0.2}),
             ctl({"cmd": "assign", "sr": 16000, "seq": 0, "n": 4}), b"aaaa"]
    out = _serve_lines(lines, embed=lambda pcm: [1.0, 0.0, 0.0])
    assert out[-1]["id"] == "S1"   # the loop survived the bad enroll and kept serving


def test_enroll_from_pcm_returns_unit_centroid_and_export_centroid_roundtrip():
    pcm = b"\x00\x00" * 16000 * 3
    def ctl(d): return (json.dumps(d) + "\n").encode()
    out = _serve_lines([ctl({"cmd": "enroll_from_pcm", "sr": 16000, "n": len(pcm)}), pcm,
                        ctl({"cmd": "assign", "sr": 16000, "seq": 0, "n": len(pcm)}), pcm,
                        ctl({"cmd": "export_centroid", "id": "S1"})],
                       embed=lambda pcm: [3.0, 4.0, 0.0])
    assert out[0]["centroid"] == pytest.approx([0.6, 0.8, 0.0]) and out[0]["seconds"] == pytest.approx(3.0)
    assert out[2]["centroid"] == pytest.approx([0.6, 0.8, 0.0]) and out[2]["seconds"] == pytest.approx(3.0)


def test_worker_echoes_op_id_on_both_centroid_ops_including_the_error_fallback():
    """Fix-round-1 Critical 1: the worker must echo `op_id` on the success
    reply AND the framed-error fallback for both `export_centroid` and
    `enroll_from_pcm` -- the backend's request correlation depends on it
    being present on every reply shape, not just the happy path."""
    pcm = b"\x00\x00" * 1600

    def ctl(d):
        return (json.dumps(d) + "\n").encode()

    # Happy path: both ops echo the op_id they were sent.
    out = _serve_lines(
        [
            ctl({"cmd": "enroll_from_pcm", "sr": 16000, "n": len(pcm), "op_id": 7}), pcm,
            ctl({"cmd": "export_centroid", "id": "S1", "op_id": 8}),
        ],
        embed=lambda pcm: [1.0, 0.0, 0.0],
    )
    assert out[0]["op_id"] == 7
    assert out[1]["op_id"] == 8

    # Framed-error fallback: a malformed export_centroid still echoes op_id.
    def _raising_embed(pcm):
        raise RuntimeError("boom")

    err_out = _serve_lines(
        [ctl({"cmd": "enroll_from_pcm", "sr": 16000, "n": len(pcm), "op_id": 9}), pcm],
        embed=_raising_embed,
    )
    assert err_out[0] == {"centroid": None, "op_id": 9}


def test_enroll_from_pcm_reports_null_centroid_for_a_zero_magnitude_embedding():
    """Minor 9 (review round 1): a zero embedding must not come back as a
    live-looking centroid the owner would happily persist as a dead
    voiceprint (`_cos` -> 0.0 similarity -> distance 1.0 against everything)."""
    pcm = b"\x00\x00" * 16000
    def ctl(d): return (json.dumps(d) + "\n").encode()
    out = _serve_lines([ctl({"cmd": "enroll_from_pcm", "sr": 16000, "n": len(pcm)}), pcm],
                       embed=lambda pcm: [0.0, 0.0, 0.0])
    assert out[0] == {"centroid": None, "op_id": None}  # op_id echoed even when absent from the control line


def test_worker_command_loop_pins_the_cluster_it_is_told_to():
    """A pinned cluster's centroid is never moved by a fold at the speaker
    cap -- so the centroid standing still after a second, unrelated voice is
    proof the `pin` line reached the live clusterer. The control below is the
    identical script WITHOUT the pin: there the fold averages it away."""
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer

    assign_a = b'{"cmd": "assign", "seq": 0, "n": 4}\n' + b"aaaa"
    assign_b = b'{"cmd": "assign", "seq": 1, "n": 4}\n' + b"bbbb"

    pinned = OnlineClusterer(threshold=0.01, max_speakers=1)
    replies, handed = _drive_worker(
        _serve_script(assign_a, b'{"cmd": "pin", "id": "S1"}\n', assign_b), pinned
    )

    # The reply framing is unchanged: one line per assign, `seq` echoed, and
    # `pin` answers nothing at all (three commands in, two replies out).
    assert replies == [{"id": "S1", "seq": 0, "self": False}, {"id": "S1", "seq": 1, "self": False}]
    assert handed == [b"aaaa", b"bbbb"]      # PCM read by the control line's `n`
    assert list(pinned.centroids()["S1"]) == [1.0, 0.0]

    control = OnlineClusterer(threshold=0.01, max_speakers=1)
    _drive_worker(_serve_script(assign_a, assign_b), control)
    assert list(control.centroids()["S1"]) == [0.5, 0.5]   # folded, as expected


def test_worker_command_loop_ignores_a_garbled_line_and_an_unknown_command():
    """Protocol robustness the loop already had, now actually exercised: a
    truncated control line must not kill the meeting's worker."""
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer

    live = OnlineClusterer(threshold=0.01, max_speakers=2)
    replies, _handed = _drive_worker(
        _serve_script(
            b"not json at all\n",
            b'{"cmd": "nonsense"}\n',
            b'{"cmd": "assign", "seq": 7, "n": 4}\n' + b"aaaa",
        ),
        live,
    )
    assert replies == [{"id": "S1", "seq": 7, "self": False}]
