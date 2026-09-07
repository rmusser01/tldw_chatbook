# Meetings Near-Live Diarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give a meeting real per-speaker labels, produced near-live during recording and finalized on Stop, with a cross-platform local diarizer and per-meeting speaker renaming.

**Architecture:** A per-meeting `Diarizer` (existing seam in `meeting_session.py`) with a local SpeechBrain backend that embeds each finalized segment's PCM and assigns a stable cluster id via a new pure-numpy online clusterer, all inside a subprocess for GIL/crash/memory isolation. The session swaps its energy labeller for the diarizer when configured; names key on cluster ids in a per-meeting map; the Stop batch pass reconciles and finalizes.

**Tech Stack:** Python ≥3.11, numpy, SpeechBrain ECAPA (existing `diarization` extra: torch/torchaudio/speechbrain/scikit-learn), Textual 8.x, pytest with `--import-mode=importlib`.

**Spec:** `Docs/superpowers/specs/2026-09-05-meeting-diarization-design.md`

## Global Constraints

- Diarization is best-effort: every failure falls back to phase-1 You/Others labels and NEVER blocks the recording, the transcript, or Library ingest.
- No new dependency; the local engine reuses the existing `diarization` extra (torch, torchaudio, speechbrain, scikit-learn).
- `import tldw_chatbook.app` must still import no torch and no diarizer module at boot; the UI-ready module census must not rise (`Tests/Performance/test_ui_ready_module_census.py`).
- No transcript text and no speaker names in persistent logs (`Utils/log_sanitizer.redact_user_paths`; log lengths/ids only).
- Near-live is opt-in: `[meetings] live_diarization` defaults to `false`, leaving phase-1 behavior untouched.
- The online clusterer is pure numpy, no torch, importable and unit-testable without spawning the subprocess.
- Never hold the `MeetingSession` lock across blocking diarizer I/O (phase-1 C2 lesson).
- Run pytest from the worktree with `VIRTUAL_ENV=.venv`; use `.venv/bin/python -m pytest ... -p no:cacheprovider`.
- Commit trailer on every commit: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## File Structure

- Create `tldw_chatbook/Audio/diarizer_cluster.py` — pure-numpy `OnlineClusterer` (centroids, `assign`, sticky rename pin, `max_speakers` fold) and `reconcile(live_centroids, final_labels)`.
- Create `tldw_chatbook/Audio/diarizer_worker.py` — subprocess entry point: reads length-prefixed PCM + commands on stdin, embeds via SpeechBrain, runs `OnlineClusterer`, writes cluster ids on stdout.
- Create `tldw_chatbook/Audio/diarizer_local.py` — `SpeechBrainDiarizer`: app-side handle implementing the `Diarizer` protocol over the worker subprocess; owns spawn/restart/close and the coarse-after-crash rule.
- Modify `tldw_chatbook/Audio/meeting_session.py` — extend `Diarizer` protocol (`assign`, `close`), add `speaker_id` to `MeetingSegment`, per-meeting name map, diarizer swap on segment finalize, Stop reconciliation, versioned `meeting.json`/`transcript.jsonl`, name rendering.
- Modify `tldw_chatbook/Audio/meeting_capture.py` — bounded recent-PCM ring per channel + `pcm_window(source, start_s, end_s) -> bytes`.
- Modify `tldw_chatbook/Audio/meeting_owner.py` — new `MeetingSettings` fields, backend factory, live-implies-Stop-pass, extended diarization readout.
- Modify `tldw_chatbook/UI/Screens/meetings_screen.py` — speaker legend + live inline rename.
- Modify `tldw_chatbook/Widgets/Library/library_media_canvas.py` — after-the-fact speaker rename over `Media.content` (+ FTS reindex + versioned `Transcripts` write).
- Modify `tldw_chatbook/config.py` — new `[meetings]` keys.
- Modify `Docs/User_Guide/meetings.md` — document live speaker labels and renaming.
- Tests under `Tests/Audio/` and `Tests/UI/`.

---

### Task 1: Online clusterer (pure numpy)

**Files:**
- Create: `tldw_chatbook/Audio/diarizer_cluster.py`
- Test: `Tests/Audio/test_diarizer_cluster.py`

**Interfaces:**
- Consumes: nothing (numpy only).
- Produces: `OnlineClusterer(threshold: float = 0.25, max_speakers: int = 8)` with `assign(embedding: np.ndarray) -> str` returning stable ids `"S1"`, `"S2"`, …; `pin(cluster_id: str) -> None` marking a cluster sticky; `centroids() -> dict[str, np.ndarray]`; and module function `reconcile(live_centroids: dict[str, np.ndarray], final: list[tuple[str, np.ndarray]]) -> dict[str, str]` mapping final cluster ids to live ids by nearest cosine centroid.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Audio/test_diarizer_cluster.py
import numpy as np
from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer, reconcile

def _v(*xs): return np.array(xs, dtype=np.float32)

def test_two_distinct_voices_get_two_stable_ids():
    c = OnlineClusterer(threshold=0.25, max_speakers=8)
    a1 = c.assign(_v(1, 0, 0)); b1 = c.assign(_v(0, 1, 0))
    a2 = c.assign(_v(0.95, 0.05, 0)); b2 = c.assign(_v(0.02, 0.98, 0))
    assert a1 == a2 and b1 == b2 and a1 != b1

def test_cap_folds_extra_speaker_into_nearest():
    c = OnlineClusterer(threshold=0.01, max_speakers=2)
    c.assign(_v(1, 0, 0)); c.assign(_v(0, 1, 0))
    third = c.assign(_v(0, 0, 1))
    assert third in {"S1", "S2"}  # folded, no S3

def test_pinned_cluster_is_never_merged_away():
    c = OnlineClusterer(threshold=0.9, max_speakers=8)
    a = c.assign(_v(1, 0, 0)); c.pin(a)
    # a near-identical later vector must still resolve to the pinned id
    assert c.assign(_v(0.99, 0.01, 0)) == a

def test_reconcile_maps_final_to_live_by_nearest_centroid():
    live = {"S1": _v(1, 0, 0), "S2": _v(0, 1, 0)}
    final = [("F0", _v(0, 0.9, 0)), ("F1", _v(0.9, 0, 0))]
    assert reconcile(live, final) == {"F0": "S2", "F1": "S1"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Audio/test_diarizer_cluster.py -q -p no:cacheprovider`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# tldw_chatbook/Audio/diarizer_cluster.py
"""Incremental speaker clustering over voice embeddings (pure numpy, no torch)."""
from __future__ import annotations
import numpy as np

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

class OnlineClusterer:
    """Assign each embedding to the nearest centroid within `threshold` cosine
    distance, else start a new cluster up to `max_speakers`; past the cap, fold
    into the nearest. A pinned cluster is never dropped."""
    def __init__(self, threshold: float = 0.25, max_speakers: int = 8) -> None:
        self._threshold = threshold
        self._max = max_speakers
        self._centroids: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}
        self._pinned: set[str] = set()
        self._n = 0

    def assign(self, embedding: np.ndarray) -> str:
        emb = np.asarray(embedding, dtype=np.float32)
        best_id, best_sim = None, -1.0
        for cid, cen in self._centroids.items():
            sim = _cos(emb, cen)
            if sim > best_sim:
                best_id, best_sim = cid, sim
        near_enough = best_id is not None and (1.0 - best_sim) <= self._threshold
        if not near_enough and len(self._centroids) < self._max:
            self._n += 1
            cid = f"S{self._n}"
            self._centroids[cid] = emb.copy()
            self._counts[cid] = 1
            return cid
        cid = best_id  # nearest existing (also the cap-fold path)
        n = self._counts[cid] + 1
        self._centroids[cid] = (self._centroids[cid] * self._counts[cid] + emb) / n
        self._counts[cid] = n
        return cid

    def pin(self, cluster_id: str) -> None:
        self._pinned.add(cluster_id)

    def centroids(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._centroids.items()}

def reconcile(live_centroids: dict[str, np.ndarray],
              final: list[tuple[str, np.ndarray]]) -> dict[str, str]:
    """Map each final cluster id to the nearest live cluster id by cosine."""
    out: dict[str, str] = {}
    for fid, fcen in final:
        best_id, best_sim = None, -1.0
        for lid, lcen in live_centroids.items():
            sim = _cos(np.asarray(fcen, np.float32), np.asarray(lcen, np.float32))
            if sim > best_sim:
                best_id, best_sim = lid, sim
        if best_id is not None:
            out[fid] = best_id
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Audio/test_diarizer_cluster.py -q -p no:cacheprovider`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Audio/diarizer_cluster.py Tests/Audio/test_diarizer_cluster.py
git commit -m "feat(meetings): pure-numpy online speaker clusterer"
```

---

### Task 2: Segment speaker id, per-meeting name map, versioned meeting files

**Files:**
- Modify: `tldw_chatbook/Audio/meeting_session.py` (`MeetingSegment`, `MeetingMeta`, `write_meeting_json`/`read_meeting_json`, a new `render_label`)
- Test: `Tests/Audio/test_meeting_speaker_model.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `MeetingSegment` gains `speaker_id: str | None = None`; `MeetingMeta` gains `speaker_names: dict[str, str] = {}` and `format_version: int = 2`; `render_label(segment, names: dict[str, str], user_display_name: str) -> str | None` returns the display string (channel + person) or None; `read_meeting_json` back-fills `format_version=1` recordings with empty `speaker_names` and `speaker_id=None`.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Audio/test_meeting_speaker_model.py
from tldw_chatbook.Audio.meeting_session import (
    MeetingSegment, render_label, write_meeting_json, read_meeting_json,
)

def _seg(**kw):
    base = dict(seq=0, t_audio_start=0.0, t_audio_end=1.0, t_wall_start=0.0,
                t_wall_end=1.0, label="others", text="hi")
    base.update(kw); return MeetingSegment(**base)

def test_render_uses_the_name_map_by_cluster_id():
    seg = _seg(label="others", speaker_id="S2")
    assert render_label(seg, {"S2": "Alice"}, "Me") == "Alice"

def test_render_falls_back_to_generic_speaker_when_unnamed():
    seg = _seg(label="others", speaker_id="S2")
    assert render_label(seg, {}, "Me") == "Speaker 2"

def test_you_channel_renders_the_user_display_name():
    seg = _seg(label="you", speaker_id=None)
    assert render_label(seg, {}, "Me") == "Me"

def test_old_meeting_json_backfills_speaker_fields(tmp_path):
    write_meeting_json(tmp_path, {"mode": "call", "format_version": 1})
    payload = read_meeting_json(tmp_path)
    assert payload["format_version"] == 1
    assert payload.get("speaker_names", {}) == {}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Audio/test_meeting_speaker_model.py -q -p no:cacheprovider`
Expected: FAIL (`speaker_id` unexpected / `render_label` missing).

- [ ] **Step 3: Implement**

Add `speaker_id: str | None = None` to `MeetingSegment` (after `label`). Add `speaker_names: dict = field(default_factory=dict)` and `format_version: int = 2` to `MeetingMeta`. In `read_meeting_json`, after loading, `payload.setdefault("speaker_names", {})`. Add:

```python
# tldw_chatbook/Audio/meeting_session.py
def render_label(segment, names: dict[str, str], user_display_name: str) -> str | None:
    """Display name for a segment: the user for the mic channel, else the
    named or generic speaker; None when the segment has no label (room mode
    pre-diarization) or is overlap-coarse."""
    if segment.label == "you":
        return user_display_name
    if segment.speaker_id:
        if segment.speaker_id in names:
            return names[segment.speaker_id]
        n = segment.speaker_id[1:] if segment.speaker_id.startswith("S") else segment.speaker_id
        return f"Speaker {n}"
    if segment.label in ("others", "both"):
        return "Others"
    return None
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Audio/test_meeting_speaker_model.py -q -p no:cacheprovider`
Expected: PASS. Also run `.venv/bin/python -m pytest Tests/Audio/test_meeting_session.py -q -p no:cacheprovider` (phase-1 regression) — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Audio/meeting_session.py Tests/Audio/test_meeting_speaker_model.py
git commit -m "feat(meetings): speaker id on segments + per-meeting name map + json versioning"
```

---

### Task 3: Capture recent-PCM ring and window accessor

**Files:**
- Modify: `tldw_chatbook/Audio/meeting_capture.py`
- Test: `Tests/Audio/test_meeting_capture.py` (add cases)

**Interfaces:**
- Consumes: nothing new.
- Produces: `MeetingCapture.pcm_window(source: str, start_s: float, end_s: float) -> bytes` returning PCM16 mono 16 kHz bytes for `source` in `{"you","others","mixed"}` over `[start_s, end_s]`, clipped to what the bounded ring still holds (empty bytes when evicted). A per-source `deque`-backed ring holds the most recent `RING_SECONDS = 60` seconds.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Audio/test_meeting_capture.py (add)
def test_pcm_window_returns_recent_frames(meeting_capture_factory):
    cap = meeting_capture_factory()  # existing test factory / fake recorder
    cap._push_pcm("others", b"\x01\x00" * 16000)  # 1s of samples at t=[0,1)
    cap._push_pcm("others", b"\x02\x00" * 16000)  # 1s at t=[1,2)
    win = cap.pcm_window("others", 1.0, 2.0)
    assert len(win) == 16000 * 2 and win[:2] == b"\x02\x00"

def test_pcm_window_empty_when_evicted(meeting_capture_factory):
    cap = meeting_capture_factory()
    assert cap.pcm_window("others", 10_000.0, 10_001.0) == b""
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Audio/test_meeting_capture.py -q -p no:cacheprovider -k pcm_window`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add a per-source ring keyed by absolute sample index. On each frame written to a source's WAV writer, also append `(start_sample, bytes)` to `self._pcm_rings[source]` (a `collections.deque`) and drop entries older than `RING_SECONDS`. Implement:

```python
# tldw_chatbook/Audio/meeting_capture.py
RING_SECONDS = 60
SAMPLE_RATE = 16000
def pcm_window(self, source: str, start_s: float, end_s: float) -> bytes:
    ring = self._pcm_rings.get(source)
    if not ring:
        return b""
    a = int(start_s * SAMPLE_RATE); b = int(end_s * SAMPLE_RATE)
    out = bytearray()
    for start_sample, chunk in ring:
        n = len(chunk) // 2
        lo, hi = start_sample, start_sample + n
        if hi <= a or lo >= b:
            continue
        s = max(a, lo) - lo; e = min(b, hi) - lo
        out += chunk[s * 2:e * 2]
    return bytes(out)
```

Add `self._pcm_rings: dict[str, deque] = {"you": deque(), "others": deque(), "mixed": deque()}` in `__init__`, and a `_push_pcm(source, chunk)` helper called from the same place frames reach each writer, trimming by `RING_SECONDS * SAMPLE_RATE` samples.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Audio/test_meeting_capture.py -q -p no:cacheprovider`
Expected: PASS (all capture tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Audio/meeting_capture.py Tests/Audio/test_meeting_capture.py
git commit -m "feat(meetings): bounded recent-PCM ring + pcm_window accessor"
```

---

### Task 4: Diarizer protocol + session wiring with a fake backend

**Files:**
- Modify: `tldw_chatbook/Audio/meeting_session.py` (`Diarizer` protocol, `_on_final`, stop reconciliation)
- Test: `Tests/Audio/test_meeting_diarization_session.py`

**Interfaces:**
- Consumes: `OnlineClusterer.reconcile` (Task 1), `MeetingSegment.speaker_id`/`render_label` (Task 2), `MeetingCapture.pcm_window` (Task 3).
- Produces: extended `Diarizer` protocol — `assign(pcm: bytes, sample_rate: int, seq: int) -> str | None`, `diarize(wav_path, start_s, end_s) -> list[SpeakerSegment]` whose `SpeakerSegment.speaker` is **already the reconciled live cluster id** (the backend holds the live online centroids and the batch centroids and applies `reconcile` internally — the session never calls `reconcile` and never touches centroids), `centroids() -> dict[str, np.ndarray]`, `close() -> None`. `MeetingSession.__init__` gains `diarizer: Diarizer | None = None`. When set and `label == "others"` (call mode) or `label is None` (room mode), the session pulls the segment PCM and calls `assign` off the lock, sets `segment.speaker_id`, and re-emits the segment. On stop it calls `diarize` and overlays each returned segment's `speaker` onto the meeting segments whose time span it covers.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Audio/test_meeting_diarization_session.py
from tldw_chatbook.Audio.meeting_session import MeetingSession, SpeakerSegment

class FakeDiarizer:
    def __init__(self, ids): self._ids = list(ids); self.closed = False
    def assign(self, pcm, sample_rate, seq): return self._ids.pop(0) if self._ids else None
    def diarize(self, wav_path, start_s, end_s):
        return [SpeakerSegment(0.0, 1.0, "F0"), SpeakerSegment(1.0, 2.0, "F1")]
    def centroids(self): return {}
    def close(self): self.closed = True

def test_segment_gets_a_speaker_id_from_the_diarizer(meeting_session_with_fake_capture):
    session = meeting_session_with_fake_capture(diarizer=FakeDiarizer(["S1"]), mode="call")
    session.start()
    session._on_final_for_test("hello", label="others")  # test hook driving _on_final
    seg = session.segments[-1]
    assert seg.speaker_id == "S1"

def test_diarizer_closed_on_stop(meeting_session_with_fake_capture):
    fake = FakeDiarizer([])
    session = meeting_session_with_fake_capture(diarizer=fake, mode="call")
    session.start(); session.stop()
    assert fake.closed is True

def test_assign_not_called_under_the_session_lock(meeting_session_with_fake_capture):
    seen = {}
    class LockProbe(FakeDiarizer):
        def assign(self, pcm, sr, seq):
            seen["locked"] = session._lock_is_held_for_test()
            return "S1"
    session = meeting_session_with_fake_capture(diarizer=LockProbe(["S1"]), mode="call")
    session.start(); session._on_final_for_test("hi", label="others")
    assert seen["locked"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Audio/test_meeting_diarization_session.py -q -p no:cacheprovider`
Expected: FAIL.

- [ ] **Step 3: Implement**

Extend the `Diarizer` `Protocol` with `assign`, `centroids`, `close`. In `_on_final`, after the locked block appends the segment (keep that block unchanged), if `self._diarizer is not None` and the segment qualifies, compute PCM outside the lock and assign:

```python
# tldw_chatbook/Audio/meeting_session.py  (inside _on_final, after releasing the lock)
if self._diarizer is not None and segment is not None and segment.label in ("others", None):
    source = "others" if segment.label == "others" else "mixed"
    pcm = self.capture.pcm_window(source, segment.t_audio_start, segment.t_audio_end)
    sid = None
    if pcm:
        try:
            sid = self._diarizer.assign(pcm, 16000, segment.seq)  # OFF the lock
        except Exception as exc:  # best-effort
            logger.warning("meeting: diarizer assign failed ({})", type(exc).__name__)
    if sid is not None:
        with self._lock:
            segment.speaker_id = sid
        self._emit("segment", segment)
        self._each_sink("on_segment", segment)
```

In `stop()` (before finalizing the result), if a diarizer is set and `mixed.wav` exists, call `segments = self._diarizer.diarize(mixed_or_others_path, 0.0, duration)`; each returned `SpeakerSegment.speaker` is already a reconciled live id (§ backend does `reconcile` internally). Overlay them: for each meeting `MeetingSegment`, set `speaker_id` to the `SpeakerSegment.speaker` whose `[start_s, end_s]` covers the meeting segment's midpoint. Guard the whole block in try/except so a Stop-pass failure never blocks the result (log length/name-free, no text). Add the test hooks `_on_final_for_test` and `_lock_is_held_for_test` used above (thin wrappers, test-only but real code paths).

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Audio/test_meeting_diarization_session.py Tests/Audio/test_meeting_session.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Audio/meeting_session.py Tests/Audio/test_meeting_diarization_session.py
git commit -m "feat(meetings): diarizer seam + session near-live labelling and stop reconciliation"
```

---

### Task 5: SpeechBrain worker + subprocess backend

**Files:**
- Create: `tldw_chatbook/Audio/diarizer_worker.py`
- Create: `tldw_chatbook/Audio/diarizer_local.py`
- Test: `Tests/Audio/test_diarizer_local.py`, `Tests/Audio/test_diarizer_helper_real.py` (opt-in, gated)

**Interfaces:**
- Consumes: `OnlineClusterer` (Task 1), the `Diarizer` protocol (Task 4).
- Produces: `SpeechBrainDiarizer(max_speakers: int = 8, spawn=subprocess.Popen)` implementing `assign`/`diarize`/`centroids`/`close`. Wire protocol: newline-delimited JSON control + length-prefixed PCM on the worker's stdin; one line `{"id": "S1"}` or `{"id": null}` per `assign` on stdout; `READY` on stderr once the model is warm.

- [ ] **Step 1: Write the failing test (app-side protocol, no torch)**

```python
# Tests/Audio/test_diarizer_local.py
from tldw_chatbook.Audio.diarizer_local import SpeechBrainDiarizer

class FakeProc:
    def __init__(self, replies): self._replies = list(replies); self.stdin = _Pipe(); self.stdout = _Replies(self._replies); self.stderr = _Ready(); self._alive = True
    def poll(self): return None if self._alive else 0
    # ... minimal stdin/stdout doubles emitting one JSON line per assign

def test_assign_parses_worker_reply():
    d = SpeechBrainDiarizer(spawn=lambda *a, **k: FakeProc(['{"id": "S1"}\n']))
    assert d.assign(b"\x00\x00" * 1600, 16000, 0) == "S1"

def test_crash_then_coarse_returns_none_for_the_rest():
    proc = FakeProc([])  # dies immediately
    d = SpeechBrainDiarizer(spawn=lambda *a, **k: proc)
    proc._alive = False
    assert d.assign(b"\x00\x00" * 1600, 16000, 0) is None  # crash -> coarse
    assert d.assign(b"\x00\x00" * 1600, 16000, 1) is None  # stays coarse
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Audio/test_diarizer_local.py -q -p no:cacheprovider`
Expected: FAIL.

- [ ] **Step 3: Implement**

`diarizer_worker.py`: `main()` reads control lines, lazily imports SpeechBrain (`from tldw_chatbook.Local_Ingestion.diarization_service import _lazy_import_speechbrain`), loads the ECAPA encoder once, prints `READY` on stderr, then for each PCM window computes an embedding, feeds an `OnlineClusterer` (Task 1), and writes the id JSON to stdout. On a Stop/`diarize` command it clusters the whole file's embeddings (batch), computes final centroids, and applies `reconcile` (Task 1) against the live centroids it accumulated during the meeting, returning `SpeakerSegment`s whose `speaker` is the reconciled live id — so reconciliation lives in the worker, never in the session. `diarizer_local.py` imports no torch at module scope; the worker is a separate process. `meeting_owner.build_diarizer` (Task 6) imports `SpeechBrainDiarizer` lazily, inside the function, so nothing torch-adjacent is pulled in before a meeting starts. `diarizer_local.py`: spawn `[sys.executable, "-m", "tldw_chatbook.Audio.diarizer_worker"]` (with the frozen-app fallback resolved here — see spec §3.4; if spawn or `READY` fails, mark the backend degraded so `assign` returns `None`). Implement the crash rule: any `poll() is not None`, broken pipe, or read timeout marks the backend permanently degraded (`self._degraded = True`) and returns `None` thereafter (one restart attempt before degrading). `assign` is time-bounded: a read that exceeds `ASSIGN_BUDGET_S` returns `None`. Names and text never cross the pipe — only PCM and ids.

- [ ] **Step 4: Run to verify it passes + the gated real test**

Run: `.venv/bin/python -m pytest Tests/Audio/test_diarizer_local.py -q -p no:cacheprovider`
Expected: PASS.
Gated real test `Tests/Audio/test_diarizer_helper_real.py` mirrors `test_audiotap_helper_macos.py`: `@pytest.mark.skipif` unless `TLDW_RUN_DIARIZER_TEST=1` and the `diarization` extra is importable; it spawns the real worker, feeds two synthetic tones, and asserts two stable ids. Skipped by default.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Audio/diarizer_worker.py tldw_chatbook/Audio/diarizer_local.py Tests/Audio/test_diarizer_local.py Tests/Audio/test_diarizer_helper_real.py
git commit -m "feat(meetings): SpeechBrain diarizer subprocess worker + backend"
```

---

### Task 6: Owner config, backend factory, degradation

**Files:**
- Modify: `tldw_chatbook/Audio/meeting_owner.py` (`MeetingSettings`, `from_config`, backend factory, `prepare` readout)
- Test: `Tests/Audio/test_meeting_owner.py` (add cases)

**Interfaces:**
- Consumes: `SpeechBrainDiarizer` (Task 5).
- Produces: `MeetingSettings` gains `live_diarization: bool = False`, `diarizer_backend: str = "local"`, `max_speakers: int = 8`; the owner builds a diarizer only when `live_diarization` is on and modules are present, injects it into the session, and forces the Stop pass when live is on even if `post_diarize` is off.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Audio/test_meeting_owner.py (add)
def test_no_diarizer_built_when_live_off(tmp_path, monkeypatch):
    owner, _, _ = _owner(tmp_path, live_diarization=False)
    owner.prepare(); session = owner.start()
    assert session._diarizer is None

def test_diarizer_built_when_live_on_and_deps_present(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ())
    built = {}
    monkeypatch.setattr(mo, "build_diarizer", lambda settings: built.setdefault("d", object()))
    owner, _, _ = _owner(tmp_path, live_diarization=True)
    owner.prepare(); session = owner.start()
    assert session._diarizer is built["d"]

def test_live_on_missing_deps_falls_back_to_coarse(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ("torch",))
    owner, _, _ = _owner(tmp_path, live_diarization=True)
    owner.prepare(); session = owner.start()
    assert session._diarizer is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Audio/test_meeting_owner.py -q -p no:cacheprovider -k diariz`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add the three fields to `MeetingSettings` (pydantic) and to `from_config` (`get_setting("meetings", "live_diarization", False)` etc.). Add a module function `build_diarizer(settings) -> Diarizer | None` that returns `None` when `not settings.live_diarization` or `diarization_requirements()` is non-empty, else constructs `SpeechBrainDiarizer(max_speakers=settings.max_speakers)` for `diarizer_backend == "local"` (server raises `NotImplementedError`, caught → `None` + log). In `start()`, `diarizer = build_diarizer(self.settings)` and pass to `MeetingSession(...)`; if a diarizer is built, force the Stop pass regardless of `post_diarize`. Extend the `prepare` readout so `diarization_available`/`_missing` distinguishes live from offline.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Audio/test_meeting_owner.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Audio/meeting_owner.py Tests/Audio/test_meeting_owner.py
git commit -m "feat(meetings): owner config + diarizer backend factory + degradation"
```

---

### Task 7: Live speaker legend and inline rename

**Files:**
- Modify: `tldw_chatbook/UI/Screens/meetings_screen.py`
- Test: `Tests/UI/test_meetings_screen.py` (add cases)

**Interfaces:**
- Consumes: session segments carrying `speaker_id`; `render_label` (Task 2).
- Produces: a legend region listing seen speakers with an inline rename input; renaming updates `session.meta.speaker_names[cluster_id]`, calls the diarizer's `pin(cluster_id)` when available, persists via `update_meeting_json`, and re-renders visible lines. All worker callbacks stay `is_mounted`-guarded (phase-1 rule).

- [ ] **Step 1: Write the failing test**

```python
# Tests/UI/test_meetings_screen.py (add)
def test_rename_updates_map_and_rerenders(meetings_screen_with_session):
    screen = meetings_screen_with_session(segments=[("others", "S1", "hello")])
    screen._apply_rename("S1", "Alice")
    assert screen._session.meta.speaker_names["S1"] == "Alice"
    assert "Alice:" in screen._rendered_transcript_text()

def test_rename_pins_the_cluster_when_diarizer_present(meetings_screen_with_session):
    screen = meetings_screen_with_session(segments=[("others", "S1", "hi")], with_diarizer=True)
    screen._apply_rename("S1", "Bob")
    assert screen._session._diarizer.pinned == ["S1"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_meetings_screen.py -q -p no:cacheprovider -k rename`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add a `#meetings-speaker-legend` region to `compose`. Track seen `speaker_id`s from incoming segments; render one row per speaker (`render_label`) with a rename `Input`. On submit, call `_apply_rename(cluster_id, name)` which updates `meta.speaker_names`, calls `session._diarizer.pin(cluster_id)` if present, `update_meeting_json(folder, speaker_names=...)`, and refreshes the transcript log lines through `render_label`. Empty name removes the map entry. Guard every worker callback with `if not self.is_mounted: return`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/UI/test_meetings_screen.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/meetings_screen.py Tests/UI/test_meetings_screen.py
git commit -m "feat(meetings): live speaker legend and inline rename"
```

---

### Task 8: Rename speakers on the Library media item

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py`
- Test: `Tests/UI/test_library_media_speaker_rename.py`

**Interfaces:**
- Consumes: the meeting folder reachable from `Media.url` (its parent dir) holding `meeting.json` (name map) and `transcript.jsonl` (segments).
- Produces: a rename control on a media item that is a meeting recording; an edit updates the folder's name map, re-renders the transcript, and in one DB transaction rewrites `Media.content` (the displayed and FTS-searched field), reindexes `media_fts`, and writes a new versioned `Transcripts` row. Disabled when the folder is gone.

- [ ] **Step 1: Write the failing test**

```python
# Tests/UI/test_library_media_speaker_rename.py
def test_rename_after_rewrites_content_and_reindexes(tmp_media_db, meeting_folder_media_item):
    media_id, folder = meeting_folder_media_item(names={}, segments=[("S1", "hello")])
    from tldw_chatbook.Widgets.Library.library_media_canvas import rename_meeting_speaker
    rename_meeting_speaker(tmp_media_db, media_id, "S1", "Alice")
    row = tmp_media_db.get_media_by_id(media_id)
    assert "Alice:" in row["content"]
    hits = tmp_media_db.search_media_db(search_query="Alice")
    assert any(h["id"] == media_id for h in hits)

def test_rename_after_disabled_when_folder_gone(tmp_media_db, meeting_folder_media_item):
    media_id, folder = meeting_folder_media_item(names={}, segments=[("S1", "hi")])
    import shutil; shutil.rmtree(folder)
    from tldw_chatbook.Widgets.Library.library_media_canvas import can_rename_meeting_speakers
    assert can_rename_meeting_speakers(tmp_media_db, media_id) is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_library_media_speaker_rename.py -q -p no:cacheprovider`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add module functions `can_rename_meeting_speakers(db, media_id)` (True when the media url's parent holds `meeting.json`) and `rename_meeting_speaker(db, media_id, cluster_id, name)` which reads the folder's `transcript.jsonl` + `meeting.json`, updates the name map, re-renders the transcript text via `render_label`, and calls one DB transaction updating `Media.content` + FTS + a versioned `Transcripts` insert (reuse the DB's existing media-update path so sync metadata is set). Surface a rename control in the canvas only when `can_rename_meeting_speakers` is True.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/UI/test_library_media_speaker_rename.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Library/library_media_canvas.py Tests/UI/test_library_media_speaker_rename.py
git commit -m "feat(library): rename meeting speakers on the media item"
```

---

### Task 9: Config keys, docs, and boot/log invariants

**Files:**
- Modify: `tldw_chatbook/config.py` (`[meetings]` block)
- Modify: `Docs/User_Guide/meetings.md`
- Test: `Tests/Audio/test_meeting_import_safety.py` (extend), `Tests/Audio/test_meeting_diarization_session.py` (add a log-privacy case)

**Interfaces:**
- Consumes: everything above.
- Produces: documented, defaulted config; the boot-no-torch and no-names-in-logs invariants pinned.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Audio/test_meeting_import_safety.py (extend the existing subprocess check)
def test_app_import_pulls_in_no_diarizer_module():
    # run in the existing numpy-blocked subprocess harness
    assert "tldw_chatbook.Audio.diarizer_local" not in imported_modules
    assert "tldw_chatbook.Audio.diarizer_worker" not in imported_modules

# Tests/Audio/test_meeting_diarization_session.py (add)
def test_diarizer_failure_log_has_no_text_or_names(caplog, meeting_session_with_fake_capture):
    class Boom:
        def assign(self, *a): raise RuntimeError("secret meeting content")
        def diarize(self, *a): return []
        def centroids(self): return {}
        def close(self): pass
    session = meeting_session_with_fake_capture(diarizer=Boom(), mode="call")
    session.start(); session._on_final_for_test("secret words", label="others")
    assert "secret words" not in caplog.text and "secret meeting content" not in caplog.text
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Audio/test_meeting_import_safety.py Tests/Audio/test_meeting_diarization_session.py -q -p no:cacheprovider -k "import or log"`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add to the `[meetings]` block in `config.py`'s default TOML: `live_diarization = false`, `diarizer_backend = "local"`, `max_speakers = 8`, each with a comment. Make the diarizer `assign`-failure log print only `type(exc).__name__` (no exception message, no text). Document live labels and renaming in `meetings.md` (a "Speaker labels" subsection: near-live behavior, the legend, live and after-the-fact renaming, the opt-in switch, and the deps), and refresh its "Verified against" stamp.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Audio -q -p no:cacheprovider` and `.venv/bin/python -m pytest Tests/Performance/test_ui_ready_module_census.py -q -p no:cacheprovider`
Expected: PASS; census unchanged.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/config.py Docs/User_Guide/meetings.md Tests/Audio/test_meeting_import_safety.py Tests/Audio/test_meeting_diarization_session.py
git commit -m "feat(meetings): config keys, docs, and boot/log invariants for diarization"
```

---

## Self-Review

**Spec coverage:** §3.1 seam → Task 4; §3.2 backends → Tasks 5-6; §3.3 clusterer → Task 1; §3.4 subprocess → Task 5; §4 data flow (channels, rolling loop, overlap, reconciliation) → Tasks 3-4; §5 identity/rename live → Tasks 2, 7; §5.3 rename-after → Task 8; §6 config/deps/perf → Task 6, 9; §7 degradation → Tasks 5-6, 9; §8 testing → every task's tests plus the gated real test in Task 5. Covered.

**Placeholder scan:** no TBD/TODO; the frozen-app spawn detail is resolved in Task 5 with a documented fallback, not left open.

**Type consistency:** `assign(pcm, sample_rate, seq) -> str | None`, cluster ids `"S<n>"`, `speaker_id` on `MeetingSegment`, `speaker_names` on `MeetingMeta`, and `render_label(segment, names, user_display_name)` are used identically across Tasks 1, 2, 4, 5, 7, 8.
