# Persistent ingest job history — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Spec: `Docs/superpowers/specs/2026-07-12-library-persistent-job-history-design.md`. Branch `claude/followups-job-history` off dev `d8c9bec1`. Anchors exact at branch point; grep symbols, line numbers drift.

**Goal:** Persist the Library ingest job registry to SQLite so history survives restarts (and crashes), interrupted jobs come back as retryable FAILED, and each job carries a `retry_count`.

**Architecture:** A small `BaseDB`-subclass store (`LibraryIngestJobsDB`) with ONE persistent WAL connection (single-threaded UI access) persists every visible job. The registry gets an optional `store` hook and writes through on each mutation; a pure `plan_restore()` transforms the loaded rows (normalize interrupted → FAILED, prune to a cap) and the app seeds the registry from it in `on_mount`.

**Tech Stack:** Python ≥3.11, sqlite3, pytest.

## Global Constraints

- **No behavior change with `store=None`:** the registry with no store behaves exactly as today; every existing registry test stays green unchanged.
- **Persisted fields only round-trippable ones** — the `time.monotonic()` fields `submitted_at`/`started_at`/`finished_at` are NOT persisted (restored `submitted_at=0.0`, `started_at`/`finished_at=None`); only `finished_at_wall` (ISO) round-trips.
- **`state` column has `CHECK (state IN ('queued','parsing','writing','done','failed'))`.**
- **`retry_count`:** `LibraryIngestJob.retry_count: int = 0`; `submit` → 0; `requeue` → `source.retry_count + 1`; persisted; surfaced as `· retry {n}` (n>0) on the ingest job row. No cap / no auto-retry / no backoff.
- **Interrupted-on-restart:** QUEUED/PARSING/WRITING → `FAILED`, `error="Interrupted by app restart"`, `permanent=False`, `finished_at_wall=<restore time>`.
- **Prune cap:** `_MAX_PERSISTED_JOBS = 500` (keep most recent by `seq`).
- **Best-effort persistence:** store errors are caught + debug-logged; a mutation never fails on a store error. A corrupt store at load → start empty + warn.
- **Single-threaded store:** all registry mutations run on the UI thread (verified: writer marshals `mark_done`/`mark_failed` via `call_from_thread`), so the persistent connection needs no cross-thread handling.
- **Staging:** explicit paths only. Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Test command** (venv, isolated HOME):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
    /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> \
    -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```

---

### Task 1: `LibraryIngestJobsDB` store (new)

**Files:**
- Create: `tldw_chatbook/DB/Library_Ingest_Jobs_DB.py`
- Modify: `tldw_chatbook/Library/library_ingest_jobs.py` (add the one `retry_count` field the store persists — the rest of the retry/hook logic is Task 2)
- Test: `Tests/DB/test_library_ingest_jobs_db.py` (create)

**Interfaces:**
- Produces: `LibraryIngestJob.retry_count: int = 0` (dataclass field only); `LibraryIngestJobsDB(db_path, client_id="default")` with `upsert_job(job)`, `delete_job(job_id)`, `all_jobs() -> list[dict]` (seq-ascending), `close()`. Consumes `LibraryIngestJob` (reads its attributes) — imported from `tldw_chatbook.Library.library_ingest_jobs`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/DB/test_library_ingest_jobs_db.py`:
```python
import sqlite3
import pytest

from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJobRegistry, IngestJobState


def _db(tmp_path):
    return LibraryIngestJobsDB(tmp_path / "jobs.db")


def test_upsert_and_all_jobs_roundtrip_ordered(tmp_path):
    reg = LibraryIngestJobRegistry()
    j1 = reg.submit(source_path="/a.mp3", title="A", keywords=("k1", "k2"), detected_type="audio")
    j2 = reg.submit(source_path="/b.txt", title="B")
    db = _db(tmp_path)
    db.upsert_job(j1)
    db.upsert_job(j2)
    rows = db.all_jobs()
    assert [r["job_id"] for r in rows] == [j1.job_id, j2.job_id]     # seq order
    assert rows[0]["source_path"] == "/a.mp3" and rows[0]["detected_type"] == "audio"
    assert rows[0]["keywords"] == '["k1", "k2"]'
    assert rows[0]["state"] == "queued" and rows[0]["retry_count"] == 0
    db.close()


def test_upsert_is_idempotent_update_in_place(tmp_path):
    reg = LibraryIngestJobRegistry()
    j = reg.submit(source_path="/a.mp3")
    db = _db(tmp_path)
    db.upsert_job(j)
    reg.mark_parsing(j.job_id, detected_type="audio")
    db.upsert_job(reg.jobs()[0])          # same job_id, now PARSING
    rows = db.all_jobs()
    assert len(rows) == 1 and rows[0]["state"] == "parsing"
    db.close()


def test_delete_job(tmp_path):
    reg = LibraryIngestJobRegistry()
    j = reg.submit(source_path="/a.mp3")
    db = _db(tmp_path)
    db.upsert_job(j)
    db.delete_job(j.job_id)
    assert db.all_jobs() == []
    db.close()


def test_state_check_constraint_rejects_bad_state(tmp_path):
    db = _db(tmp_path)
    conn = db._get_connection()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ingest_jobs (seq, job_id, source_path, state) VALUES (1,'x','/p','bogus')"
        )
    db.close()
```

- [ ] **Step 2: Run to verify it fails**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/DB/test_library_ingest_jobs_db.py -q -p no:cacheprovider -o addopts="" --timeout=120
```
Expected: FAIL — `ModuleNotFoundError: tldw_chatbook.DB.Library_Ingest_Jobs_DB`.

- [ ] **Step 3: Implement the store**

FIRST add the `retry_count` field the store persists — in `tldw_chatbook/Library/library_ingest_jobs.py`, add `retry_count: int = 0` to the `LibraryIngestJob` dataclass (after `permanent`, keep it last for field-order back-compat). (Task 2 adds the requeue-carry + store hook; this task only needs the field to exist so `upsert_job` can read it.) Then create `tldw_chatbook/DB/Library_Ingest_Jobs_DB.py`:
```python
"""SQLite persistence for the Library ingest job registry.

Single-user, UI-thread-only: keeps ONE persistent WAL connection reused across
all reads/writes (safe because every registry mutation runs on the UI thread),
rather than opening/closing per operation.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Union

from loguru import logger

from .base_db import BaseDB


class LibraryIngestJobsDB(BaseDB):
    _CURRENT_SCHEMA_VERSION = 1

    def __init__(self, db_path: Union[str, Path], client_id: str = "default") -> None:
        self._conn: sqlite3.Connection | None = None
        super().__init__(db_path, client_id)  # calls _initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path_str, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                logger.opt(exception=True).debug("LibraryIngestJobsDB: close failed")
            finally:
                self._conn = None

    def _initialize_schema(self) -> None:
        conn = self._get_connection()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY NOT NULL);
            INSERT OR IGNORE INTO schema_version (version) VALUES (1);

            CREATE TABLE IF NOT EXISTS ingest_jobs (
                seq INTEGER PRIMARY KEY,
                job_id TEXT UNIQUE NOT NULL,
                source_path TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                author TEXT NOT NULL DEFAULT '',
                keywords TEXT NOT NULL DEFAULT '[]',
                perform_analysis INTEGER NOT NULL DEFAULT 0,
                chunk_enabled INTEGER NOT NULL DEFAULT 0,
                chunk_size INTEGER NOT NULL DEFAULT 0,
                state TEXT NOT NULL CHECK (state IN ('queued','parsing','writing','done','failed')),
                retry_count INTEGER NOT NULL DEFAULT 0,
                detected_type TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                finished_at_wall TEXT NOT NULL DEFAULT '',
                media_id INTEGER,
                superseded INTEGER NOT NULL DEFAULT 0,
                dismissed INTEGER NOT NULL DEFAULT 0,
                permanent INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.commit()

    @staticmethod
    def _seq_of(job_id: str) -> int:
        # "ingest-job-{n}" -> n
        return int(job_id.rsplit("-", 1)[-1])

    def upsert_job(self, job) -> None:
        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO ingest_jobs
              (seq, job_id, source_path, title, author, keywords, perform_analysis,
               chunk_enabled, chunk_size, state, retry_count, detected_type, error,
               finished_at_wall, media_id, superseded, dismissed, permanent)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(job_id) DO UPDATE SET
              source_path=excluded.source_path, title=excluded.title, author=excluded.author,
              keywords=excluded.keywords, perform_analysis=excluded.perform_analysis,
              chunk_enabled=excluded.chunk_enabled, chunk_size=excluded.chunk_size,
              state=excluded.state, retry_count=excluded.retry_count,
              detected_type=excluded.detected_type, error=excluded.error,
              finished_at_wall=excluded.finished_at_wall, media_id=excluded.media_id,
              superseded=excluded.superseded, dismissed=excluded.dismissed, permanent=excluded.permanent
            """,
            (
                self._seq_of(job.job_id), job.job_id, job.source_path, job.title, job.author,
                json.dumps(list(job.keywords)), int(job.perform_analysis), int(job.chunk_enabled),
                job.chunk_size, job.state.value, job.retry_count, job.detected_type, job.error,
                job.finished_at_wall, job.media_id, int(job.superseded), int(job.dismissed),
                int(job.permanent),
            ),
        )
        conn.commit()

    def delete_job(self, job_id: str) -> None:
        conn = self._get_connection()
        conn.execute("DELETE FROM ingest_jobs WHERE job_id = ?", (job_id,))
        conn.commit()

    def all_jobs(self) -> list[dict]:
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM ingest_jobs ORDER BY seq ASC").fetchall()
        return [dict(r) for r in rows]
```
- [ ] **Step 4: Run to verify it passes + registry suite (field addition is back-compat)**

Run the Step-1 store tests AND the existing registry suite (the new `retry_count` field defaults to 0, so no existing test changes):
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/DB/test_library_ingest_jobs_db.py Tests/Library/test_library_ingest_jobs.py \
  -q -p no:cacheprovider -o addopts="" --timeout=180
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/Library_Ingest_Jobs_DB.py tldw_chatbook/Library/library_ingest_jobs.py Tests/DB/test_library_ingest_jobs_db.py
git commit -m "feat(ingest): SQLite store for persistent ingest job history + retry_count field (161)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Registry — `retry_count`, store hook, restore, `plan_restore` (pure)

**Files:**
- Modify: `tldw_chatbook/Library/library_ingest_jobs.py`
- Test: `Tests/Library/test_library_ingest_jobs.py`

**Interfaces:**
- Produces:
  - `LibraryIngestJob.retry_count: int = 0` (new field).
  - `IngestJobStore` Protocol: `upsert_job(job)`, `delete_job(job_id)`.
  - `LibraryIngestJobRegistry.attach_store(store)`; per-mutation write-through; `restore(jobs, next_id)`.
  - module fn `plan_restore(rows: list[dict], *, max_persisted: int, now_iso: str) -> RestorePlan` where `RestorePlan` is a dataclass `(jobs: list[LibraryIngestJob], next_id: int, upsert: list[LibraryIngestJob], delete_ids: list[str])`.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Library/test_library_ingest_jobs.py`:
```python
class _FakeStore:
    def __init__(self):
        self.upserts = []
        self.deletes = []
    def upsert_job(self, job):
        self.upserts.append(job.job_id)
    def delete_job(self, job_id):
        self.deletes.append(job_id)


def test_submit_starts_retry_count_zero_and_requeue_increments():
    from tldw_chatbook.Library.library_ingest_jobs import IngestJobState
    reg = LibraryIngestJobRegistry()
    j = reg.submit(source_path="/a.mp3")
    assert j.retry_count == 0
    reg.mark_parsing(j.job_id); reg.mark_failed(j.job_id, error="boom")
    r = reg.requeue(j.job_id)
    assert r.retry_count == 1
    reg.mark_parsing(r.job_id); reg.mark_failed(r.job_id, error="boom2")
    r2 = reg.requeue(r.job_id)
    assert r2.retry_count == 2


def test_store_hook_writes_through_on_mutations():
    store = _FakeStore()
    reg = LibraryIngestJobRegistry()
    reg.attach_store(store)
    j = reg.submit(source_path="/a.mp3")
    reg.mark_parsing(j.job_id)
    assert store.upserts.count(j.job_id) >= 2       # submit + mark_parsing
    reg.mark_done(j.job_id, media_id=1)
    reg.clear_finished()
    assert j.job_id in store.deletes


def test_store_hook_none_is_pure_and_errors_swallowed():
    reg = LibraryIngestJobRegistry()          # no store
    reg.submit(source_path="/a.mp3")          # must not raise
    class _Boom:
        def upsert_job(self, job): raise RuntimeError("disk full")
        def delete_job(self, job_id): raise RuntimeError("disk full")
    reg2 = LibraryIngestJobRegistry()
    reg2.attach_store(_Boom())
    j = reg2.submit(source_path="/b.mp3")     # store raises, mutation still succeeds
    assert j.state.value == "queued"


def test_plan_restore_normalizes_interrupted_and_prunes():
    from tldw_chatbook.Library.library_ingest_jobs import plan_restore, IngestJobState
    rows = [
        {"seq": 1, "job_id": "ingest-job-1", "source_path": "/a", "title": "", "author": "",
         "keywords": "[]", "perform_analysis": 0, "chunk_enabled": 0, "chunk_size": 0,
         "state": "done", "retry_count": 0, "detected_type": "", "error": "",
         "finished_at_wall": "2026-07-12T00:00:00+00:00", "media_id": 7,
         "superseded": 0, "dismissed": 0, "permanent": 0},
        {"seq": 2, "job_id": "ingest-job-2", "source_path": "/b", "title": "", "author": "",
         "keywords": "[]", "perform_analysis": 0, "chunk_enabled": 0, "chunk_size": 0,
         "state": "parsing", "retry_count": 1, "detected_type": "audio", "error": "",
         "finished_at_wall": "", "media_id": None, "superseded": 0, "dismissed": 0, "permanent": 0},
    ]
    plan = plan_restore(rows, max_persisted=500, now_iso="2026-07-12T09:00:00+00:00")
    by_id = {j.job_id: j for j in plan.jobs}
    assert by_id["ingest-job-1"].state == IngestJobState.DONE and by_id["ingest-job-1"].media_id == 7
    assert by_id["ingest-job-2"].state == IngestJobState.FAILED
    assert by_id["ingest-job-2"].error == "Interrupted by app restart"
    assert by_id["ingest-job-2"].finished_at_wall == "2026-07-12T09:00:00+00:00"
    assert by_id["ingest-job-2"].retry_count == 1          # count preserved, not reset
    assert plan.next_id == 3
    assert "ingest-job-2" in [j.job_id for j in plan.upsert]   # normalized -> re-persist
    assert plan.delete_ids == []


def test_plan_restore_prune_cap_drops_oldest():
    rows = [
        {"seq": i, "job_id": f"ingest-job-{i}", "source_path": "/p", "title": "", "author": "",
         "keywords": "[]", "perform_analysis": 0, "chunk_enabled": 0, "chunk_size": 0,
         "state": "done", "retry_count": 0, "detected_type": "", "error": "",
         "finished_at_wall": "", "media_id": None, "superseded": 0, "dismissed": 0, "permanent": 0}
        for i in range(1, 6)
    ]
    plan = plan_restore(rows, max_persisted=3, now_iso="2026-07-12T09:00:00+00:00")
    assert [j.job_id for j in plan.jobs] == ["ingest-job-3", "ingest-job-4", "ingest-job-5"]
    assert plan.delete_ids == ["ingest-job-1", "ingest-job-2"]
    assert plan.next_id == 6
```

- [ ] **Step 2: Run to verify it fails**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Library/test_library_ingest_jobs.py -q -p no:cacheprovider -o addopts="" --timeout=120
```
Expected: FAIL — no `retry_count` / no `attach_store` / no `plan_restore`.

- [ ] **Step 3: Add `retry_count` + `IngestJobStore` + store plumbing**

In `library_ingest_jobs.py` (`LibraryIngestJob.retry_count` already exists from Task 1):
- Add imports: `from typing import Protocol` (extend the existing typing import); `replace`/`dataclass` are already imported.
- Define near the top:
```python
class IngestJobStore(Protocol):
    def upsert_job(self, job: "LibraryIngestJob") -> None: ...
    def delete_job(self, job_id: str) -> None: ...
```
- In `__init__`, add `self._store: IngestJobStore | None = None`.
- Add methods:
```python
    def attach_store(self, store: IngestJobStore) -> None:
        self._store = store

    def _persist(self, job: LibraryIngestJob) -> None:
        if self._store is None:
            return
        try:
            self._store.upsert_job(job)
        except Exception:
            logger.opt(exception=True).debug(f"ingest job persist failed: {job.job_id}")

    def _persist_delete(self, job_id: str) -> None:
        if self._store is None:
            return
        try:
            self._store.delete_job(job_id)
        except Exception:
            logger.opt(exception=True).debug(f"ingest job delete-persist failed: {job_id}")
```
- At the end of each mutation (`submit`, `mark_parsing`, `mark_writing`, `mark_done`, `mark_failed`, `dismiss`), after updating `self._jobs` and before/after `self._notify_listeners()`, call `self._persist(self._jobs[index])` (for `submit`, the appended job). For `requeue`, persist BOTH the superseded original and the new copy. For `clear_finished`, call `self._persist_delete(job_id)` for each removed job.
- `requeue`'s new-copy constructor already copies `detected_type` (task-160 fix) — ADD `retry_count=source.retry_count + 1,`.

- [ ] **Step 4: Add `restore` + `plan_restore`**

```python
@dataclass
class RestorePlan:
    jobs: list["LibraryIngestJob"]
    next_id: int
    upsert: list["LibraryIngestJob"]     # normalized jobs to re-persist
    delete_ids: list[str]                # pruned jobs to delete from the store


_INTERRUPTED_STATES = (IngestJobState.QUEUED, IngestJobState.PARSING, IngestJobState.WRITING)


def _job_from_row(row: dict) -> "LibraryIngestJob":
    return LibraryIngestJob(
        job_id=row["job_id"],
        source_path=row["source_path"],
        title=row["title"] or "",
        author=row["author"] or "",
        keywords=tuple(json.loads(row["keywords"] or "[]")),
        perform_analysis=bool(row["perform_analysis"]),
        chunk_enabled=bool(row["chunk_enabled"]),
        chunk_size=int(row["chunk_size"]),
        state=IngestJobState(row["state"]),
        detected_type=row["detected_type"] or "",
        media_id=row["media_id"],
        error=row["error"] or "",
        finished_at_wall=row["finished_at_wall"] or "",
        superseded=bool(row["superseded"]),
        dismissed=bool(row["dismissed"]),
        permanent=bool(row["permanent"]),
        retry_count=int(row["retry_count"]),
        # monotonic fields are not round-trippable -- leave defaults.
        submitted_at=0.0, started_at=None, finished_at=None,
    )


def plan_restore(rows: list[dict], *, max_persisted: int, now_iso: str) -> RestorePlan:
    jobs = [_job_from_row(r) for r in rows]            # rows are seq-ascending
    normalized_ids: set[str] = set()
    for i, job in enumerate(jobs):
        if job.state in _INTERRUPTED_STATES:
            jobs[i] = replace(
                job, state=IngestJobState.FAILED,
                error="Interrupted by app restart", permanent=False,
                finished_at_wall=now_iso,
            )
            normalized_ids.add(job.job_id)
    delete_ids: list[str] = []
    if len(jobs) > max_persisted:
        pruned = jobs[:-max_persisted]
        delete_ids = [j.job_id for j in pruned]
        jobs = jobs[-max_persisted:]
    upsert = [j for j in jobs if j.job_id in normalized_ids]   # kept + normalized
    next_id = max((int(j.job_id.rsplit("-", 1)[-1]) for j in jobs), default=0) + 1
    return RestorePlan(jobs=jobs, next_id=next_id, upsert=upsert, delete_ids=delete_ids)
```
And the registry seeder (does NOT fire per-job persist — bulk restore):
```python
    def restore(self, jobs: list[LibraryIngestJob], next_id: int) -> None:
        self._jobs = list(jobs)
        self._next_id = next_id
        self._notify_listeners()
```
(Add `import json` and confirm `replace` is imported — it already is.)

- [ ] **Step 5: Run to verify it passes + the full registry suite**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Library/test_library_ingest_jobs.py -q -p no:cacheprovider -o addopts="" --timeout=180
```
Expected: PASS (new tests + all pre-existing registry tests unchanged).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Library/library_ingest_jobs.py Tests/Library/test_library_ingest_jobs.py
git commit -m "feat(ingest): registry retry_count + store write-through hook + restore/plan_restore (161)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: App wiring — config path, on_mount restore, teardown close, retry indicator + backlog Done

**Files:**
- Modify: `tldw_chatbook/config.py` (path helper)
- Modify: `tldw_chatbook/app.py` (`_restore_ingest_jobs`, `on_mount` call, teardown close)
- Modify: `tldw_chatbook/Home/active_work_adapter.py` (`· retry {n}` indicator)
- Modify: `backlog/tasks/task-161 - Persistent-ingest-job-history-across-restarts.md`
- Test: `Tests/Library/test_library_ingest_jobs_restore.py` (create) + `Tests/Home/` retry-indicator unit

**Interfaces:**
- Consumes: `LibraryIngestJobsDB` (Task 1); `attach_store`/`restore`/`plan_restore`/`RestorePlan` + `retry_count` (Task 2); `get_user_data_dir` (existing).

- [ ] **Step 1: Write the failing tests**

Create `Tests/Library/test_library_ingest_jobs_restore.py` (end-to-end store↔registry round-trip via the real store):
```python
from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.Library.library_ingest_jobs import (
    LibraryIngestJobRegistry, IngestJobState, plan_restore,
)


def _restore(store, max_persisted=500, now_iso="2026-07-12T09:00:00+00:00"):
    reg = LibraryIngestJobRegistry()
    reg.attach_store(store)
    plan = plan_restore(store.all_jobs(), max_persisted=max_persisted, now_iso=now_iso)
    reg.restore(plan.jobs, plan.next_id)
    for j in plan.upsert:
        store.upsert_job(j)
    for jid in plan.delete_ids:
        store.delete_job(jid)
    return reg


def test_history_survives_restart_interrupted_normalized(tmp_path):
    store = LibraryIngestJobsDB(tmp_path / "jobs.db")
    a = LibraryIngestJobRegistry(); a.attach_store(store)
    done = a.submit(source_path="/done.pdf"); a.mark_parsing(done.job_id); a.mark_writing(done.job_id); a.mark_done(done.job_id, media_id=5)
    interrupted = a.submit(source_path="/x.mp4"); a.mark_parsing(interrupted.job_id)   # left PARSING (quit)
    failed = a.submit(source_path="/y.mp3"); a.mark_parsing(failed.job_id); a.mark_failed(failed.job_id, error="bad codec")
    store.close()

    store2 = LibraryIngestJobsDB(tmp_path / "jobs.db")            # reopen (restart)
    reg = _restore(store2)
    by_id = {j.job_id: j for j in reg.jobs()}
    assert by_id[done.job_id].state == IngestJobState.DONE and by_id[done.job_id].media_id == 5
    assert by_id[interrupted.job_id].state == IngestJobState.FAILED
    assert by_id[interrupted.job_id].error == "Interrupted by app restart"
    assert by_id[failed.job_id].state == IngestJobState.FAILED and by_id[failed.job_id].error == "bad codec"
    # _next_id advanced past the max so a new submit doesn't collide
    fresh = reg.submit(source_path="/z.txt")
    assert fresh.job_id == "ingest-job-4"
    store2.close()


def test_interrupted_retry_after_restart_requeues(tmp_path):
    store = LibraryIngestJobsDB(tmp_path / "jobs.db")
    a = LibraryIngestJobRegistry(); a.attach_store(store)
    j = a.submit(source_path="/x.mp4"); a.mark_parsing(j.job_id)     # interrupted
    store.close()
    reg = _restore(LibraryIngestJobsDB(tmp_path / "jobs.db"))
    restored = reg.jobs()[0]
    assert restored.state == IngestJobState.FAILED
    requeued = reg.requeue(restored.job_id)                          # AC2: retryable
    assert requeued.state == IngestJobState.QUEUED and requeued.retry_count == 1
```

Add a retry-indicator unit to `Tests/Home/` (find the existing active-work adapter test file via `ls Tests/Home/`; if none, create `Tests/Home/test_active_work_ingest_retry.py`) asserting a FAILED job with `retry_count=2` yields a `status_detail` containing `retry 2`. (Match the adapter's construction — see Step 4 for the exact field.)

- [ ] **Step 2: Run to verify it fails**

Run the two files; Expected: FAIL — `_restore` uses real store (exists) but the retry-indicator assertion fails (no indicator yet), and the round-trip may need the app path helper only in later steps. The restore tests should PASS already if Tasks 1–2 are done (they use only library + DB) — if so, they're a regression guard; the retry-indicator test is the RED for this task.

- [ ] **Step 3: Config path helper**

In `config.py`, next to `get_library_collections_db_path` (~:3597), add:
```python
def get_library_ingest_jobs_db_path() -> Path:
    custom_path = get_cli_setting("database", "library_ingest_jobs_db_path", None)
    if custom_path and custom_path != DEFAULT_CONFIG_FROM_TOML.get("database", {}).get("library_ingest_jobs_db_path"):
        db_path = validate_path_simple(Path(str(custom_path)).expanduser(), require_exists=False).resolve()
    else:
        db_path = get_user_data_dir() / "tldw_chatbook_library_ingest_jobs.db"
    return db_path
```

- [ ] **Step 4: `_restore_ingest_jobs` + on_mount + teardown + retry indicator**

In `app.py`, add the constant `_MAX_PERSISTED_INGEST_JOBS = 500` near the ingest constants, and a method on `LibraryIngestQueueMixin`:
```python
    def _restore_ingest_jobs(self) -> None:
        """One-time on_mount restore of persisted ingest job history."""
        from datetime import datetime, timezone
        from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
        from tldw_chatbook.Library.library_ingest_jobs import plan_restore
        try:
            store = LibraryIngestJobsDB(get_library_ingest_jobs_db_path())
            self._library_ingest_jobs_store = store
            self.library_ingest_jobs.attach_store(store)
            plan = plan_restore(
                store.all_jobs(),
                max_persisted=_MAX_PERSISTED_INGEST_JOBS,
                now_iso=datetime.now(timezone.utc).isoformat(),
            )
            self.library_ingest_jobs.restore(plan.jobs, plan.next_id)
            for job in plan.upsert:
                store.upsert_job(job)
            for job_id in plan.delete_ids:
                store.delete_job(job_id)
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to restore persisted ingest job history; starting empty."
            )
```
Call `self._restore_ingest_jobs()` once in `on_mount` (`app.py:5421`), after the registry exists (it's created in `__init__`), early in the Library-related setup. Import `get_library_ingest_jobs_db_path` from `.config` (extend the existing config import). In `on_unmount` (`app.py:6029`), after the pool shutdown, close the store:
```python
        store = getattr(self, "_library_ingest_jobs_store", None)
        if store is not None:
            store.close()
```
Retry indicator — in `Home/active_work_adapter.py._local_ingest_job_items` (~:621), change the `status_detail` to append the retry suffix. Add a tiny helper near the module's other job helpers (`~:787`):
```python
def _ingest_retry_suffix(job) -> str:
    return f" · retry {job.retry_count}" if job.retry_count else ""
```
and build the detail as:
```python
                    status_detail=(
                        (short_ingest_error(job.error)
                         if job.state == IngestJobState.FAILED and job.error else "")
                        + _ingest_retry_suffix(job)
                    ),
```
(The suffix is markup-safe plain text — consistent with the existing `markup=False` rendering.) If `library_screen.py`'s ingest queue row builds a parallel detail line from `short_ingest_error`, apply the same suffix there (grep `short_ingest_error` in `library_screen.py`).

- [ ] **Step 5: Run to verify it passes + import smoke**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Library/test_library_ingest_jobs_restore.py Tests/DB/test_library_ingest_jobs_db.py Tests/Library/test_library_ingest_jobs.py Tests/Home/ \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
HOME=/private/tmp/tldw-chatbook-test-home PYTHONPATH=$(pwd) \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('import ok')"
```
Expected: PASS + `import ok`.

- [ ] **Step 6: Mark backlog Done + commit**

```bash
perl -0pi -e 's/- \[ \] (#\d)/- [x] $1/g' "backlog/tasks/task-161 - Persistent-ingest-job-history-across-restarts.md"
perl -0pi -e 's/^status: .*/status: Done/m' "backlog/tasks/task-161 - Persistent-ingest-job-history-across-restarts.md"
```
Add a short `## Implementation Notes` (store + write-through hook + on_mount restore/normalize/prune + retry_count).
```bash
git add tldw_chatbook/config.py tldw_chatbook/app.py tldw_chatbook/Home/active_work_adapter.py \
  Tests/Library/test_library_ingest_jobs_restore.py Tests/Home/ \
  "backlog/tasks/task-161 - Persistent-ingest-job-history-across-restarts.md"
git commit -m "feat(ingest): persist + restore ingest job history on restart; retry indicator; task 161 done (161)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Final gate (after Task 3)

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/DB/test_library_ingest_jobs_db.py Tests/Library/ Tests/Home/ Tests/DB/ \
  -q -p no:cacheprovider -o addopts="" --timeout=600 --timeout-method=thread
```
Plus `python -c "import tldw_chatbook.app"`. Then the whole-branch review (opus) and finishing-a-development-branch. Served-TUI visual QA (submit a job, quit mid-parse, relaunch → job shows "interrupted by restart" + Retry works) is worthwhile but optional.
