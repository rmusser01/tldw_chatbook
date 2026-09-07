# Persistent ingest job history across restarts (task 161)

**Status:** Design approved (brainstorm), pending spec review.
**Backlog:** task-161 — "Persistent ingest job history across restarts".
**Builds on:** F3 parallel-parse ingest (PR #594) + the in-memory `LibraryIngestJobRegistry`.

## Problem

`LibraryIngestJobRegistry` (`tldw_chatbook/Library/library_ingest_jobs.py`) stores jobs in an in-memory `list`; queued/failed/done history is lost on quit, and a job interrupted mid-processing simply vanishes. We want job history to survive a restart and interrupted/failed jobs to be retryable afterward.

## Goal / Acceptance

- **AC1** — ingest job history survives an app restart.
- **AC2** — failed/queued jobs can be retried after restart.

## Chosen approach

A small SQLite store persists every visible ingest job; the registry **writes through** on each mutation and is **loaded once at startup**. Interrupted in-flight jobs (QUEUED/PARSING/WRITING at quit) normalize to `FAILED("Interrupted by app restart")` on load, so the app is idle on launch and the user retries manually (chosen over auto-resume in brainstorm — no surprise heavy work on launch). Done/failed jobs restore as-is. Mechanism follows the existing `LibraryCollectionsDB` template.

**Crash recovery (bonus):** because the store writes through on every transition — including the transient PARSING/WRITING — a job caught mid-parse by a *crash* (not just a clean quit) is already persisted as PARSING and comes back as retryable FAILED-interrupted on the next launch. This is why the transient states are persisted rather than only terminal ones.

### Why per-mutation write-through is safe (threading)

Every registry mutation runs on the **UI thread**: `submit`/`mark_parsing`/`mark_writing` in UI-thread coordinator methods, and the writer thread marshals `mark_done`/`mark_failed` back via `self.call_from_thread(...)` (`app.py:2113`/`2128`), as do the pool-completion (`_on_ingest_parse_complete`) and broken-pool paths. So the store is accessed from a single thread — per-mutation synchronous SQLite writes need no cross-thread connection handling.

## Components

### 1. `LibraryIngestJobsDB` (`tldw_chatbook/DB/Library_Ingest_Jobs_DB.py`, new)

A `BaseDB` subclass mirroring `LibraryCollectionsDB`: `_CURRENT_SCHEMA_VERSION = 1`, a `transaction()` context manager, `_initialize_schema()` via `executescript` with a `schema_version` table. **WAL mode** enabled. Because writes fire per mutation and store access is single-threaded (UI thread), the store keeps a **persistent connection** (opened once at `attach_store`, reused, closed on app shutdown) rather than opening/closing per write — a large drop (e.g. 100 files → ~100 `submit` upserts plus per-transition writes) then pays only the ~sub-ms write, not connection setup each time. (`close()` on app teardown.) One table `ingest_jobs`:

| column | type | notes |
|---|---|---|
| `seq` | INTEGER PRIMARY KEY | the numeric `n` parsed from `job_id` ("ingest-job-{n}") — the submission-order key; `ORDER BY seq ASC` restores insertion order and `max(seq)+1` restores `_next_id` |
| `job_id` | TEXT UNIQUE NOT NULL | "ingest-job-{n}" |
| `source_path` | TEXT NOT NULL | |
| `title`,`author` | TEXT | |
| `keywords` | TEXT | JSON array of strings |
| `perform_analysis`,`chunk_enabled`,`superseded`,`dismissed`,`permanent` | INTEGER | 0/1 |
| `chunk_size` | INTEGER | |
| `state` | TEXT NOT NULL | IngestJobState value; `CHECK (state IN ('queued','parsing','writing','done','failed'))` — documents valid states + catches corruption (borrowed from the server jobs schema's status CHECK) |
| `retry_count` | INTEGER NOT NULL DEFAULT 0 | how many times this job's lineage has been retried (see §2a) |
| `detected_type`,`error`,`finished_at_wall` | TEXT | |
| `media_id` | INTEGER | nullable |

**Not stored (can't round-trip):** the `time.monotonic()` fields `submitted_at`/`started_at`/`finished_at` — meaningless across restart. `finished_at_wall` (real ISO-8601 UTC) is the one preserved timestamp.

Methods: `upsert_job(job: LibraryIngestJob) -> None`, `delete_job(job_id: str) -> None`, `all_jobs() -> list[dict]` (ordered by `seq`), `transaction()`. `upsert_job` derives `seq` from `job_id`.

### 2. Registry ↔ store hook (`library_ingest_jobs.py`)

`LibraryIngestJobRegistry(store: IngestJobStore | None = None)` where `IngestJobStore` is a small `Protocol` (`upsert_job(job)`, `delete_job(job_id)`). **Default `None` preserves today's pure, DB-free behavior — every existing registry test is unchanged.**

- Each mutation point (`submit`, `mark_parsing`, `mark_writing`, `mark_done`, `mark_failed`, `requeue`, `dismiss`) upserts the *stored* job object (`self._jobs[index]`, not the returned copy) after mutating.
- `requeue` upserts BOTH the superseded original and the new queued copy.
- `clear_finished` calls `store.delete_job(job_id)` for each hard-removed job.
- `attach_store(store)` sets `self._store` after construction (so the app can defer DB I/O to `on_mount`); the constructor param stays for tests.
- A `restore(jobs: list[LibraryIngestJob], next_id: int)` method seeds `self._jobs` + `self._next_id` without firing per-job upserts (bulk restore is not a mutation to re-persist, except the interrupted-normalization — see §3).

Persistence is **best-effort**: `_persist`/`_delete` wrap the store call in `try/except` and log at debug on failure — the in-memory registry stays the live source of truth; a store error never breaks a mutation or blocks the UI.

### 2a. Retry-count tracking (borrowed from the server jobs schema)

`LibraryIngestJob` gains `retry_count: int = 0`. A fresh `submit` starts at 0; `requeue` (the retry path) sets the new queued copy's `retry_count = source.retry_count + 1`, so the count carries forward through the supersede chain and survives restart (persisted). No `max_attempts` cap — retries are manual (the user clicks), so there is no auto-retry to gate; the count is informational. It is surfaced minimally in the existing ingest job row (the secondary line appends `· retry {n}` when `retry_count > 0`) — no new widget/screen. This is the desktop-simplified take on the server's `retry_count`/`retry_now_jobs` (`manager.py:6386`): a plain counter, no exponential backoff.

### 3. Startup restore (`app.py`, `_restore_ingest_jobs()` called from `on_mount`)

1. `rows = store.all_jobs()` (ordered by `seq`).
2. Rebuild `LibraryIngestJob`s (keywords JSON→tuple, ints→bools, monotonic fields defaulted: `submitted_at` re-seeded to a fresh `time.monotonic()`, `started_at`/`finished_at` = None).
3. **Normalize interrupted states:** any job whose stored `state` ∈ {QUEUED, PARSING, WRITING} → `FAILED`, `error="Interrupted by app restart"`, `permanent=False` (retryable), and stamp `finished_at_wall = datetime.now(timezone.utc).isoformat()`.
4. **Prune cap:** keep at most the most recent `_MAX_PERSISTED_JOBS` (constant, 500) rows by `seq`; older terminal rows are dropped from the load AND deleted from the store.
5. Seed the registry via `restore(jobs, next_id=max(seq)+1)`, then **incrementally** reconcile the store to the post-restart truth: `upsert_job` ONLY the jobs whose state was normalized (interrupted→FAILED) and `delete_job` ONLY the pruned rows. Everything else already loaded is current — do NOT re-write all rows (no startup write storm).
6. No auto-dispatch: `_top_up_ingest_parse_pool()` is NOT called at startup — the app is idle until the user retries.

The registry stays store-less in `TldwCli.__init__` (`app.py:3190`, unchanged, no DB I/O at construction); `_restore_ingest_jobs()` runs once in `on_mount`, where it creates the `LibraryIngestJobsDB` (independent of `media_db`), `attach_store`s it to the registry, then loads/normalizes/restores. No submits happen between `__init__` and `on_mount`, so nothing is lost by attaching late.

### 4. Path / config (`config.py`)

`get_library_ingest_jobs_db_path()` mirroring `get_library_collections_db_path()` (`config.py:~3597`): file `tldw_chatbook_library_ingest_jobs.db` under `get_user_data_dir()`, with an optional `[database] library_ingest_jobs_db_path` custom-path key.

## Data flow

```
mutation (UI thread) → registry updates self._jobs → _persist(job) → store.upsert_job  (best-effort)
clear_finished       → registry removes rows        → store.delete_job each
app start (on_mount) → store.all_jobs() → rebuild + normalize-interrupted + prune
                     → registry.restore(...) → re-persist normalized/pruned
user Retry (existing) → requeue → upsert both → _top_up dispatches (already wired)
```

## Error handling

- Store writes/deletes are best-effort (try/except + debug log); the registry never fails a mutation on a store error.
- A corrupt/unreadable store at load → log a warning and start with an empty registry (never crash `on_mount`).
- WAL mode + short per-transaction connections (the `LibraryCollectionsDB` pattern) keep writes fast and non-blocking of readers.

## Testing

- **Store unit (`Tests/DB/test_library_ingest_jobs_db.py`):** `upsert_job`/`all_jobs`/`delete_job` round-trip over a tmp sqlite file; `all_jobs()` returns `seq`-ascending; schema init + WAL; keywords JSON round-trip; `upsert` is idempotent (update-in-place on the same `job_id`).
- **Registry-with-store (`Tests/Library/test_library_ingest_jobs.py`):** with a fake in-memory store, each mutation calls `upsert_job` with the right job; `clear_finished` calls `delete_job`; `store=None` fires nothing (existing tests unchanged); a store that raises does not break the mutation.
- **Round-trip + restore (`Tests/Library/`):** build a mix (queued, parsing, writing, done, failed, superseded, dismissed) in registry A + a real tmp store; run the restore path into registry B; assert QUEUED/PARSING/WRITING → FAILED("Interrupted by app restart"), DONE/FAILED/superseded/dismissed preserved, `jobs()`/`counts()` match the expected post-normalization set, `_next_id` = max(seq)+1, and the prune cap drops the oldest beyond `_MAX_PERSISTED_JOBS`.
- **Retry-after-restart (AC2, coordinator harness `Tests/Library/test_library_ingest_runner.py`):** a restored interrupted→FAILED job, when retried (`retry_library_ingest_job` → `requeue`), dispatches to the pool (mirrors the existing retry tests).
- **Retry-count (`Tests/Library/test_library_ingest_jobs.py`):** `submit` → `retry_count == 0`; `requeue` of a job with `retry_count == k` → new copy has `retry_count == k+1`; the count round-trips through the store; the CHECK constraint rejects an out-of-enum `state` on direct insert.

## Scope / non-goals

- No auto-resume / startup auto-dispatch; interrupted jobs come back as retryable FAILED.
- No new UI screens/widgets — the existing Home/Library ingest list + Retry control render whatever's in the registry; the only copy change is the minimal `· retry {n}` indicator on a job's existing secondary line (§2a).
- Retry-count is a plain counter only: no `max_attempts` cap / no auto-retry / no backoff (all deferred with the server's richer retry machinery — out of scope).
- `progress_percent`/`progress_message` (live parse progress) and `source_path` idempotency/dedup — both good server ideas, logged as separate backlog follow-ups, not built here.
- `time.monotonic()` timestamps deliberately not preserved; only `finished_at_wall` round-trips. Known cosmetic limit: restored rows lose their original *submit* wall-time (`submitted_at` is re-seeded to a fresh monotonic), so a relative "submitted N ago" reads ~now for restored jobs; terminal jobs still show a real finished time. Not adding a `submitted_at_wall` field (scope creep).
- Prune cap is a fixed constant (500), not user-configurable (YAGNI); `clear_finished()` remains the manual purge.
