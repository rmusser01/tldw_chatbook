# Ingest parallelism — heavy-lane cap (task 160)

**Status:** Design approved (brainstorm), pending spec review.
**Backlog:** task-160 — "Ingest parallelism: heavy-lane cap for concurrent transcriptions".
**Builds on:** F3 parallel-parse pool (PR #594).

## Problem

F3 fans all ingest media types through a single multiprocessing parse pool
sized `min(3, max(1, cpu-1))`. Audio/video parses run Whisper transcription and
are RAM/CPU heavy; two of them at once can thrash a machine while document
parses are comparatively light. We want to cap concurrent **transcription**
parses (default 1) independently of document parses, which keep fanning out to
fill the pool.

## Goal / Acceptance

- **AC1** — concurrent transcription (audio/video) parses are capped
  independently of document/pdf/ebook/text parses.
- **AC2** — a config key controls the heavy-lane cap.

## Chosen approach

Keep the single parse pool; make the **dispatcher** heavy-lane-aware. "Heavy" =
`{"audio", "video"}` (the only `detect_file_type` values that transcribe). The
dispatcher fills the pool up to `worker_count` total as today, but caps
concurrent heavy parses at `heavy_cap`, and when the heavy lane is full it
**skips heavy jobs and dispatches lighter jobs past them** (skip-ahead, approved
in brainstorm) so pool slots never idle. Jobs may therefore complete out of
enqueue order — acceptable, each ingest job is independent.

## Components

### 1. Config (`tldw_chatbook/app.py`)

Add `_ingest_heavy_lane_max_workers()` next to `_ingest_parse_worker_count`
(`app.py:1371-1393`), mirroring its parse/clamp:

```python
def _ingest_heavy_lane_max_workers(self) -> int:
    try:
        configured = int(get_cli_setting("library.ingest_heavy_lane_max_workers"))
    except (TypeError, ValueError):
        configured = 0
    return configured if configured > 0 else 1   # default 1; <=0 clamps to 1
```
`<=0 → 1` so a mis-set value can never permanently starve heavy work. A cap
larger than `worker_count` is harmless (the pool total still bounds everything).

### 2. Classify once at enqueue (`app.py` + registry)

`detect_file_type` (a pure extension map, `local_file_ingestion.py:35-84`) is
computed **at enqueue** and stored on the QUEUED job row, so the dispatcher can
select by type without re-deriving:

- `LibraryIngestJobs.submit(...)` (`library_ingest_jobs.py:266`) gains a
  `detected_type: str = ""` keyword param, stored on the created
  `LibraryIngestJob` (the dataclass already has `detected_type: str = ""` at
  `library_ingest_jobs.py:173`). The registry does NOT import `detect_file_type`
  — the caller passes the string, keeping the registry decoupled.
- The single `submit(...)` call site (`app.py:1326`) computes
  `detect_file_type(source_path)` (wrapped `try/except → ""`, same guard the
  dispatch loop uses today) and passes `detected_type=...`.
- **Remove the now-redundant dispatch-time recompute** (`app.py:1574`): the
  claim passes the job's stored type instead —
  `mark_parsing(job.job_id, detected_type=job.detected_type)`. Classifying twice
  (and risking drift) is eliminated. `mark_parsing`'s signature is unchanged.

### 3. Registry selection (`library_ingest_jobs.py`)

- **Extend `next_queued`** (not a new method) with an optional filter:
  ```python
  def next_queued(self, *, skip_types: frozenset[str] = frozenset()) -> LibraryIngestJob | None:
      for job in self._jobs:
          if job.state == IngestJobState.QUEUED and job.detected_type not in skip_types:
              return replace(job)
      return None
  ```
  `skip_types=frozenset()` (default) → today's "oldest QUEUED" behavior, so the
  four existing `next_queued()` unit tests and callers are unaffected. A
  non-empty `skip_types` returns the oldest QUEUED job whose `detected_type` is
  not skipped (skip-ahead).
- **Add** `parsing_count_for_types(types)`:
  ```python
  def parsing_count_for_types(self, types: frozenset[str]) -> int:
      return sum(1 for j in self._jobs
                 if j.state == IngestJobState.PARSING and j.detected_type in types)
  ```

### 4. Dispatcher (`_top_up_ingest_parse_pool`, `app.py:1542+`)

```python
HEAVY_TYPES = frozenset({"audio", "video"})   # module/class constant
...
worker_count = self._ingest_parse_worker_count()
heavy_cap = self._ingest_heavy_lane_max_workers()
while self.library_ingest_jobs.counts().get("parsing", 0) < worker_count:
    heavy_full = self.library_ingest_jobs.parsing_count_for_types(HEAVY_TYPES) >= heavy_cap
    job = self.library_ingest_jobs.next_queued(
        skip_types=HEAVY_TYPES if heavy_full else frozenset()
    )
    if job is None:
        return   # queue empty, or only heavy jobs remain and the lane is full
    claimed = self.library_ingest_jobs.mark_parsing(job.job_id, detected_type=job.detected_type)
    if claimed is None:
        ... existing break-with-log guard (unchanged) ...
    ... existing apply_async submit (unchanged) ...
```
`heavy_full` is re-evaluated each iteration — a heavy job just claimed via
`mark_parsing` (synchronous, UI thread) is reflected in
`parsing_count_for_types` on the next pass, so within one top-up a single heavy
+ N light fill correctly. When the loop finds nothing dispatchable (only
heavy-and-full remain) it returns with pool slots deliberately idle; the next
completion re-runs top-up, and a freed heavy slot admits the next transcription.

## Data flow

```
submit(source_path) → detect_file_type → detected_type on QUEUED row
top-up loop:
  heavy_full = parsing_count_for_types({audio,video}) >= heavy_cap
  job = next_queued(skip_types = {audio,video} if heavy_full else ∅)   # skip-ahead
  mark_parsing(job, job.detected_type) → apply_async(run_parse_job)
completion → mark done/failed → top-up re-evaluates (freed heavy slot admits next)
```

## Error handling

Unchanged. The heavy lane is pure dispatch **selection** — no new failure modes.
Per-job parse failures stay isolated (`run_parse_job` returns a structured
error, never raises across the process boundary); a broken pool still fails only
the in-flight jobs of that generation. The `mark_parsing`-rejected `break` guard
and the "one dispatch per iteration, re-evaluate at top" structure are
preserved, so no `continue`/infinite-loop risk is introduced. Unknown extensions
classify to `""` (not heavy) and fan out as light — correct, since only
audio/video transcribe.

## Testing

- **Registry units (`Tests/Library/test_library_ingest_jobs.py`):**
  - `next_queued(skip_types={"audio","video"})` returns the oldest QUEUED
    non-heavy job, skipping heavy ones; returns `None` when only heavy queued
    jobs remain; `next_queued()` (no args) is unchanged.
  - `parsing_count_for_types({"audio","video"})` counts only PARSING heavy jobs;
    ignores QUEUED/DONE/FAILED and light PARSING jobs.
  - `submit(..., detected_type="audio")` stores the type on the QUEUED row.
- **Config unit:** `_ingest_heavy_lane_max_workers` returns the configured value,
  defaults to 1 when unset, clamps `0`/negative to 1.
- **Coordinator (fake-pool harness, `Tests/Library/test_library_ingest_runner.py`):**
  the headline scenario — `worker_count=3`, `heavy_cap=1`, enqueue five jobs
  whose `source_path` extensions classify as `[audio1, audio2, doc1, doc2, doc3]`
  → in-flight (`len(pool.calls)`) = 3 = `{audio1, doc1, doc2}`; `audio2` stays
  QUEUED (heavy lane full); `doc3` stays QUEUED (pool full). Then
  `trigger_success(audio1)` → top-up promotes `audio2` to PARSING. Mirrors the
  existing `test_submit_cap_backpressure_*` test; the harness already supports a
  `worker_count` override — add a `heavy_lane` override the same way.

## Scope / non-goals

- Heavy set is hardcoded `{"audio", "video"}` (the two transcription types) — not
  configurable (YAGNI); only the cap count is config-controlled.
- One pool (the cap is a dispatch-selection policy, not a second pool).
- Skip-ahead ⇒ out-of-enqueue-order completion is accepted.
- No change to the parse worker (`run_parse_job`) or the real-spawn integration
  test — the cap is entirely coordinator/registry-side.
