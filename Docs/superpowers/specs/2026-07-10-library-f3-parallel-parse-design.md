# F3 — Library ingest: parallel parsing, single-writer persistence — Design

**Status:** approved by user 2026-07-10 (architecture dictated: "multi-processing for parsing of files, and fan that out, but single-thread ingest of parsed items"; scope decision: all media types through a small pool).

## Problem

The Library ingest queue (`LibraryIngestJobRegistry` + `LibraryIngestQueueMixin`) is strictly serial: one `@work(exclusive=True, thread=True, group="library_ingest_queue")` loop parses AND persists each job in turn. One large file (a long transcription, a heavy PDF) blocks every job behind it, and bulk imports use a single core. SQLite cannot handle concurrent writers, so naive N-worker parallelism is off the table.

## Architecture

A two-stage pipeline replacing the parse half of the serial loop:

1. **Parse stage — process pool.** A lazily-created spawn-context `multiprocessing.Pool` fans file parsing out to worker processes. All media types go through the pool (documents, ebooks, audio/video transcription alike). Workers never touch the media DB.
2. **Write stage — the existing single writer.** Today's exclusive thread worker keeps its atomic claim-or-release loop but now only persists parsed payloads: one `add_media_with_keywords` call per job, always on the one writer thread. SQLite never sees a concurrent ingest writer.

A UI-thread **coordinator** (in `LibraryIngestQueueMixin`) owns the pool, submits up to N queued jobs, receives completions (marshaled to the UI thread), stores payloads, and wakes the writer. The registry stays UI-thread-only.

The pool is a `multiprocessing.get_context("spawn").Pool`, NOT a `concurrent.futures.ProcessPoolExecutor`: the executor's atexit hook joins running tasks, so an in-flight long transcription would block app exit for its full duration; `Pool` has a public `terminate()` for the quit path. Jobs are submitted with `apply_async(callback=..., error_callback=...)`; callbacks run on the pool's parent-side result-handler thread and marshal into the UI thread via `call_from_thread`.

## Components

### Parse function (spawn-safe, top-level)
`parse_local_file_for_ingest(path, options) -> dict`, extracted from the pre-DB half of `ingest_local_file` (everything before the `add_media_with_keywords` call). The pool's entry point lives in a NEW dedicated light module `Local_Ingestion/ingest_parse_worker.py` whose module scope imports nothing heavy — every parsing import (including `local_file_ingestion` itself) is deferred into the function body. Measured cost of importing `local_file_ingestion` is ~6.8s / ~6,000 modules: each worker process pays that once, on its FIRST parse (documented, amortized across the worker's lifetime), while pool spawn itself stays fast. When `perform_analysis` is enabled, analysis (LLM summarization) runs inside the per-type processors — i.e., IN the worker process, which therefore makes network calls using API keys loaded from the same config file; this is existing behavior relocated, not new capability. Returns a picklable payload (content, title, media type, metadata, keywords, analysis inputs). The worker wrapper catches ALL exceptions and returns a structured result `{ok: bool, payload | error: str, permanent: bool}` — never raises across the process boundary (no exception-pickling surprises), and permanent-vs-retryable classification (F1b M4: unsupported type, missing file) happens inside the worker where the real exception type is available. `ingest_local_file` becomes parse + persist composed, so `batch_ingest_files`, `quick_ingest`, and the server path are unchanged.

Heavy optional dependencies (PDF, transcription models) import lazily inside worker processes and are reused for that worker's lifetime. Spawn-start-method safe: plain module-level function, plain-data args.

### Registry states
`QUEUED → PARSING → WRITING → DONE / FAILED`. `RUNNING` is renamed `WRITING`; `PARSING` is new. The `"running"` status literal is consumed by the Home adapter (`_HOME_INGEST_JOB_ACTIVE_STATES`), dashboard status categories, queue-row copy, and many test asserts — the implementation plan must sweep every consumer of the literal, not just the enum. New transitions `mark_parsing(job_id)` and `mark_writing(job_id)` follow the existing frozen-dataclass replace pattern; `mark_failed(error, permanent)`, retry/supersede/dismiss, `finished_at_wall`, and the listener contract are unchanged. Queue rows render "parsing" / "writing"; the Home adapter maps both into the Running feed.

### Coordinator (UI thread, in `LibraryIngestQueueMixin`)
- Owns the pool: created lazily on first submission; `max_workers` from config `library.ingest_parse_workers` (1-arg `get_cli_setting` fallback, same bug-class guard as rail state), default `min(3, max(1, os.cpu_count() - 1))`.
- Submits queued jobs while fewer than N are `PARSING` (this cap IS the backpressure: at most N parsed payloads + 1 write in flight are ever held in memory).
- Completions land via the `apply_async` callback/error_callback (pool result-handler thread) → `call_from_thread` → coordinator: store payload in a coordinator-side dict `job_id → payload` (payloads can be MBs; they never enter the frozen registry snapshots the UI renders), transition the job, wake the writer, submit the next queued job.
- Parse failure → `mark_failed(error, permanent)` straight from the structured result.

### Writer (existing worker, narrowed)
The exclusive thread worker's loop claims the oldest payload-ready job in submission order (atomic claim-or-release via `call_from_thread`, exactly today's discipline — a fast small file may finish parsing before an older large one, but writes still happen in submission order among ready payloads), performs the DB write, resolves media_id (including the existing `get_media_by_url` re-ingest fallback), and marks DONE/FAILED. Write failures are retryable by default.

### Shutdown / failure containment
- App exit: the writer thread joins as today (the in-flight DB write completes — quit-contract parity), then `pool.terminate()` kills parse workers immediately. Abandoned in-flight parses are the same loss class as QUEUED jobs that are never claimed, which the existing contract already accepts (the registry is in-memory; persistence is a logged, separate follow-up). The coordinator sets a shutdown flag first and ignores any late pool callbacks after it (no `call_from_thread` into a closing app).
- Worker-process death (e.g. OOM during transcription): ALL in-flight parse jobs fail with a broken-pool error message (retryable — the pool cannot attribute the death to one job); the coordinator drops the broken pool and lazily rebuilds on the next submission. Retry recovers the innocent casualties.

## Config

`[library] ingest_parse_workers = <int>` (config.toml), read through `get_cli_setting("library.ingest_parse_workers")` fallback; invalid/missing → default formula. No UI settings surface in this phase.

## UI impact

Queue rows gain the "parsing" state label (counts line e.g. "2 parsing · 1 writing · 3 queued · 1 done"); Home Running feed shows parsing and writing jobs. No new controls. Visual QA required for the queue canvas and Home Running states before merge.

## Testing

- Registry: state-machine units for `PARSING`/`WRITING` transitions, retry/supersede interplay, invalid-transition guards.
- Coordinator: fake-pool seam — an inline synchronous executor injected in tests so pilots stay deterministic (gated-fake pattern, bounded waits); tests for submit-cap/backpressure, completion marshaling, failure classification passthrough, broken-pool rebuild.
- Writer: unchanged claim-or-release tests re-anchored to `WRITING`.
- One real spawn-`Pool` integration test: parse a small text file end-to-end through a real subprocess (marked, kept fast).
- Shutdown: pool terminated on exit, late callbacks ignored; no writer strand.

## Out of scope (logged follow-ups)

Persistent job history across restarts; URL/web ingest; a heavy-lane cap for concurrent transcriptions (revisit if RAM pressure shows up in practice); exposing worker count in the Settings UI; pre-chunking in parse workers (today `add_media_with_keywords` is called with `chunks=None`, leaving chunk status "pending" for downstream processing — the writer stage is already light, and pre-chunking would change behavior).
