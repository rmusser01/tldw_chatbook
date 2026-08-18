# Durable, resumable research jobs — design

Date: 2026-08-18 · Task: TASK-18060 · Closes: TASK-17386 AC #3

## Problem

A research run whose evidence pool is large enough loses its report to a
provider timeout during synthesis. TASK-17386 made that failure legible and
counted; it did not make the run finish.

The first design for this attacked the clock: derive a size-aware timeout so a
big synthesis fits. Reviewing it produced a better framing. A research run is
not something a user watches — its result is an artifact and a message in the
conversation that asked for it. Once the run is a durable job, taking twenty
minutes is not a failure, and almost all of the timeout machinery becomes
unnecessary. What matters instead is that the job SURVIVES, RESUMES without
redoing paid work, and SURFACES when it finishes.

Measured on the repositories lane (local Qwen3.8-27B), which justifies the
generous bounds below:

| admitted pool | synthesis wall clock |
|---|---|
| 14 / 26 / 32 | 241s / 328s / 287s |
| 46 | FAILED — two 600s attempts, `MaxRetryError` |
| 66 | 970s (a timed-out attempt plus a landing retry) |

## What this promises, and what it does not

The scheduler loop runs INSIDE the app process (`app.py` starts it with
`run_worker`). So this design makes a run survive an app exit and continue on
the next launch. It does NOT run research while the app is closed, and the spec
says so plainly because "background job" invites that expectation.

Resume granularity is the PHASE. A synthesis interrupted at 900 of 970 seconds
restarts that synthesis; the collection and judgement before it are not redone.
Synthesis is the most expensive single phase, so this is the design's main
residual cost, stated rather than implied.

## Design

### 1. Exactly one executor per run

Today `execute_run` refuses only terminal runs, and the window's guard
(`run_worker(exclusive=True, group=...)`) is per-session. Adding a scheduler
that can invoke the same run makes concurrent execution possible: duplicate
searches, duplicate spend, racing writes on one row.

A run therefore carries a lease. The shape is taken from the server's job
manager (`tldw_Server_API/app/core/Jobs/`, dev), which has solved this already;
four of its decisions are adopted deliberately rather than re-derived:

- **A lease id as well as a worker id.** The server's `jobs` table carries
  `worker_id`, `lease_id`, `leased_until`, `acquired_at`, and
  `renew_job_lease` can enforce that BOTH the worker and the lease id match
  before a renewal or completion is accepted. A worker id alone is not enough:
  a process that stalled past its lease, had its run taken over, and then woke
  up would still match on worker id and could complete a run it no longer owns.
  The lease id is the fencing token that makes takeover safe.
- **Stale leases are reclaimed at acquisition, not by a reaper.** The server
  requeues or terminally fails expired `processing` jobs as the first step of
  `acquire_next_job`, "according to their retry budget". Folding recovery into
  the acquire path means there is no separate sweeper to schedule, and no
  window where a dead run is invisible.
- **A retry budget decides requeue versus terminal failure.** `max_retries` and
  `retry_count` on the row, with `available_at` carrying the backoff, so a run
  that keeps dying is eventually failed rather than retried forever. A research
  run that crashes its executor three times is a broken run, not a slow one.
- **The heartbeat carries progress.** `renew_job_lease` takes
  `progress_percent` and `progress_message`, so one call both proves liveness
  and updates what the user sees. The engine already emits phase progress; that
  emission becomes the heartbeat rather than a second mechanism beside it.

What is NOT adopted: the server's multi-tenant machinery — domains, queues,
fair-share scheduling, priority bands, quarantine for poison messages, the
archive table. A single-user SQLite app with one research queue needs a lease
and a retry budget, not a scheduler-of-schedulers. Its lease cap
(`JOBS_LEASE_MAX_SECONDS`, 3600s) is worth noting though, because a synthesis
measured at 970s sits inside it but a pathological one would not: the lease
must be renewable mid-phase, not sized to cover a whole phase in one grant.

### 2. Resume restores the budget, not just the position

`execute_run` rebuilds the ledger with `BudgetLedger.from_limits(limits)` on
every entry, and `budget_ledger.json` is written but never read back. With
resume becoming routine rather than exceptional, a run resumed three times
would be granted its full search and token budget three times. Resume restores
the ledger from its artifact, and only falls back to `from_limits` for a run
that has never executed.

### 3. Phase state durable enough to resume

`collection_summary.json` persists counts, sub-questions and warnings — not the
collected evidence — so a resumed run re-searches everything. Each round's
evidence pool becomes an artifact.

Bounded explicitly, because "bounded" without a number is how the previous
draft of this spec hid its worst case: 66 sources at roughly 10-50 KB of
scraped text each is 0.7-3 MB per round, up to about 6 MB per run. The design
persists evidence WITH content up to a per-run cap (default 8 MB), and beyond
that cap persists references only, recording in the artifact that it did so. A
resumed run rehydrates what it can and re-fetches the remainder, so the cap
trades a bounded amount of network for a bounded amount of storage instead of
letting either grow with the pool.

### 4. Scheduler integration

Execution is driven by a scheduler task (`task_type="research_run"`) whose
handler claims the run and invokes the engine. The handler skips runs in an
`awaiting_*` control state: a checkpointed run parked for review would
otherwise be re-entered on every tick, re-park immediately, and emit an event
each time.

### 5. Surfacing

- Window-launched runs gain a handoff target, so they alert the originating
  conversation the way Console `/research` already does; `chat_handoff` appears
  zero times in `Research_Window.py` today.
- Research reports gain the `artifact_source` / `artifact_kind` metadata the
  artifacts screen filters on. That screen currently requires
  `artifact_source == "console"`, so widening it is a screen change and carries
  the matching `Docs/User_Guide/` update.

### 6. Bounds

What remains of the timeout work: a generous per-attempt synthesis ceiling, a
phase-2 deadline that cannot pre-empt it, and no default runtime cap. No
calibrated curve, no multiplier, no instrumentation step — being slow stops
being a failure once the result is a job. The measurements above set the
ceiling; a user-set `max_runtime_seconds` still refuses before spending when it
cannot cover the work, recording that through the `synthesis_failed` channel
shipped in TASK-17386.

## Out of scope

`_default_gap_fn` and `answer_follow_up` share the timeout exposure and are
recorded as a follow-up. Running research while the application is closed would
require an out-of-process scheduler and is a different piece of work.

## Testing

- Two executors race one run: exactly one claims it, the other declines, and no
  phase runs twice.
- A stale lease is taken over; a live one is not.
- A resumed run restores its spent budget rather than being re-granted it.
- A resumed run does not re-search a completed round, and rehydrates evidence
  from the artifact.
- Evidence beyond the cap persists as references, and the artifact records the
  truncation.
- The handler skips `awaiting_*` runs without emitting events.
- A window-launched run alerts its originating conversation.
- A research report appears in the artifacts screen's listing.
- No test contacts a network.

## Success criteria

A research run interrupted by an app exit resumes on the next launch without
redoing its searches or re-granting its budget, finishes a synthesis that takes
longer than any single provider timeout, and announces itself in the
conversation that asked for it and in the artifacts screen.
