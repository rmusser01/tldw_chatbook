# ADR-070: Research run lease, fencing, and durable budget/evidence

- **Status:** Accepted
- **Date:** 2026-08-18
- **Task:** task-18060 (Make research runs durable, resumable jobs)
- **Related:** ADR-068 (local research execution engine — this layers
  durability onto its engine); tldw_server `app/core/Jobs/` (the lease model
  adopted); `Docs/superpowers/specs/2026-08-18-durable-research-jobs-design.md`
  (design); PR #1822 review (the live/expired release semantics and
  keep-alive containment decisions below)

## Context

ADR-068's engine executes a run inside one app session with no execution
ownership: `execute_run` refuses only terminal runs, and the window's
exclusive-worker guard is per-session. Two processes (or a future scheduler)
could execute the same run concurrently — duplicate searches, duplicate
spend, and racing terminal writes. Resume also re-granted the whole budget
(the ledger was rebuilt from limits while `budget_ledger.json` was written
and never read), and an interrupted run left no evidence behind, so a resume
re-searched everything it had already paid for.

## Decision

1. **A lease with a fencing token, owned by the service layer.**
   `LocalResearchService.claim_run` takes the execution lease in one atomic
   `UPDATE` (`leased_until IS NULL OR expired`, plus `status NOT IN
   terminal`); `renew_lease` / `release_lease` / terminal writes authorise
   on `lease_id` — not `worker_id` — so a stalled worker that was taken over
   cannot act on a run it no longer owns. Lease timestamps all flow through
   one formatter (`_format_timestamp`, microsecond precision, `Z` suffix)
   because the expiry comparison is a string compare and divergent formats
   once let a live lease compare as expired.

2. **A crash-retry budget of *consecutive abandonments*.** Each successful
   claim increments `lease_attempts`; reclaiming an EXPIRED lease enforces
   `max_attempts` and signals exhaustion by raising `LeaseBudgetExhausted`
   (the caller fails the run) — distinct from `None` (another executor
   holds a live lease; leave the run alone). A **live** release resets the
   counter; a release of an **already-expired** lease is an abandonment
   being acknowledged, not a clean hand-off — it leaves the lease record in
   place so the next claim counts it (PR #1822 review: otherwise a
   stalling-but-alive executor could loop claim → expire → release forever
   without ever spending the budget).

3. **A keep-alive on a timer, not a progress hook.** The synthesis phase
   emits no progress for its whole duration (~970s measured), so renewal is
   an `asyncio` timer task for as long as a phase is in flight. Blocking
   pipeline seams (`search_fn`, `paper_search_fn`, the gap-analysis LLM
   call) are offloaded via `asyncio.to_thread` so the timer actually runs.
   A renewal that *raises* (e.g. transient SQLITE_BUSY) is contained inside
   the keep-alive — treated as lost, never allowed to surface in
   `execute_run`'s `finally` block where it would skip `release_lease`.

4. **A fence before every persisting write.** `_require_lease()` guards
   each artifact write, run-state write, and checkpoint creation; terminal
   writes (`complete_run`/`fail_run`) additionally carry `lease_id` so the
   UPDATE itself is lease-conditional (no check-then-act gap). A displaced
   executor's terminal write is a no-op that returns the current truth.
   Engine lease state is per-`execute_run`; every current caller builds a
   fresh engine per run, and a shared-engine caller would silently disable
   the fence — called out as a constraint on future scheduler design.

5. **Resume restores spent budget across all four axes** (searches, docs,
   tokens, elapsed runtime) via `BudgetLedger.from_snapshot`, with current
   limits preferred over the snapshot's (an approved plan-review patch must
   not be silently reverted). Reservations are deliberately not restored —
   they belonged to calls that died with their executor.

6. **Each round's evidence pool is an artifact under a stated 8 MB cap**
   (`evidence_pool.json`), degrading to references-only past the cap;
   entries that fit even without bodies only when dropped are dropped and
   counted, and so are entries that cannot be serialized at all (a
   non-JSON-native value degrades the artifact, never fails the run).

7. **Observers append events; only the lease holder writes run state.** A
   declined executor records `lease_declined` via
   `record_run_event` (append-only) rather than `update_run_progress`,
   which would stomp the live executor's progress and version mid-flight.

## Alternatives considered

- **Derive a size-aware synthesis timeout instead** (the superseded
  task-17386 framing): rejected — a durable run is not watched live, so
  outliving a provider timeout is acceptable once work survives and is not
  duplicated; almost all timeout machinery became unnecessary.
- **A separate reaper/sweeper for dead leases** (server-style): rejected —
  stale-lease reclaim is folded into `claim_run`'s atomic UPDATE, so there
  is no extra scheduled component and no window where a dead run is
  invisible.
- **Reuse `Library_Ingest_Jobs_DB`**: rejected — it has a retry counter but
  no lease columns and a lifecycle shaped for file ingestion.
- **Heartbeat on progress events**: rejected — see Decision 3; the longest
  phase emits nothing, so the lease must be renewed by time, not by
  activity.
- **Persisted leases for external-db mode**: rejected as infeasible — the
   external object has no lease columns or API; external mode degrades to a
   documented, per-instance in-memory exclusion instead of failing
   outright.

## Consequences

- Schema: four added columns on `research_runs` (`lease_owner`, `lease_id`,
  `leased_until`, `lease_attempts`), applied by idempotent
  `PRAGMA table_info`-guarded `ALTER`s (the service has no migration
  framework; introducing one was adjudicated out of scope for this change).
- Callers of `claim_run` must handle `LeaseBudgetExhausted` separately from
  a `None` return.
- External-DB mode leases are single-process only; cross-process exclusion
  exists only in SQLite-backed mode.
- A SIGKILLed executor's run stays declined for up to `lease_seconds`
  before takeover. Evidence read-back (skipping a completed round on
  resume) and scheduler auto-resume remain open (task-18060 ACs #3, #5–#11).
