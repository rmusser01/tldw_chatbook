---
id: TASK-21127
title: >-
  Research runs - per-op leaked connects and loop-side periodic reads/writes while a run is active
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-24'
labels:
  - performance
  - research
  - database
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21127).

`Research_Interop/local_research_service.py:99-123` opens per-op and GC-leaks connections
(~with conn: sites); the engine is launched as a loop coroutine (Research_Window.py:594,
chat_screen.py:16200 - run_worker without thread), with a 30 s lease WRITE
(local_research_engine.py:387-393) and a 2 s `get_run` read poll (Research_Window.py:816-831)
on the loop while a run is active.

## Acceptance Criteria

Amended at implementation time (2026-08-24) after the measurement harness was built
BEFORE the fix. Two of the three original criteria prescribed work that banks
essentially nothing once the connection is held; they are struck through with the
measurement that retired them, and the replacements are what shipped. See
"Where the brief was wrong" in the Implementation Notes.

- [x] #1 The service holds ONE connection per thread instead of opening (and GC-leaking)
      one per operation; `close()` releases every one, including the schema connection
- [x] #2 ~~engine service calls route through to_thread~~ **NOT SHIPPED** -- after #1 the
      engine's whole loop-side DB cost is 1.6 ms per run with a 2.3 ms worst stall.
      Offloading it needs a nested loop or ~40 cascading `await`s, cross-thread UI
      dispatch and notification dispatch from a worker thread, to move 1.6 ms. Measured,
      recorded, and deliberately declined
- [x] #3 Every read-modify-write in the store is ONE `BEGIN IMMEDIATE` transaction, so a
      caller on the backend thread and the engine on the loop cannot interleave into a
      lost update (this is the prerequisite the offload creates, not optional polish)
- [x] #4 ~~The 30 s keepalive is batched with progress writes~~ **NOT SHIPPED** -- after #1
      one keepalive tick costs 0.020 ms (was 0.55 ms). Batching it would bank 0.02 ms per
      30 s while putting the double-execution lease guard at risk
- [x] #5 The 2 s auto-refresh, and every other UI-driven research call, runs off the loop:
      the scope service dispatches a synchronous backend on a single-thread executor
- [x] #6 Research behavior unchanged - existing tests green, and the quit / DB-error /
      cancel / empty-first-run paths are walked explicitly on a file-backed store

## Re-verification against dev 2be18842a (2026-08-23)

An independent read-only pass re-checked this finding. **All three legs still true; line cites
have drifted; one prescribed fix has a data-loss shape.**

**Confirmed, with corrected cites** (the filed `99-123` is now the schema-deferral block):
- `Research_Interop/local_research_service.py:124-158` — fresh connection per operation in
  file-backed mode, re-running `PRAGMA journal_mode = WAL` and `synchronous = NORMAL` on every
  open. **21** `with self._connect() as conn:` sites, **one** `.close()` in the file; `with conn:`
  is a transaction manager, not a closer, so the rest are GC-leaked.
- **Worse than filed**: `_update_row` (`:429-450`) opens **three** connections per single update
  (`_require_one`, the UPDATE, then `_require_one` again), each paying the private seam's
  owner-policy validation and `verify_trusted_directory`.
- `UI/Research_Window.py:595-600` — `run_worker(_run_engine(), ...)` with no `thread=True`.
- Every DB method on `LocalResearchService` is synchronous and the engine calls ~40 of them
  directly, so unlike an earlier finding in this programme, **an offload here would move real
  work** rather than zero.
- Keepalive: `local_research_engine.py:371-400` — a synchronous WRITE on the loop every 30 s for
  the life of a run. 2 s poll confirmed at `Research_Window.py:816`, correctly gated to an active
  non-terminal local run, reaching a `_call_service` seam that evaluates the method synchronously
  before its `await` plus an uncached `inspect.signature()` per call.

**Trap for the implementer**: do NOT naively add `conn.close()` to the 21 sites. In `:memory:`
mode `_open_connection` (`:137-148`) returns the **shared** `self._memory_conn`, and closing it
destroys the database; `close()` (`:161-165`) is the only legitimate closer.

**Revised severity**: every leg is real, but the whole surface is gated behind "user opens the
Research screen and starts a local run" — not a boot, keystroke or per-frame cost.

## Implementation Plan

1. Build the measurement harness BEFORE the fix: a probe running the real
   `LocalResearchEngine` against a real FILE-BACKED `LocalResearchService`
   (the engine tests use `:memory:`, which never opens per-op connections at
   all), counting connection opens through the private seam, attributing every
   service call to a thread, and measuring loop stalls with an independent
   ticker. A second probe for the UI path (2 s poll + bundle load) through the
   real controller -> scope service -> service graph.
2. A/B against a pinned base worktree at `d589c56c5`, arms interleaved.
3. Held connection per thread (`dict[thread_ident, Connection]`, not
   `threading.local`), `isolation_level=None`, explicit `_transaction()`
   BEGIN/COMMIT/ROLLBACK, lifecycle gate + re-arming `close()`. Reuse
   TASK-21125's shape; keep `:memory:` sharing exactly one connection.
4. Decide the offload legs from the re-measured numbers, not the brief.
5. Make every read-modify-write one `BEGIN IMMEDIATE` transaction before any
   offload can interleave them.
6. Scope-service offload on a SINGLE-thread executor; shutdown wiring through
   `asyncio.to_thread`.
7. Mutation-verify every new test against a deliberately broken implementation;
   walk quit / error / cancel / empty explicitly.

## Implementation Notes

`LocalResearchService` now holds ONE connection per thread for the life of the
service instead of opening (and GC-leaking) a fresh one per operation, every
read-modify-write is a single `BEGIN IMMEDIATE` transaction, and the UI-driven
half of the research surface runs on a worker thread instead of the Textual
loop. The engine offload and the keepalive batching the ACs asked for were
measured and deliberately NOT shipped.

**Connection lifecycle.** `_connect()` returns a held connection from a
`dict[thread_ident, Connection]` (not a `threading.local`: shutdown has to reach
connections it does not own), opened through the same `connect_private_sqlite`
seam with `check_same_thread=False` and `isolation_level=None`. All 21
`with self._connect() as conn:` sites became `with self._transaction(...)`, an
explicit BEGIN / COMMIT / ROLLBACK manager -- sqlite3's connection context
manager is a *transaction* manager, not a closer, which is what leaked the
connections. `:memory:` still shares exactly one connection (closing it destroys
the database) and serialises its transactions under an RLock; `close()` there
also clears `_schema_ready` so the re-armed store rebuilds. `_init_schema` now
CLOSES its own connection in file mode rather than leaking it, and fences the
migration span in an explicit transaction (`apply()` documents that the caller
owns it, and with `isolation_level=None` nothing opens one implicitly).

**Atomicity is a prerequisite, not polish.** `_update_row`, `_soft_delete`,
`_update_run_state`, `update_run_progress`, `claim_run`, `release_lease`,
`patch_and_approve_checkpoint`, `create_session`, `launch_run`,
`create_checkpoint`, `save_artifact` and `record_run_event` each read and then
write. Splitting that across transactions was harmless only while every caller
ran inline on the event loop; the moment ANY caller moves to a thread it is a
live lost update (TASK-21125 measured 0/60 -> 59/60). They are now one
`BEGIN IMMEDIATE` transaction each -- which also removes two of the three
connections `_update_row` was opening per update. `release_lease` is the case
where IMMEDIATE is mandatory rather than preferred: it SELECTs then UPDATEs, and
a deferred BEGIN under `isolation_level=None` opens a read snapshot whose later
write fails `BUSY_SNAPSHOT` ("database is locked"), which SQLite's busy handler
does NOT retry.

**Thread offload, scoped by measurement.** `ResearchScopeService._service_for_mode`
returns a `_ThreadOffloadedBackend` proxy dispatching non-coroutine callables on
a **single-thread** executor -- one worker deliberately, so the ordering the
event loop was silently providing survives the move. `scope.local_service` keeps
its identity for the wiring tests. Two research-specific details the writing
precedent did not have: the passthrough predicate must also accept async
GENERATOR functions (`LocalResearchService.stream_run_events` is
`async def ... yield`, for which `inspect.iscoroutinefunction` is False, so a
coroutine-only predicate hands `observe_run_events` a coroutine wrapping the
generator -- mutation-confirmed, `TypeError: 'async_generator' object is not
iterable`); and `_call_service`'s per-call `inspect.signature` is now cached on
the CLASS (a bound method is a fresh object per access, so caching on it would
never hit), with a fallback for test doubles the class lookup cannot describe.

**Shutdown.** `TldwCli._close_local_research_service()` peeks the slot (never
constructs a service to close it) and runs `close()` through
`asyncio.to_thread` -- inline it would freeze the loop for the whole settle
timeout and starve the very operation it waits for. On a settle timeout a busy
thread KEEPS its connection; closing it anyway produces
`ProgrammingError: Cannot operate on a closed database` inside live work.

### Where the brief was wrong

The re-verification's three legs are all real, but **two of the three prescribed
FIXES bank essentially nothing once the connection is held** -- the per-op
connect was 99.7% of the cost, not a share of it:

| | base `d589c56c5` | after |
|---|---|---|
| one `connect_private_sqlite` open + pragmas + close | **0.631 ms** | -- |
| one SELECT on a held connection | 0.002 ms | 0.002 ms |

- **Engine offload (AC #1, second half): declined.** 20 engine runs, 5
  interleaved A/B pairs: loop-side DB **877-906 ms -> 29-33 ms** (~29x), worst
  contiguous loop stall **46-51 ms -> 2.0-2.5 ms**, with the 800 service calls
  still on the loop thread in both arms. That residue is 1.6 ms per run. Moving
  it needs a nested event loop (or ~40 cascading `await`s through 8 currently
  synchronous engine methods), cross-thread Textual dispatch, and notification
  dispatch from a worker thread -- and it applies to two call sites, since
  Console's `/research` builds the engine the same way. Not worth it.
- **Keepalive batching (AC #2, first half): declined.** One 30 s tick:
  **0.55 ms -> 0.020 ms**. Batching it with progress writes would save 0.02 ms
  per 30 s while entangling the lease that prevents double execution.
- **The 2 s poll (AC #2, second half) is worth ~nothing ON ITS OWN** -- 0.023 ms
  of loop time per tick after the held connection. It ships because the same
  one-line proxy also covers the bundle load, which is **~15 ms of loop time at
  a 5.5 MB bundle even with the connection held** and grows with artifact size.
  That, not the poll, is the UI-side case for an offload.

### Evidence

Probes and logs in `test-logs/` (gitignored); A/B against a pinned base worktree
at `d589c56c5`, arms **interleaved** per the TASK-21130 lesson.

- **Engine, 20 runs x 5 interleaved pairs:** connection opens **1741 -> 3**
  (87/run -> 0 marginal; the 3 are schema + one per thread), loop-side DB
  **877-906 ms -> 29-33 ms**, worst loop stall **46-51 ms -> 2.0-2.5 ms**, all
  runs `completed` in both arms. Work is GONE, not relocated: the service-call
  count on the loop thread is 800 in BOTH arms.
- **UI path (30 poll ticks + one bundle load, 5.5 MB of artifacts):** connection
  opens **33 -> 0**; database operations on the loop thread **33/33 -> 0/32**
  (all on `research-backend_0`); an independent 1 ms ticker got **0 wakeups in
  the base arm** across the whole ~33 ms window -- the shipped UI research path
  never yields to the loop at all -- versus **10-11 wakeups, worst stall
  0.17-0.24 ms** after.
- **Pragmas read back on the LIVE held connection:** `journal_mode=wal`,
  `synchronous=1`, `isolation_level=None`, `busy_timeout=5000` (so IMMEDIATE
  contention is retried); WAL confirmed persistent in the file through an
  independent connection.
- **Mutation results: 15 of 15 new guards go RED against a deliberately broken
  implementation, 0 vacuous** (`__pycache__` cleared per mutation). Signatures
  worth naming: reverting `update_run_progress` to separate read/write
  transactions gives *"40 writes all reported success but version advanced 11
  time(s): a lost update"*; letting `close()` close a wedged thread's connection
  gives the exact TASK-21101 `ProgrammingError('Cannot operate on a closed
  database.')`; closing inline from unmount gives *"the event loop was frozen
  during close() (0 ticks)"*; dropping `isasyncgenfunction` gives
  *"TypeError: 'async_generator' object is not iterable"*; leaking
  `_init_schema`'s connection is caught by the `-wal` sidecar surviving
  `close()`.
- **Two mutations initially stayed GREEN and were investigated rather than
  written off.** `_begin`'s stale-handle heal and `_transaction`'s rollback are
  both second lines of defence behind mechanisms that already covered the
  end-to-end walks (`close()` POPS what it closes; `_begin` re-rolls-back a
  stray transaction), so no single-point mutation could red a walk assertion.
  Two targeted tests were added for the mechanisms themselves -- a connection
  closed while still mapped, and `in_transaction` after a failed body -- and
  both mutations then went red.
- **Lifecycle walk, file-backed** (`test_local_research_service_lifecycle.py`):
  quit mid-run (`close()` from another thread during a live engine run -> the
  run still completes), a DB error mid-run (run fails legibly, store still
  usable afterwards), cancel between phases (resolved exactly once), and the
  empty/first-run case (construction creates no file; TASK-21105's lazy-open
  contract survives).
- **Tests:** research set **323 passed -> 349 passed / 0 failed** (+26).
  `Tests/App` + `Tests/ProductionApp`: **243 passed / 4 failed**, the same four
  A/B-proven on pristine base `d589c56c5` with byte-identical signatures
  (console-stop root state, two reactive-maturity scanners, retired-destination
  state -- the scanner names `Tests/ProductionApp/test_notes_sync_runtime_lifecycle.py`,
  not any file this task touched). Collect sweep **59,592 -> 59,618, zero
  collection errors on both sides**.
- `ruff check` clean on every touched file. `ruff format --check` reports
  `local_research_service.py` and `test_local_research_service.py` as
  unformatted -- **both are already unformatted at base**, and every hunk in the
  diff sits on pre-existing lines (verified hunk by hunk); the added regions are
  format-clean.
- `./scripts/preflight.sh` green after reviewing the 6 new diagnostic-inventory
  rows individually (5 in `local_research_service.py`, 1 in `app.py`): all are
  log constants, an integer connection count, or `type(exc).__name__` -- none
  interpolates the database path, user content, a secret, or a URL.

### Out-of-scope findings (not fixed here)

- `Research_Window._start_local_engine` and Console's `/research` handler
  (`chat_screen.py:16068`) each build a `LocalResearchEngine` around the RAW
  `local_research_service`, bypassing `ResearchScopeService` entirely -- so the
  scope service's policy enforcement and normalisation do not apply to engine
  writes, and neither does this task's offload. That is a design question, not a
  perf one.
- `_begin`'s `sqlite3.ProgrammingError` arm is unreachable through the shipped
  `close()` (which pops every connection it closes). It is kept as a documented
  guard and now has a direct test, mirroring the TASK-21125 MINOR-2 precedent.

### Review fix round (coordinator review, pre-merge)

One finding, taken as reported: **`_transaction()`'s nested branch silently
ignored `immediate`.** A body inside a deferred transaction that opened
`_transaction(immediate=True)` would JOIN the deferred outer, running its
read-then-write under a read snapshot -- exactly the `BUSY_SNAPSHOT` the flag
exists to prevent.

I re-derived the reachability claim independently before changing anything, with
an AST pass over the class that walks the self-call graph **transitively** rather
than checking direct calls: 9 methods open a deferred transaction, 12 open an
immediate one, and there are **0 live deferred -> immediate paths**. Confirmed
latent, not live.

Shipped the loud option -- a nested `immediate=True` inside a deferred outer
raises `RuntimeError`; the outer's mode is recorded on `_tx_state` so
immediate-inside-immediate still joins normally. The decisive argument for
raising over documenting is the *failure mode of the alternative*:
`BUSY_SNAPSHOT` only occurs when another writer holds the lock, so a silent
downgrade survives every single-threaded test and then appears intermittently in
the field as "database is locked" -- a message that reads like a transient
condition to retry, and is the one failure SQLite's busy handler will NOT retry.
Refusing converts that into a deterministic failure on the first execution, with
a message naming the actual mistake and the actual fix (open the OUTER
transaction as immediate). No lock upgrade is attempted; SQLite cannot do that.

Deferred-inside-immediate stays legal and is deliberately pinned by a second
test: it is the shipped hot path (`_require_one` inside `save_artifact`, on every
artifact the engine saves), so over-restricting here would break real behaviour
rather than protect it. Both new tests are mutation-verified: removing the guard
gives `Failed: DID NOT RAISE <class 'RuntimeError'>`, and failing to record the
outer's mode makes the guard misfire on a LEGAL nesting, reddening the
over-restriction test. Research set **349 -> 351 passed / 0 failed**; full
mutation set now **17 of 17 RED, 0 vacuous**; `./scripts/preflight.sh` green
(the guard is a `raise`, not a diagnostic, so the inventory is unchanged).

**Files**: `tldw_chatbook/Research_Interop/local_research_service.py`,
`tldw_chatbook/Research_Interop/research_scope_service.py`,
`tldw_chatbook/app.py`, `Tests/Research/test_local_research_service.py` (+11),
`Tests/Research/test_local_research_service_lifecycle.py` (new, 12),
`Tests/Research/test_research_scope_service.py` (+5),
`Tests/Research/test_app_research_wiring.py` (+2),
`Tests/DB/test_private_sqlite_interop_owners.py` (the research owner factory now
returns a real closer), `Docs/security/production-diagnostic-inventory.json`.
