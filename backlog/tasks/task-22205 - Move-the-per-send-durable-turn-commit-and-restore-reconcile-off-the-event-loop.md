---
id: TASK-22205
title: >-
  Move the per-send durable-turn commit and restore reconcile off the event loop
status: Done
assignee: ['@claude']
created_date: '2026-08-24'
labels:
  - performance
  - console
  - database
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22205).

New since the pin (console dispatch checkpoint work). Per send,
`Chat/console_chat_controller.py:5635` runs `store.commit_durable_turn(acceptance)`
synchronously on the event loop before the provider request goes out: one
`BEGIN IMMEDIATE` transaction (`Chat/chat_persistence_service.py:384`) containing ~10
statements including two message INSERTs (each firing the FTS trigger and a full-content
`sync_log` JSON write — ~3x write amplification of the message text), an attachments
`executemany`, and a readback (`Chat/console_dispatch_repository.py:103-262`). A second
IMMEDIATE transaction runs at `:657` pre-dispatch and a third at settle
(`console_chat_store.py:2141`). Separately, every conversation restore/switch runs
`reconcile_for_session` (`console_dispatch_repository.py:324-345`) — an IMMEDIATE (write)
transaction taken to read recovery state that usually writes nothing, plus a recursive
active-path CTE, inline on the loop. Steady-state cost is tens of ms; under the 22200
backfill window it is unbounded up to the 15 s busy timeout — on the event loop.

## Acceptance Criteria

- [x] The durable-turn commit runs off the event loop (worker/`to_thread`) with its ordering guarantees preserved and the dispatch-checkpoint test suite green
- [x] `reconcile_for_session` takes a write transaction only when it actually has a write to make; the read path uses a read transaction
- [x] Send-to-dispatch latency measured before/after (steady state and with an artificial write-lock holder)
- [x] No new shutdown/error-path regressions: the review's standing rule — when you defer, walk the teardown and failure paths in real teardown order

## Implementation Plan

1. Nesting census (done before any code): production callers of `store.commit_durable_turn`
   are exactly one — `console_chat_controller._accept_durable_turn:5635`; the only caller of
   `persistence.commit_durable_turn` is the store (`console_chat_store.py:3649`). Neither the
   controller nor the store ever opens a `db.transaction()` themselves (zero `.transaction(`
   hits in both files), so the persistence service's `BEGIN IMMEDIATE` at
   `chat_persistence_service.py:384` is always the OUTERMOST manager-owned transaction on the
   calling thread — nothing borrows an enclosing same-connection transaction on this path.
   Moving the whole `store.commit_durable_turn` call to a worker thread moves the entire
   outer transaction (and its internal borrowed nests, e.g. `create_conversation` →
   `db.add_conversation`) to that one thread intact.
2. Offload gate: follow the existing `chat_conversation_scope_service._is_memory_backed`
   precedent — `asyncio.to_thread` only when `store.persistence.db` is not a `:memory:`
   CharactersRAGDB (a memory DB is per-connection, so a worker thread would see an empty,
   unmigrated database). Fakes without `.db` thread harmlessly.
3. Red-first probes (new test module):
   (a) event-loop-stall probe — second connection holds `BEGIN IMMEDIATE`, submit a send,
       measure max event-loop stall (base: ~lock-hold duration; after: bounded small);
   (b) ordering — provider dispatch must not begin before the durable commit is visible in
       the DB (gateway records commit-visibility at first stream call);
   (c) reconcile read path — sqlite trace on a clean restore shows no `BEGIN IMMEDIATE`.
4. Implement:
   - `_accept_durable_turn`: `commit = await` a small helper that runs
     `self.store.commit_durable_turn(acceptance)` via `asyncio.to_thread` behind the memory
     gate. Ordering guarantees: dispatch-after-commit holds because the coroutine only
     resumes after the thread returns; per-session serialization holds because
     `begin_preparation` is a single live slot per session and the prompt-queue dispatcher
     refuses/queues while `preparing_before_acceptance`/`accepted_live_turn`; result
     visibility holds because everything after the `await` sees the returned commit.
   - `transition_checkpoint` (pre-dispatch `cas_state`, the second per-send IMMEDIATE
     transaction, squarely inside send-to-dispatch latency): same offload via the async
     effect callback (`_run_durable_postcommit_effect` already awaits awaitables).
   - `ConsoleDispatchRepository.reconcile_for_session`: two-pass shape. Pass 1 is a
     DEFERRED read transaction that never issues a write statement (parametrized
     `allow_writes=False` returns a sentinel at the exact two write points: active
     continuation precedence, terminal-state checkpoint delete). Only when pass 1 reports a
     write does pass 2 open a fresh `BEGIN IMMEDIATE` and re-run the full logic from
     scratch (re-reads inside the write txn; no deferred-upgrade — the wave-1
     snapshot-upgrade lesson).
5. Measure send-to-dispatch latency before/after, steady state and with a 2 s artificial
   write-lock holder.
6. Targeted suites: dispatch-checkpoint, durable-turn (acceptance + fix rounds 1-4),
   dispatch recovery, controller, first-send atomicity, persistence; `--collect-only`
   sweep; tee everything. `./scripts/preflight.sh`.
7. Mutation test: remove the await barrier (dispatch no longer waits on the commit) → the
   ordering probe must go red.
8. Shutdown/failure walk: task cancelled while the commit thread runs → `to_thread`
   survives cancellation, the transaction completes or rolls back atomically on the worker
   thread, `except Exception` does not swallow `CancelledError`, and a post-cancel
   committed checkpoint lands in the exact crash-recovery state restore reconcile already
   handles. Off-loop commit failure re-raises the same exception object in the awaiting
   coroutine → identical PERSISTENCE-pause + "Couldn't save the prepared turn." path.

## Implementation Notes

Both per-send pre-dispatch `BEGIN IMMEDIATE` transactions now run off the event loop, and
the restore reconcile no longer takes the write lock just to read.

**Controller** (`Chat/console_chat_controller.py`): new `_run_durable_db_call` helper runs a
durable persistence call via `asyncio.to_thread`, gated by `_durable_db_call_offloadable`
(the `_is_memory_backed` precedent: a `:memory:` CharactersRAGDB is per-connection, so
memory-backed persistence stays inline; test fakes without `.db` thread harmlessly). Used
at both per-send transaction sites: the `store.commit_durable_turn(acceptance)` call in
`_accept_durable_turn`, and the pre-dispatch `repository.cas_state(...)` inside the
`checkpoint_transition` postcommit effect (the effect runner already awaits awaitable
callbacks). The `await` IS the ordering barrier — dispatch cannot precede the commit;
per-session serialization is upstream (`begin_preparation` single live slot + the
prompt-queue dispatcher refusing/queueing while a turn is preparing/accepted).

**Repository** (`Chat/console_dispatch_repository.py`): `reconcile_for_session` is now
two-pass. Pass 1 is a DEFERRED read transaction that never issues a write statement —
`_reconcile_checkpoint_row(..., allow_writes=False)` returns a `_ReconcileWriteNeeded`
sentinel at the exact two write points (active continuation precedence; terminal-state
checkpoint delete). Only then does pass 2 open a fresh `BEGIN IMMEDIATE` and re-run the
full logic from scratch inside the write lock — never an in-place deferred upgrade (the
task-21100 wave-1 snapshot-upgrade lesson; also why 22200's chunk commits could kill a
deferred read-then-write instantly).

**Nesting census** (prerequisite): the only production caller of `store.commit_durable_turn`
is `_accept_durable_turn`; the only caller of `persistence.commit_durable_turn` is the
store; neither the controller nor the store opens `db.transaction()` themselves, so the
persistence service's outer IMMEDIATE is always outermost on its thread — nothing borrowed
an enclosing transaction, and the internal borrowed nests (`create_conversation` →
`add_conversation`) move to the worker thread intact.

**Tests**: new `Tests/Chat/test_console_durable_commit_offload.py` (6 tests: loop-stall
probe, independent-connection ordering probe, cancellation/teardown walk, 3 reconcile
lock-shape probes — the stall and two reconcile probes were red-first on base). Two
existing injection mechanisms changed from `CREATE TEMP TRIGGER` to permanent
`CREATE TRIGGER` (`test_console_first_send_atomicity.py`,
`_install_failure` in `test_console_durable_turn_acceptance.py`): TEMP triggers are
per-connection and the commit now runs on a worker-thread connection — behavioral
assertions unchanged, and the direct-call (same-thread) tests fire them identically.

**Measurements** (file DB, this machine): steady-state send-to-dispatch median 3.1 ms →
3.5 ms (+0.4 ms thread-hop, negligible); with a 2 s write-lock holder the send correctly
still takes ~2 s (durability requires the lock) but the **max event-loop stall drops
2049 ms → 11 ms**. Clean-restore reconcile trace: `BEGIN IMMEDIATE` → plain `BEGIN`.

**Mutation**: making the commit fire-and-forget behind a fabricated checkpoint (the
"dispatch no longer waits" mutation at the store's `durable_commit` call) reds the
ordering probe (provider never legitimately entered); reverted byte-identical.

**Shutdown/failure walk**: cancellation mid-commit is exercised by
`test_cancel_during_offloaded_commit_leaves_consistent_state` — the surviving thread
commits atomically, no loop-exception-handler leakage, provider never called, and
`reconcile_for_session` surfaces the committed-but-undispatched turn as a recovery owner
(the pre-existing crash-window contract). Off-loop failures re-raise the original
exception in the awaiting coroutine → the identical PERSISTENCE-pause path
(`test_precommit_failure_keeps_input_and_never_calls_provider`, now via a worker-thread
connection, still green).

**Suite evidence**: targeted suites 571 passed (durable-turn acceptance + fix rounds 1-4,
first-send atomicity, dispatch recovery + fix rounds 1-4, queue recovery, continuation
handoff/review fixes, persistence service, controller, prompt-queue coordinator,
automatic library preparation, offload probes); store suites + UI native chat flow: 650
passed, 11 failed — **all 11 UI-flow failures reproduce identically at base `e94cd66b1`**
(pre-existing dev reds, verified test-by-test in a clean baseline worktree; "no such
table: conversations" memory-DB symptom, unrelated to this change). Collect-only sweep:
23,546 collected, no errors. `./scripts/preflight.sh` all green.

**Out of scope / residue**: the third per-send IMMEDIATE transaction — settle
(`console_chat_store.py` `settle_dispatch_recovery` → `settle_with_assistant`) — still
runs on the loop (post-response, not in send-to-dispatch latency; offloading it needs an
async refactor of a sync store API). Same for `publish_durable_turn_owners`' persistence
writes if any. Worth a follow-up if settle shows up in loop-stall traces.
