---
id: TASK-22502
title: >-
  Harden the live-continuation retry against a cancel-during-CAS ledger/DB divergence
status: Done
assignee:
  - rmusser01
created_date: '2026-08-26'
labels:
  - console
  - correctness
priority: medium
dependencies: []
---

## Description

Source: close-out of the 2026-08-24 holistic performance review's burn-down (29 tasks,
TASK-22200..22228, all merged 2026-08-25/26). Evidence: `Docs/Design/2026-08-24-holistic-perf-review.md` plus the originating task's
Implementation Notes.

Found by TASK-22205's adversarial reviewer (probe P2). If a cancel lands DURING the
pre-dispatch checkpoint CAS hop, the surviving thread commits `dispatch_started` while the
effect ledger records the transition as abandoned; `retry_dispatch_recovery` then fails
"Accepted turn is retained for recovery" on EVERY retry — an in-process wedge until restart
or session reopen (no data loss; the restart path's `reconcile_for_session` classifies it
correctly).

Not reachable by any production path today: only `close_session` (which retires the
continuation, and reopen recovers) and shutdown cancel that task; the Stop button cancels
only the stream task. Filed so it does not become reachable silently.

## Acceptance Criteria

- [x] The live-continuation retry accepts an already-DISPATCH_STARTED checkpoint (or reconciles the ledger against the DB before re-claiming the effect)
- [x] A test reproduces the cancel-during-CAS interleave and proves in-process recovery without a restart
- [x] Coverage gaps noted by the same review are closed: a unit test for the `:memory:` offload gate, and the ordering probe's race-satisfiable `checkpoint_states` clause is tightened or documented as covered by the first-send atomicity tests

## Implementation Plan

1. `transition_checkpoint`: on a CONFLICT from the pre-dispatch CAS, read
   the live owner back through the repository; when it is OUR assistant
   message already DISPATCH_STARTED, accept it as the committed result
   instead of raising -- the exact divergence state a cancel-during-CAS
   leaves (surviving worker thread committed, effect re-run still holds its
   ACCEPTED expectations).
2. Test the seam deterministically: hold the effect's CAS on the worker
   thread, commit the same transition from a second caller (the surviving
   thread's state), release, and assert the send completes in-process.
3. Unit-test the `:memory:` offload gate (file-backed offloads, memory
   stays inline, the no-`.db` fake threads per TASK-22205's documented
   contract).
4. Document the ordering probe's `checkpoint_states` clause as corroborated
   by test_console_first_send_atomicity.py (the same-thread, no-race pin).

## Implementation Notes (2026-10-11, dev `9f8533686c`)

- **Fix**: `console_chat_controller.transition_checkpoint` (the
  `checkpoint_transition` durable effect body) now handles a CAS CONFLICT
  by reading the live owner (`repository.read_for_session`, offloaded like
  the CAS itself) and accepting it when it is the same assistant message
  already DISPATCH_STARTED -- identity is the ASSISTANT MESSAGE (the
  checkpoint row is keyed to it and only this controller transitions it),
  not the attempt id (the original send may carry a trace-admission
  attempt id the retry does not). The accepted result publishes the live
  checkpoint and the effect completes, re-converging the effect ledger
  with the DB. Every other CONFLICT (a foreign owner, a stale revision
  from a genuinely changed turn) still raises exactly as before.
- **Test** (`Tests/Chat/test_console_dispatch_retry_cancel_during_cas_t22502.py`,
  private-profile):
  `test_retry_recovers_after_cancel_during_checkpoint_cas` gates the
  effect's CAS on the worker thread, commits the same transition from the
  test thread (the surviving thread's DB state), then releases: pre-fix
  the send wedges ("the provider was never entered", turn retained);
  post-fix the turn completes, the provider streams once, and the ledger
  records `checkpoint_transition` complete (or the acceptance retires on
  settle -- equally terminal). RED-first verified by `git checkout HEAD --
  console_chat_controller.py` swap: 1 failed at base, 2 passed after.
  The duplicate committer is the test thread rather than a cancelled
  coroutine's surviving worker -- an end-to-end cancel repro was built
  first (transition-task cancel + gated CAS) but the submit's unwind
  takes the close-owned retirement path (TASK-22587 terminal-benign) or
  the sticky-settlement refusal depending on commit-vs-unwind ordering;
  neither leaves the ledger-recovery state P2 describes, because the
  drain (`_drain_dispatch_transition`) shields the transition from every
  production cancel path. The DB state the effect's CAS must tolerate is
  identical either way, and that is the seam this hardening owns.
- **AC3a**: `test_memory_backed_persistence_never_offloads_durable_calls`
  pins the `:memory:` gate (file-backed True, memory False) and the
  documented no-`.db` fake contract (True -- TASK-22205's "threads
  harmlessly"; an earlier draft of this test asserted fail-closed and was
  corrected against the shipped intent).
- **AC3b**: the ordering probe's `checkpoint_states` clause is documented
  in place (`test_console_durable_commit_offload.py`) as the
  independent-connection CORROBORATION, with
  `Tests/Chat/test_console_first_send_atomicity.py` named as the
  same-thread, race-free load-bearing pin.
- **No regression**: `test_console_durable_commit_offload.py` (8) +
  both new tests: 14 passed; `test_console_dispatch_recovery.py` +
  `test_console_dispatch_continuation_handoff.py` +
  `test_console_first_send_atomicity.py`: **114 passed**.
- **Ruff**: no new findings on `console_chat_controller.py` (class-count
  diff empty); the new test file clean after autofix.
- **ADR required: no** -- failure-mode hardening inside an existing
  recovery seam; the durable-dispatch design records (TASK-22205/22587
  lineage) already govern the ledger/checkpoint contract.
