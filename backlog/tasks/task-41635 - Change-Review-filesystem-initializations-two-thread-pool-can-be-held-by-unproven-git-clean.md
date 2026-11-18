---
id: TASK-41635
title: "Change Review filesystem initialization's two-thread pool can be held by unproven git cleanups"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- workspaces
- console
priority: medium
references:
- tldw_chatbook/Workspaces/change_review_consent.py
dependencies:
- TASK-41307
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** Change Review runs filesystem initialization on a fixed daemon pool (`Workspaces/change_review_consent.py`, `worker_count: int = 2`), so that a slow root never creates an unbounded thread population. A git cleanup whose return was never proven stays retained (ADR-134/285), and TASK-41307's Release frees only the tool-worker slot. Two such cleanups can hold both pool threads, and every later Change Review initialization, for any root, then waits behind them with nothing on screen to say why.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A retained unproven git cleanup cannot block Change Review initialization for other roots, or the UI states why initialization is waiting and what releases it
- [x] #2 A test holds two unproven cleanups and starts a third initialization, asserting the chosen behaviour
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new decision. It brings Change Review initialization under ADR-134's existing TASK-40716 implementation rule that custody holds a daemon worker, never a joined pool thread; recorded as a dated implementation-note amendment to ADR-134.

Chosen behaviour: a retained unproven cleanup cannot block initialization for other roots. A native initialization job runs on a daemon custody worker (`run_on_custody_thread`, the helper Raw CLI, `@git` references and the startup SSH read already use). The pool thread waits for it; when the job's operation is retained, it marks that root failed and takes the next job, while the custody worker keeps holding the scopes. Threads stay bounded: the two pool threads, at most one live custody worker per pool thread, plus one per retained cleanup. Each retention already holds a tool slot (the capacity's tool limit) until the user releases it, and its scopes until restart, so the extra thread costs nothing the retention does not already hold.

1. Test first (`Tests/Workspaces/test_change_review_consent.py`): a native-path service with two pool threads and a real `RuntimeCapacity`; two roots' initializers retain an unproven cleanup and hold (as `run_native_git` does); then a third root is admitted. Assert the third root becomes READY, the two held roots read FAILED, and the two holding threads are not pool threads. Show it red on the unfixed code.
2. Fix `_worker_loop`: run the ticket's activate/run/return on `run_on_custody_thread(..., lambda: ticket.operation)`. A `CleanupUnprovenRefused` marks the root failed (not "busy; retry", which a retry cannot clear).
3. Update the isolated real-Git custody test's pin that the pool's queue keeps one unfinished task (the pool thread used to hold); it now asserts the queue drained and the holder is not a pool thread.
4. ADR-134 dated amendment; covering Workspaces/consent files on base and head; preflight; `Tests/Architecture` serial.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Choice: a retained unproven cleanup can no longer block initialization for other roots.** This needed no UI copy, because the wait no longer happens. A native initialization job (`work.source` set) now runs its ticket's activate/run/return on `run_on_custody_thread`, the daemon-worker helper Raw CLI, `@git` references and the startup SSH read already use (TASK-40716). The pool thread waits for it. When the job's operation is retained, the helper raises `CleanupUnprovenRefused`, the pool thread marks that root failed and takes the next job, and the custody worker keeps holding the original scopes until restart. The pool stays at two threads. At most one custody worker per pool thread is live at a time, plus one per retained cleanup. A retained cleanup already holds its tool slot (bounded by the capacity's tool limit) until Release, and its scopes until restart, so its thread adds no new kind of retention. That is why this is acceptable without an unbounded thread population.

- The retained root reads "Change history preparation failed; this turn continues without it.", not "busy; retry from Settings", because a retry cannot clear a retention. `CleanupUnprovenRefused` is a `CapacityRefused` subclass, so the busy branch now excludes it.
- Not changed, and recorded in the ADR amendment: a retained job that ran under its own owner still holds its opaque activity claim until restart (the TASK-40716 decision). That claim refuses later own-owner initializations on its own (they read "busy"). So the pool could fill only with retentions that borrow concurrent ordinary turns' owners, whose claims are not opaque. That is the case this change fixes.
- ADR-134: dated implementation-note amendment ("Change Review initialization custody amendment"). No owner decision changes.

**Tests.**
- `test_unproven_cleanups_cannot_hold_initialization_for_other_roots` (`Tests/Workspaces/test_change_review_consent.py`): a native-path service (two pool threads, real `RuntimeCapacity`, fake source/ticket). Two roots' initializers retain "native Git" and `hold()`, as `run_native_git` does. The test waits until both are retained, adds a third root and admits it. It asserts the third root is READY within 5 s, the two held roots are FAILED with the failed reason, and both holders are daemon threads outside the pool. Red on the unfixed code (the third root's READY wait times out). Mutation checks: reverting only `change_review_consent.py` fails the READY wait, and restoring only the old busy flag fails the reason assertion. Teardown calls `prove()` on both retentions, so no thread is left holding.
- `test_unavailable_cleanup_proof_retains_original_worker_in_isolated_process` (real Git wrapper, isolated process) pinned `_queue.unfinished_tasks == 1`, which held only because the pool thread itself was holding. It now asserts the holder is not a pool thread and the queue drains. Passes on head. On the unfixed product file it fails ("change-review-init-1" in `_workers`).

**Evidence** (2026-11-18, base d0486978dd, load 56-105).
- `test_change_review_consent.py`: 13 passed on head.
- `test_native_change_review_git_custody.py` + `test_native_change_review_initialization.py` + the consent file, serial: head 8 failed / 44 passed, base 5 failed / 46 passed (load 70-82). Every failure is a 3-5 s wait for the real git init under load, or `executions == ()` (also red on base). The three that failed only on head, rerun in alternating head/base rounds: round 1 head 1 failed vs base 2, round 2 head 1 vs base 1, round 3 head 1 vs base 1. The one that keeps failing (`test_admitted_callback_cannot_publish_ready_after_original_source_changes[registry]`) fails on base in every round.
- The other consent-touching files (`test_console_backup_worker_retirement`, `test_change_turn_tracking`, `test_console_hook_preparation_lifetime`, `test_console_ordinary_tree`, `test_console_environment_consent_revoked_hop`, `test_console_unproven_cleanup_release`, `test_settings_change_review_journeys`, `test_native_change_review_ownership`; `-n 4`): head 62 failed / 263 passed, base 61 failed / 264 passed. The only head-only failure, `test_original_stop_hook_holds_same_ordinary_tree[True-False]`, passes alone on head (2 of 2) and on base (2 of 2).
- `./scripts/preflight.sh` passes.
- `Tests/Architecture` (serial, load 40-90): 4 failed, 1475 passed, 5 skipped. Three also fail on base d0486978dd: `test_reviewed_diagnostic_changes_are_metadata_only` and the two `test_timer_path_static_update_inventory.py` tests. The fourth, `test_task_15103_review_ledger_canonical_schema_and_arithmetic`, failed only in the full run and passes alone on head (136 s), the same load timeout TASK-41317 recorded.

Files: `tldw_chatbook/Workspaces/change_review_consent.py`, `Tests/Workspaces/test_change_review_consent.py`, `Tests/Workspaces/test_native_change_review_git_custody.py`, `backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md`.
<!-- SECTION:NOTES:END -->
