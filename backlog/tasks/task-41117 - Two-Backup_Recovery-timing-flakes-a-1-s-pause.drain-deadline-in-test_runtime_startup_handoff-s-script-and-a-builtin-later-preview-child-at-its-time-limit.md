---
id: TASK-41117
title: "Two Backup_Recovery timing flakes: a 1 s pause.drain deadline in test_runtime_startup_handoff's script, and a builtin later-preview child at its time limit"
status: In Progress
assignee:
- '@claude'
updated_date: '2026-11-15 20:07'
created_date: 2026-11-15 02:20
labels:
- follow-up
- backup
- tests
- flaky
priority: low
references:
- Tests/Backup_Recovery/test_runtime_startup_handoff.py
- Tests/Backup_Recovery/test_operational_owners.py
- Tests/Backup_Recovery/test_builtin_later_snapshot.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** Both files run in the push-to-dev owner job (TASK-40307); a time limit that only fits a quiet host fails it on time.

**Evidence.**
- `test_runtime_startup_handoff.py:57`: `assert runtime.pause.drain(time.monotonic()+1)`, a 1 s drain deadline inside the child script that `test_operational_owners.py::test_run_log_real_bound_writer_is_excluded` runs. TASK-40307's run 3 (load 22-49) failed there once; runs 1 and 2 passed.
- `test_builtin_later_snapshot.py`: `test_fresh_process_rederives_finite_sources_and_executes_later_rollback` (`:1066`) timed out its child at load about 44 on base and head alike (TASK-40305 review r14act-rev1); the job's `--timeout=300` leaves the shared later-preview tests near their limit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The drain assertion waits on the condition it needs with a deadline measured under load, or its 1 s is justified in a comment with measurements
- [ ] #2 The builtin later-snapshot children's limits come from measured times with at least 2x headroom, recorded
- [ ] #3 Both files pass 5 of 5 at -n 4 at load 40 or more
<!-- AC:END -->

## Implementation Plan

1. Measure pause.drain's elapsed time under load and replace the 1 s deadline with a measured one (or justify it).
2. Measure the builtin later-snapshot fixture and child times under load and set the child cap and the test's own timeout to at least 2x the measured maximum, recorded in a comment.
3. Run both files 5 times at -n 4 at load 40 or more.

## Implementation Notes

NEEDS_CONTEXT: no criterion met; the drain flake is a product race, not a deadline.

- **Drain (AC#1), cause found.** Instrumented the script (temporary, reverted) at load 80-127. In the failing runs, at the drain call `storage._operations` held one operation and `_live_leases` four (passing runs: 0 and 1). Its stack: an asyncio default-executor thread in `runtime_producer_lifetime.invoke` -> `Chat/console_attention_preparation.py:139` -> `conversation_local_marks_service.list_console_unseen_marks` -> `ChaChaNotes_DB.get_connection` -> `_get_thread_connection` -> `connect_private_sqlite` (helper exchange). The Console attention preparation read opens a ChaChaNotes connection cached on that executor thread; `retire_local_caches()` retires only the calling thread's caches (`participants.retire_current_thread_caches`). After 60 s the operation was gone and the executors idle, but the thread-cached leases stayed live, so `pause.drain` stayed False. A blocking 60 s drain and a non-blocking 10 ms poll for 60 s (loop free) both failed, so no deadline fixes it. Under load the read lands in the window more often; quiet runs never start it before the drain.
  - Options: (a) product: `settle_producers` drains Console preparation reads and their executor-thread connection caches before the local pause (Backup_Recovery runtime maintenance with Console preparation reads; likely Console-owned code). (b) test: wait for Console attention preparation to be idle before `RuntimeMaintenance` starts; hides a race a real backup can hit. (c) keep the 1 s and record the race.
- **Second budget in the same tests.** `_run`'s fixed 45 s child timeout (`test_home_citation_retirement._run`) also fails these children at load 80-90: the startup children took 86-91 s.
- **Builtin later snapshot (AC#2), measured, not changed.** At -n 4 and load 76-98 the file took 20.6 min: 6 failed, all pytest-timeout (>300 s per test), mostly in fixture setup (135-240 s, `complete_builtin_case`/`later_case`), calls up to 264 s. The fresh-process child was cut at 165 s by the test's 300 s, not by its own limits (`idle=60`, cap 210). True durations above 300 s are unmeasured; a run with the per-test limit lifted is needed before setting limits with 2x headroom (likely a module timeout mark, as the fixtures alone need most of 300 s at that load).
- **AC#3 not run:** blocked by AC#1.
