---
id: TASK-41117
title: "Two Backup_Recovery timing flakes: a 1 s pause.drain deadline in test_runtime_startup_handoff's script, and a builtin later-preview child at its time limit"
status: To Do
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
