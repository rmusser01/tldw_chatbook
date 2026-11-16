---
id: TASK-41117
title: "Two Backup_Recovery timing flakes: a 1 s pause.drain deadline in test_runtime_startup_handoff's script, and a builtin later-preview child at its time limit"
status: In Progress
assignee:
- '@claude'
updated_date: '2026-11-16 22:58'
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
- [x] #1 The drain assertion waits on the condition it needs with a deadline measured under load, or its 1 s is justified in a comment with measurements
- [x] #2 The builtin later-snapshot children's limits come from measured times with at least 2x headroom, recorded
- [ ] #3 Both files pass 5 of 5 at -n 4 at load 40 or more
<!-- AC:END -->

## Implementation Plan

1. Measure pause.drain's elapsed time under load and replace the 1 s deadline with a measured one (or justify it).
2. Measure the builtin later-snapshot fixture and child times under load and set the child cap and the test's own timeout to at least 2x the measured maximum, recorded in a comment.
3. Run both files 5 times at -n 4 at load 40 or more.

## Implementation Notes

Commits 3a4ac827ec and 18d515fc69. AC#1 and AC#2 met; AC#3 met for the handoff file only, so the task stays In Progress.

**Correction to the first round-15 report.** That report called the drain flake a product race and proposed a product fix; the coordinator ruled option (a) on that basis. It is not a product race. The only production caller, `runtime_maintenance.monitor_app`, runs `retire_local_caches()` and then polls `pause.drain(now)` every 10 ms, calling `retire_current_thread_caches` again each turn, within a 30 s budget (since b5251e9a6e). `retire_current_thread_caches` deliberately skips a store whose cache another thread is still operating on, and `Tests/Backup_Recovery/test_runtime_cache_revisit.py` pins that skip and the monitor's revisit, so the proposed product change would have broken a pinned design. No product code changed and no shared mechanism was added, so there is nothing to reconcile with fu-r15-actor's TASK-41110. That case (an idle worker-thread cache with no operation) is not covered by the revisit and remains theirs.

- **Cause (AC#1).** Reproduced deterministically in the real app: a Console attention preparation read (`console_attention_preparation` -> `list_console_unseen_marks`) was held mid-open on its worker thread across the pause. With an operation live on ChaChaNotes, `retire_local_caches` kept the main thread's ChaChaNotes cache. The read then finished and closed its own handle, but the script never revisited, so the main-thread lease stayed and `drain(+1)` was False.
- **Fix (AC#1).** The script now drains as monitor_app does: poll, sleep 10 ms, revisit, within monitor_app's own 30 s. New case `test_actual_app_startup_returns_before_ordinary_admission[attention]` forces that interleaving: red with the old line (`AssertionError: DRAIN_REFUSED`), green after.
- **`_run` (test_home_citation_retirement).** It had a fixed 45 s, and the app-startup children took 86-91 s at load 80-90. A child now fails after `timeout` seconds with no CPU (the TASK-40306 pattern), with a wall backstop of 4x `timeout`. `_wait_while_busy` moved to `Tests/Backup_Recovery/restart_observation.py`, which is already on perf-guard's paths (importing `test_config_sibling_capture` from `_run` failed `test_perf_guard_runs_when_a_file_its_steps_load_changes`). `test_config_sibling_capture` imports it from there. 194 test files use `_run`; the covering run (restart observation, home citation retirement, handoff, operational run-log node, activation startup/notes, cache revisit, the child-wait node, inventory unused owners) gave 103 passed at load 44-81.
- **AC#2.** The fresh-process later-rollback child: at -n 4 and load 76-98 the per-test 300 s cut it at 165 s; at load 100-129 it passed with setup 212 s and child 301 s. Its CPU-idle detector stays at 60 s; wall backstop 600 s (2x 301); test limit `@pytest.mark.timeout(1050)` (2x 212 + 600). Recorded in the test's comment. Follow-up: re-measure at load ~40.
- **AC#3.**
  - Handoff file plus `test_operational_owners.py::test_run_log_real_bound_writer_is_excluded` at -n 4: 6 of 6 runs passed (9 passed each, including `[attention]`), at load 75-129 (5 consecutive runs at load 84-116).
  - `test_builtin_later_snapshot.py` at -n 4, load 75-129, one run (29 min): 6 failed and 3 errors, all pytest-timeout (>300 s) in in-process tests whose fixture setup alone took 155-240 s. The fresh-process test passed. The four further runs were stopped: at this load they cannot pass without per-test limits for those in-process tests, which the ruling did not cover.
  - Follow-up: measure the file at load ~40, then give its fixture-heavy tests (or the module) a measured `timeout` mark, or make `complete_builtin_case` cheaper.
- Files: `Tests/Backup_Recovery/test_runtime_startup_handoff.py`, `test_home_citation_retirement.py`, `restart_observation.py`, `test_config_sibling_capture.py`, `test_builtin_later_snapshot.py`. Lessons entry: "A drain the test performs by hand is not the product's drain".
