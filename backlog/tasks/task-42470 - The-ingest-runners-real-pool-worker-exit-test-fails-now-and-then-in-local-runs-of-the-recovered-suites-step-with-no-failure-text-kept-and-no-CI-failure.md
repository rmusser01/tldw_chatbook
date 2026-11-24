---
id: TASK-42470
title: "The ingest runner's real-pool worker-exit test fails now and then in local runs of the recovered-suites step, with no failure text kept and no CI failure"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: '2026-11-23 23:30'
labels:
- follow-up
- library
- ingest
- tests
- flaky
priority: medium
references:
- Tests/Library/test_library_ingest_runner.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42426's and TASK-42412's implementers. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths start at the repository root.

**Why.** `Tests/Library/test_library_ingest_runner.py::test_real_pool_worker_exit_is_reported_for_owning_generation` (line 1796) runs in the required "PR lane (recovered suites)" job's "Run the recovered suites" step (`.github/workflows/derived-artifacts.yml`, where the file's entry is line 1856; flags `-p no:xdist --storage-lease-guard --timeout=180`). The test kills a real parse-pool worker (`pool.apply_async(os._exit, (17,))`, line 1841) and expects the job to fail as not permanent, with the pool and its generation released (lines 1843-1852). It failed in 2 of about 5 local runs of the whole step with CI's flags. TASK-42426 recorded it failing at base with the same flags (2004 passed, 2 failed, in 24:41 at load 34-46), and TASK-42412's second step run failed it too (load 24-46). It passed in the other runs. Neither implementer kept the failure text: TASK-42412's step script kept only the last 15 lines.

**CI.** In a scan of every attempt of the Derived Artifacts runs created from 2026-11-19 06:00Z to 2026-11-23 06:00Z (321 runs; the first page of run 45080343651's job list returned Server Error in the scan and, re-read, holds one more passing attempt), the recovered-suites job completed at least 168 times (129 passed, 39 failed). This node failed in none of the 38 failed jobs whose logs were read, and it passed in every passing one. The failures there were the managed received acceptance test (TASK-42412), the stock skill-trust dispose cases (TASK-42426) and two other nodes. Alone, with CI's flags, it passed 5 of 5 at load about 51. So it is a local red under a loaded whole-step run, not yet a CI red; priority medium for that reason. Reproducing it may need the whole step, for example because other tests' pools or storage leases are still live when this worker exits; that is a guess.

2026-11-23 (round-22 final gate, stack top a999508ae7 on dev fe8c87e3e2): the first failure text, from a whole recovered-suites step run serially with CI's flags (full log kept): `Tests/Library/test_library_ingest_runner.py:1861: assert app._ingest_parse_pool_retiring is False` failed with True, after `ERROR app_ingest_queue:_handle_broken_ingest_parse_pool:3971 - Library ingest parse pool failed: Library ingest parse-pool worker exited unexpectedly (generation 1)` and, 10 s later, `ERROR app_ingest_queue:_shutdown_workers:4166 - Error terminating the Library ingest parse pool. TimeoutError: Library ingest parse pool shutdown timed out.` The pool's shutdown timed out, so the retiring flag was still set when the test's 500-poll wait ended. The same step passed (2006) on the same stack an hour earlier.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Whole-step runs with CI's flags (or runs under a forced delay or load) keep their full output, and any failure of this test is recorded in the notes with its failure text
- [x] #2 If it fails, its cause is named with evidence and fixed where it lives (the ingest runner's pool handling, the test, or another test in the step that leaves state behind); the test keeps its assertions and then passes in repeated whole-step or loaded runs with CI's flags (at least 10, or a soak recorded in the notes)
- [x] #3 If at least 10 whole-step runs with CI's flags show no failure, the task closes with those runs and their loads recorded
<!-- AC:END -->

## Implementation Plan

1. Probe every `multiprocessing.Pool` termination (stage timings, thread stacks and worker states when one passes 3 s) through a throwaway pytest plugin.
2. Run the whole "Run the recovered suites" step with CI's flags, keeping the full output, at least 10 times (some under extra load) and record each run's load and outcome.
3. If it fails, name the cause from the probe's dump and fix it where it lives; otherwise close per AC#3.

## Implementation Notes

**Round 24 (2026-11-23, branch fu/r24-library, base dev b31e3e9241).**

**Failure, with its text (AC#1).** Three whole "Run the recovered suites" runs with CI's flags (`-p no:xdist --storage-lease-guard --timeout=180`), run side by side, full output kept, with a throwaway plugin that timed every `Pool._terminate_pool` and dumped all thread stacks and the workers' state when one passed 3 s. Load 23 at the start, about 80 when the test ran. One failed (2 failed, 2004 passed, 48:44), with the round-22 text: `:1861 assert app._ingest_parse_pool_retiring is False`, after "parse-pool worker exited unexpectedly (generation 1)" and, 10 s later, "Library ingest parse pool shutdown timed out". The other two passed (2006 each); in them `_terminate_pool` took 0.00-0.01 s and the worker exited -15. The run's other failure, `Tests/Chat/test_console_goal_dispatch.py::test_a_goal_iteration_from_a_cold_console_builds_the_skill_trust_it_captures` (`UNKNOWN_EFFECT`), is unrelated and listed for the coordinator.

**Cause (AC#2).**
- The dump: teardown was blocked in `p.join()` (`multiprocessing/pool.py:732`), after `p.terminate()`, on the replacement worker the pool had forked about 1 s before teardown, to replace the worker the test killed. The pool's handler threads had all exited.
- Five minutes later the worker was still alive: no pending or blocked signals, 0.07 s of CPU, and `sample` put it in `SemLock.acquire` (the inqueue lock that `Pool._help_stuff_finish` takes and keeps). A fresh SIGTERM killed it at once, so its SIGTERM disposition was the default by then.
- Why the pool's SIGTERM was lost: on macOS, `multiprocessing`'s spawn forks and then execs without resetting signal handlers, and `util.spawnv_passfds` returns right after the fork, so `Pool` lists the child before it has exec'd. Every pytest process here has a Python SIGTERM handler (`Tests/real_profile_guard.remove_sandbox_at_exit`, TASK-42145); the app installs one at start (`install_termination_handlers`). A SIGTERM in that window runs the inherited C handler, which only sets a flag, and the child then execs with nothing pending. `_help_stuff_finish` had already drained the pool's stop sentinels, so the child waits on the queue lock for ever and `join` never returns.
- A stdlib script (scratch, not committed) that sends SIGTERM right after `spawnv_passfds`: 0 of 100 children survived with the default disposition in the parent, 77 of 100 with a Python handler (load ~79).
- Run alone the test passed 24 of 24 at `-n 6` (load ~35); the window is only hit when the pool forks the replacement just before teardown and the fork is slow (a large process under load).
- The same race exists in the app (it has a SIGTERM handler), where it would leave the pool retiring, so imports would not resume, and orphan a worker until restart.

**Fix (`app_ingest_queue.py`, `_shutdown_ingest_workers_off_thread`).** If `pool.terminate()` / `join()` has not finished after `_INGEST_PARSE_POOL_KILL_AFTER_SECONDS` (2 s), every pool worker still alive is killed (SIGKILL cannot be caught, before or after exec), and the wait goes on for the rest of the unchanged 10 s bound. The test is unchanged.

**Red/green.** `test_pool_teardown_kills_a_worker_that_outlives_sigterm` (`Tests/Library/test_library_ingest_runner.py`): a real one-worker pool whose worker ignores SIGTERM and is busy, so it cannot take the stop sentinel either, which is the stuck worker's state. Base: red, "teardown still running after 8.0 s". Fixed: 3 of 3 green, then 20 of 20 at `-n 4` with the real-pool exit test (40 passed, load 76-80). `test_worker_shutdown_timeout_reports_failure_without_late_completion` still passes (the kill wait is capped by the patched timeout).

**Whole-step runs with the fix (AC#2).** Ten more whole "Run the recovered suites" runs with CI's flags and the probe, five at a time, full output kept:
- The first five ran at load 44-80; each passed 2007 (the step's 2006 plus the new test).
- The second five ran at load 31-117. Four passed 2007. One had an unrelated failure, `Tests/Chat/test_console_trace_maintenance_parking.py::test_a_signal_woken_real_pass_opens_only_its_write_transaction` ("the signal-woken pass opened [False, True]", a load flake TASK-41909 already lists).
- The real-pool worker-exit test passed in all ten.
- In one of them the race happened again. `_terminate_pool` started with an empty worker list, because the replacement was being added. It then took 2.01 s, and the worker ended -9: it had kept running after the SIGTERM and was killed. In the other nine the pool's SIGTERM worked (0.00-0.02 s, -15).
- In every run the probe logged the parent's SIGTERM handler as `remove_sandbox_at_exit.<locals>.remove_then_die`.

**AC#3.** The test did fail before the fix, so AC#1 and AC#2 decided this task. The ten whole-step runs after the fix (above, loads recorded) show no failure of it.

**Lane time.** Both tests are in the recovered-suites step's whole-file entry. The new test: setup 1.44 s, call 0.42 s (CI flags, load ~59).

**Size.** `app_ingest_queue.py` 5057 -> 5072 lines; `test_module_size_ratchet.py` row raised by exactly that, with a dated note.

**Lesson.** Added to `backlog/docs/lessons-testing-evidence.md`.

**Files.** `tldw_chatbook/app_ingest_queue.py`, `Tests/Library/test_library_ingest_runner.py`, `Tests/Architecture/test_module_size_ratchet.py`, `backlog/docs/lessons-testing-evidence.md`.

**Checks (final tree, shared by TASK-41723, 42470 and 42473).** `Tests/CI` at `-n 4`: 1823 passed, 2 skipped. `Tests/Architecture` serially: 1491 passed, 5 skipped, 1 failed. The failure was `test_persistent_diagnostic_inventory.py::test_task_15103_review_ledger_canonical_schema_and_arithmetic`, at its 300 s timeout while the run shared the host at load 33-67. Alone it passed in 194 s (load ~45-50). `./scripts/preflight.sh` passed; `ruff check tldw_chatbook --select F` passed; `Tests/lint/ruff_gate.py` passed (RATCHET OK).

**Review fixes (review of the round-24 branch, Minor 6).** The SIGKILL escalation now logs one warning naming how many workers it killed and their pids ("Killed 1 Library ingest parse worker(s) still running after the pool's terminate (pids=[…])."), only when it killed any. `test_pool_teardown_kills_a_worker_that_outlives_sigterm` asserts it. That test's worker now ignores SIGTERM, writes a marker and sleeps inside one task, and teardown waits for the marker: with two separate tasks, an idle worker could take the stop sentinel first and the kill never happened (1 of 3 runs). Red: base, "teardown still running after 8.0 s"; the first round-24 commit (kill without the warning), `[] == ['Killed 1 …']`. Green 5 of 5. The production diagnostic inventory reported one added call in `app_ingest_queue.py` (`--statements` shows the warning above: a count and pids, no user content, path or URL), and `--write` regenerated `Docs/security/production-diagnostic-inventory.json`. `app_ingest_queue.py` 5072 -> 5081 lines; the ratchet is raised by exactly that.

**Runs after the review fixes (shared by TASK-41723, 42470 and 42473; final tree).**
- **Soak of the touched files, 20× at `-n 4`:**
  - `test_library_ingest_readiness.py`: 580 passed (load 82-106).
  - `test_library_ingest_runner.py` with `--storage-lease-guard`: 3140 passed, 20 skipped (load 106-119).
  - The two `test_library_shell.py` nodes (ingest submit, Media retry): 40 passed (load 96-118).
  - `test_library_rail_switch_keeps_reader_and_note.py`: 550 passed, 10 failed (load 55-108). The new tests passed in every run. All 10 failures were `test_each_library_switch_lands_on_its_first_row_never_a_grip` cases:
    - 9 Collections cases ending with nothing focused: the capture loader outlasting the 2 s entry window above load 50, which TASK-42422 and TASK-41504 recorded;
    - 1 `media-click-empty-notes` ending on the rail row.
  - That test then passed an A/B at `-n 4`, 10 runs a side: base `library_screen.py` 120/120 (load 42-67), this branch 120/120 (load 35-60).
- **Checks:**
  - `Tests/CI` at `-n 4`: 1823 passed, 2 skipped.
  - Size ratchets, the Library recompose ratchet, the timer inventory and the diagnostic inventory tests (serial): 388 passed, 1 skipped, 2 failed. Both failures were the git-history `test_task_15103_review_ledger_*` tests at their 300 s timeout (load 70-118). Alone they passed in 254 s and 170 s.
  - `./scripts/preflight.sh` passed. `ruff check tldw_chatbook --select F` passed. `Tests/lint/ruff_gate.py`: RATCHET OK.
- **Lane steps via `r19-step.sh`:**
  - "Run the recovered suites" (`-p no:xdist --storage-lease-guard --timeout=180`): 2007 passed (57 min, load 41-119).
  - "Run admission-sensitive suites 2" (`-p no:xdist --timeout=300`): 1513 passed, 1 failed (50 min, load 47-118). The failure was `test_console_session_admitted_roots.py::test_the_three_readers_return_the_same_string_roots`, a private-profile child `TimeoutError`; Console code this branch does not touch. Alone it passed 2 of 2.
