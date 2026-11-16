---
id: TASK-41135
title: "Two test hangs no timeout ends: the 239-file non-UI set at -n 4 stalls with all workers idle, and test_original_stop_hook_holds_same_ordinary_tree runs over an hour"
status: Done
created_date: 2026-11-15 02:20
updated_date: 2026-11-16 21:37
assignee:
- '@claude'
labels:
- follow-up
- tests
- console
- flaky
priority: medium
references:
- Tests/Chat/test_console_ordinary_tree.py
- pyproject.toml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** A hang that `--timeout` does not end blocks a whole worker's queue (about 90 tests) and forces runs to be interrupted, so covering runs lose results without saying so.

**Evidence.**
- TASK-40300's notes (endpoint branch): 239 non-UI files at `-n 4` "hung at 91% (all four workers idle for 18 min, a pre-existing hang)", the same at 95% on the round-1 tree.
- TASK-40321's notes: `Tests/Chat/test_console_ordinary_tree.py::test_original_stop_hook_holds_same_ordinary_tree` "made no progress for over an hour while the others drained; neither `--timeout=300` method ended it". Both arms were interrupted with SIGINT.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The stop-hook node's hang is reproduced and its cause recorded (a stack dump of the stuck worker)
- [x] #2 The node completes or fails within its timeout, and the cause is fixed in the product or the test
- [x] #3 The 239-file set runs to completion at -n 4, or the node that stalls it is identified and handled the same way
<!-- AC:END -->

## Implementation Plan

1. Reproduce the stop-hook node's hang serially through run.sh and take stack dumps of the stuck process (`sample`, then SIGABRT under `PYTHONFAULTHANDLER=1`).
2. Find why `--timeout` never fires on it, and fix that where every test routes through (the root conftest), with a pin that fails without it.
3. Re-run the node, its file at `-n 4`, and a reconstruction of the 239-file non-UI set at `-n 4`, and record what still stalls.

## Implementation Notes

- **AC#1, reproduced (2026-11-15, run.sh, `-p no:xdist --timeout=120`, load ~26).** `[False-False]` and `[False-True]` pass; `[True-False]` (agent runtime on) reports FAILED about 10 s into its call and the process then sat at 0% CPU for over 10 minutes. `sample` put the main thread in `kevent`; a SIGABRT faulthandler dump showed it inside pytest-asyncio's fixture finalizer (`runners.py:118` under `plugin.py:335`, `ordinary_host`'s `await asyncio.gather(*h.tasks)`), with one executor thread in `Agents/legacy_hook_ownership.py:40 wait_closed` (`Condition.wait_for` with no timeout, reached from `hooks_v2/engine.py`'s `_drain` via `asyncio.to_thread`) and the storage-admission thread idle.
- **Cause of "no timeout ends it".** pytest-timeout 2.4 cancels its timer in `pytest_exception_interact`, which pytest calls on every setup or call failure, not only under `--pdb`. The call failed, so no timer was armed for teardown under either method; TASK-41028's 4 h stall at 74/75 is the same shape. With a one-shot re-arm the teardown timed out once, freeing the gather, and the loop close then hung in `asyncio.runners._cancel_all_tasks` (second dump), because a task there survives cancellation.
- **AC#2, fixed in the test infrastructure.** `pytest_exception_interact` in `Tests/private_profile.py` re-arms the item's timer after a setup or call failure (not under `--pdb`, not for `func_only`), and for SIGALRM makes it repeat every `timeout` seconds until the protocol's own cancel; `Tests/conftest.py` imports it. It lives there, beside the module's other pytest-timeout coordination, because `perf-guard.yml` must trigger on every file its steps load and a new helper module would need a trigger-path edit (`test_perf_guard_runs_when_a_file_its_steps_load_changes`). Pin: `Tests/CI/test_pytest_timeout_rearm.py` (Tests/CI is a whole-directory target of the fast lane) runs a pytest child whose failing test has two fixtures that sleep 60 s in teardown, under `--timeout=2`, and requires a Timeout in under 40 s. Red with the hook as a no-op: no "Timeout" in the output (only `1 failed`); red with a one-shot re-arm: 70.2 s elapsed (`1 failed, 1 error`: only the first finalizer was freed). Green: 7.8-12.7 s of call at load 45-100. After the fix the node reports within its timeout: the four params serially at `--timeout=60` gave `2 failed, 2 passed, 2 errors in 284.53s` (the errors are the teardown timeouts). What the fix does not end: the serial process still cannot exit after the summary, because the `wait_closed` executor thread is non-daemon (`Py_FinalizeEx` -> `wait_for_thread_shutdown`). That product custody, and why `agent_runtime=True` never runs the Stop hook (the turn ends `FAILED` or `until` times out at 10 s), are TASK-41028's AC#1-#3, which owns this file's 33 red nodes.
- **AC#3.** TASK-40300's notes do not list its 239 files. A reconstruction by its rule (non-UI test files naming that round's changed helpers; 255 files, minus the two `Tests/ProductionApp` files, which need their own invocation) ran to completion at `-n 4` and the process exited: `485 failed, 11251 passed, 6 skipped, 7 errors in 7676.15s` (load 70-110). The stalling node TASK-41028 identified is handled the same way: `Tests/Chat/test_console_ordinary_tree.py` at `-n 4 --timeout=120` now finishes and exits (`33 failed, 42 passed, 2 errors in 696.90s`, rc 1, 716 s wall), where TASK-41028 saw it stop at 74/75 until a watchdog killed it.
- Lesson added to `backlog/docs/lessons-testing-evidence.md`.
- Files: `Tests/private_profile.py`, `Tests/CI/test_pytest_timeout_rearm.py` (new), `Tests/conftest.py`, `backlog/docs/lessons-testing-evidence.md`.
