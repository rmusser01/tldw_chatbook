---
id: TASK-35573
title: The workspace tool executor tests never arm the worker watchdog in the pytest process
status: To Do
created_date: 2026-10-19 05:30
dependencies: []
labels:
- follow-up
- tests
- tools
priority: medium
references:
- Tests/Tools/test_workspace_tool_executor.py
- tldw_chatbook/Tools/worker_watchdog.py
- tldw_chatbook/Tools/workspace_tool_worker.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in Phase 9's post-rebase interaction check (2026-10-19), on dev 69771c1811; Phase 9 touches no Tools or Agents file.

Several tests in `Tests/Tools/test_workspace_tool_executor.py` call `run_workspace_worker` in the pytest process itself. That arms `worker_watchdog.arm_watchdog`, which is meant for a dedicated worker process. It sets `RLIMIT_CPU` to twice the budget, soft and hard alike, which an unprivileged process can never raise again, on the process that runs it. It also calls `signal.alarm(budget + 2)`, which in the pytest process replaces pytest-timeout's own SIGALRM timer (its signal method, the default here); the worker's `disarm_watchdog` then zeroes that timer, so the rest of that test runs with no timeout at all. Several of the file's hand-built requests use a 30 s budget (60 s of CPU); the production default is 300 s. An xdist worker that has already used more than 60 s of CPU then dies at once ("node down: Not properly terminated"). It was seen twice in one run, and the loadfile scheduler then hung the whole run at 98% until it was killed. Run alone, the file passes (142 passed), which is why a short run does not show it. Any shared run that includes the file risks losing a worker and every test still queued on it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No test lowers the pytest process's RLIMIT_CPU, or replaces or clears pytest-timeout's real-time timer: the worker runs in a child process, or the watchdog's arming is stubbed
- [ ] #2 A shared xdist run that includes the file, after other files have used more than 60 s of CPU in the same worker, finishes with no worker lost
- [ ] #3 A test fails if a test in the file returns with RLIMIT_CPU lowered, or with ITIMER_REAL no longer the timer pytest-timeout set
<!-- AC:END -->
