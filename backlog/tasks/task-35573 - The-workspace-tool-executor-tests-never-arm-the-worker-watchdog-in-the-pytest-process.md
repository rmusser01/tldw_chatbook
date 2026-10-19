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

Several tests in `Tests/Tools/test_workspace_tool_executor.py` call `run_workspace_worker` in the pytest process itself. That arms `worker_watchdog.arm_watchdog`, which is meant for a dedicated worker process. It sets a hard `RLIMIT_CPU` of twice the budget (60 s for the default 30 s) and a `signal.alarm` with the default action on the process that runs it. An xdist worker that has already used more than 60 s of CPU then dies at once ("node down: Not properly terminated"). It was seen twice in one run, and the loadfile scheduler then hung the whole run at 98% until it was killed. Run alone, the file passes (142 passed), which is why a short run does not show it. Any shared run that includes the file risks losing a worker and every test still queued on it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No test leaves an RLIMIT_CPU or a pending alarm on the pytest process: the tests run the worker in a child process, or stub the watchdog's arming, and the limits are restored after each test
- [ ] #2 A shared xdist run that includes the file, after other files have used more than 60 s of CPU in the same worker, finishes with no worker lost
- [ ] #3 A test fails if a test in the file returns with the process's RLIMIT_CPU lowered or an alarm still pending
<!-- AC:END -->
