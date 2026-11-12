---
id: TASK-39143
title: "test_regenerate_cancel_records_stopped_replacement allows only 1 s of wall clock for the provider to start and for the stopped regenerate to finish"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- tests
- flaky
- console
priority: low
references:
- Tests/Chat/test_trajectory_capture.py
dependencies:
- TASK-38111
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. Line numbers are dev's.

- `Tests/Chat/test_trajectory_capture.py::test_regenerate_cancel_records_stopped_replacement` waits for the provider with `asyncio.wait_for(gateway.started.wait(), timeout=1)` (`:654`) and for the task with `timeout=1` (`:657`).
- In a review run of TASK-38111 (round-11 branch fu/r11-recovery; reversed order, heavy load), the first wait raised `CancelledError` then `TimeoutError`. It was the one failure beyond the 38 known stale nodes.
- The file is runnable only since TASK-38111, so this flake will reach any lane that runs it.
- The 10 ms bound in `test_console_chat_controller.py`, also named in this follow-up, is TASK-39046's and is not repeated here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Neither wait is bounded by 1 s of wall clock; the suite's `--timeout` remains the backstop for a real hang
- [ ] #2 Under a temporary 1.5 s delay before the gateway sets `started`, the old test fails at the first wait and the new one passes
- [ ] #3 The whole file passes 10 times at `-n 4`, with the load recorded
<!-- AC:END -->
