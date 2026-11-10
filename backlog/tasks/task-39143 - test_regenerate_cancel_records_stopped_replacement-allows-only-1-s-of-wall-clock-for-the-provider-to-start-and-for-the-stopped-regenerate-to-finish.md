---
id: TASK-39143
title: "test_regenerate_cancel_records_stopped_replacement allows only 1 s of wall clock for the provider to start and for the stopped regenerate to finish"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-07 06:33'
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
- [x] #1 Neither wait is bounded by 1 s of wall clock; the suite's `--timeout` remains the backstop for a real hang
- [x] #2 Under a temporary 1.5 s delay before the gateway sets `started`, the old test fails at the first wait and the new one passes
- [x] #3 The whole file passes 10 times at `-n 4`, with the load recorded
<!-- AC:END -->

## Implementation Plan

1. Replace both `asyncio.wait_for(..., timeout=1)` waits with waits that end on the product's own events: the provider-start wait ends when the gateway sets `started` or when the regenerate task finishes first (its own exception is re-raised, so a failure never becomes a silent hang); the second wait awaits the task. The suite's `--timeout` stays the backstop.
2. Negative control: a temporary 1.5 s sleep before the gateway sets `started`; the old test fails at the first wait, the new one passes. Revert.
3. Run the whole file 10 times at `-n 4`, with the load recorded.

## Implementation Notes

**Change.** Both 1 s wall-clock bounds in `test_regenerate_cancel_records_stopped_replacement` are gone.
- The provider-start wait is now `until_provider_starts(gateway.started, task)`, a new helper in the shared `Tests/console_provider_doubles.py`. It waits on the gateway's own `started` event or on the regenerate task ending first, with no timeout. If the task ends first, the helper re-raises the task's own error, or reports its result, instead of timing out with nothing to read.
- The second wait is a plain `await task`.
- The suite's `--timeout` remains the backstop for a real hang.
- TASK-39138 reuses the helper for the controller's two stop-path nodes.

**Evidence.**
- AC#2, with a temporary `await asyncio.sleep(1.5)` before `_CancelGateway` sets `started` (reverted): the old test failed at the first wait (`CancelledError` -> `TimeoutError` at the `asyncio.wait_for(gateway.started.wait(), timeout=1)` line), and the new test passed. Both ran serially through run.sh.
- AC#3: the whole file passed 10 runs of 10 at `-n 4`, 32 passed each, at load 46-71 with up to six agents sharing the machine (23-39 s per run).
