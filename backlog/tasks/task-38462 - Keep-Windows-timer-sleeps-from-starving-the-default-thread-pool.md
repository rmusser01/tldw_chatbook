---
id: TASK-38462
title: Keep Windows timer sleeps from starving the default thread pool
status: To Do
created_date: 2026-10-31 11:30
labels:
- windows
- performance
priority: medium
references:
- backlog/tasks/task-38206 - GGUF-source-mode-recompose-test-times-out-on-Windows-CI-intermittently.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On Windows, Textual runs every timer sleep, and every timer cancellation, as a blocking job on the event loop's default executor (textual/_win_sleep.py). The app's default executor has min(32, cores + 4) threads, so 8 on a 4-vCPU machine. TASK-38206's thread dumps from two windows-latest CI jobs show all 8 threads parked in `_win_sleep.wait_inner`, with 27–48 timer sleeps queued, including three 24-hour timers that hold 3 threads permanently. Thread workers (`run_worker(..., thread=True)`) and timer cancellations queue behind the sleeping timers, so a screen recompose, which stops many timers, or an inventory read can stall for tens of seconds. This is what made the GGUF source-mode tests time out on Windows CI. It also affects the shipped app on low-core Windows machines, not only tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On Windows, long-lived timers and thread workers no longer compete for the same bounded executor, so a recompose or thread worker does not wait behind sleeping timers (shown with the TASK-38206 8-thread proxy or a Windows run).
- [ ] #2 The three 24-hour timers no longer each hold a default-executor thread for their whole duration on Windows.
- [ ] #3 macOS and Linux timer and worker behavior is unchanged.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
