---
id: TASK-39140
title: "test_executor_failure_uses_stable_job_terminal pins the STT failure's error_detail without the failure_stage and exception_category keys that df22692d70 added"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- tests
- library
priority: low
references:
- Tests/Library/test_library_ingest_runner.py
- tldw_chatbook/app_ingest_queue.py
dependencies:
- TASK-38111
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-38111's notes (round-11 branch fu/r11-recovery) list it. Line numbers are dev's.

- df22692d70 (2026-10-27, "isolate served tabs and preserve STT retry diagnostics") added `failure_stage` and `exception_category` to the STT failure's `error_detail` (`app_ingest_queue.py:3254-3255`).
- `Tests/Library/test_library_ingest_runner.py::test_executor_failure_uses_stable_job_terminal` still expects exactly the four older keys (`:3541`). Its `[artifact_incompatible-failed]` and `[engine_crashed-failed]` cases fail with "Left contains 2 more items: {'exception_category': 'unknown', 'failure_stage': 'unknown'}".
- `[cancelled-cancelled]` passes.
- The RecoveryRequired trap hid this until TASK-38111, so the new diagnostic keys are unpinned.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both failed-state cases pass and assert both keys with the values the fake failure produces
- [ ] #2 Under a temporary edit that drops either key from `error_detail`, the test fails
- [ ] #3 The cancelled case is unchanged
<!-- AC:END -->
