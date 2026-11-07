---
id: TASK-39140
title: "test_executor_failure_uses_stable_job_terminal pins the STT failure's error_detail without the failure_stage and exception_category keys that df22692d70 added"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-07 05:51'
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
- [x] #1 Both failed-state cases pass and assert both keys with the values the fake failure produces
- [x] #2 Under a temporary edit that drops either key from `error_detail`, the test fails
- [x] #3 The cancelled case is unchanged
<!-- AC:END -->

## Implementation Plan

1. Run the two failed-state cases on base 8b762b52fa and read the diff (the two keys df22692d70 added).
2. Add `failure_stage` and `exception_category` to the failed-state expectation, with the values the test's fake `ExecutorFailure` produces (it carries no `STTFailureDiagnostic`, so production writes `"unknown"` for both). Leave the cancelled branch as it is.
3. Negative control: drop each key from `error_detail` in `app_ingest_queue.py` in turn, show the test red, revert.
4. Run the file at `-n 4`.

## Implementation Notes

**Cause: test drift.** df22692d70 ("isolate served tabs and preserve STT retry diagnostics", 2026-10-27) added `failure_stage` and `exception_category` to the STT failure's `error_detail` (`app_ingest_queue.py`). The test's fake `ExecutorFailure` carries no `STTFailureDiagnostic`, so production writes `"unknown"` for both, and the exact-dict pin still listed only the four older keys.

**Change.** `Tests/Library/test_library_ingest_runner.py::test_executor_failure_uses_stable_job_terminal` now expects both keys with `"unknown"` in its failed-state cases. The cancelled case is untouched (it never checks `error_detail`).

**Evidence** (run.sh, `-n 4`, load 35-39):
- Base 8b762b52fa, the node: 2 failed (`[artifact_incompatible-failed]`, `[engine_crashed-failed]`), 1 passed. Head: 3 passed.
- AC#2: a temporary edit that dropped `"failure_stage"` from `error_detail` turned both failed-state cases red ("Right contains 1 more item"); the same for `"exception_category"`. Source restored, no diff.
- Whole file on head: 156 passed, 1 skipped (TASK-38111 recorded 154 passed / 2 failed / 1 skipped).
