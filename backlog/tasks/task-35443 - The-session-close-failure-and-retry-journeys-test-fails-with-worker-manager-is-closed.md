---
id: TASK-35443
title: The session-close failure-and-retry journeys test fails with "worker manager is closed"
status: To Do
created_date: 2026-10-18 10:10
labels:
- tests
- flaky
- console
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_session_tab_close.py::test_session_close_failure_and_retry_journeys runs in a required lane (UI Fast Lane and the Console P0 gate). In the pull-request runs from 2026-10-15 16:27Z to 2026-10-18 06:33Z it never failed before PR #3137 reached dev, then failed twice after it with the same child failure, "RuntimeError: worker manager is closed": run 39050477414 (attempt 1, PR #3113, which changed no Console or UI file) and run 39049115705 (attempt 1, codex/console-permission-filesystem-waits). A failure on unchanged code counts against a pull request in the merge queue, and two in a row evict it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the "worker manager is closed" failure is identified, in the test or the product
- [ ] #2 The test passes 20 consecutive pull-request runs in CI
- [ ] #3 If the product was at fault, a test that fails deterministically without the fix pins it
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
