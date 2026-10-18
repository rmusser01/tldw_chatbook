---
id: TASK-35443
title: The session-close failure-and-retry journeys test fails with "worker manager is closed"
status: To Do
created_date: 2026-10-18 10:10
updated_date: 2026-10-18 22:09
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

2026-10-18, after #3152 split this file's journeys test. Data: every completed attempt of the pull_request runs of derived-artifacts.yml created from #3152's merge (2026-10-18 08:50Z) to 21:36Z: 36 runs, 35 completed attempts, every head containing the merge; 25 of them finished all ten UI shards, on 11 branches. The failure-and-retry test failed in none of them. The other half of the split, test_session_close_pending_journeys, failed once with a different teardown error: 39151121023/1 (chore/task-33132-libcollections-migration, a DB-module refactor) raised asyncio CancelledError from Textual's Widget._message_loop_exit (its gather over child tasks, at ConsoleTranscript) while _verify_background_pending_close_releases_round_without_an_active_turn's app shut down in run_test. Both failures happen at app shutdown in this file; that they share a cause is not shown.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
