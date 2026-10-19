---
id: TASK-35587
title: The session-close pending journeys test fails at app shutdown with a CancelledError
status: To Do
created_date: 2026-10-19 03:53
labels:
- tests
- flaky
- console
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_session_tab_close.py::test_session_close_pending_journeys runs in a required lane (UI Fast Lane (8) and the Console P0 gate). In run 39151121023 (attempt 1, 2026-10-18, chore/task-33132-libcollections-migration, a DB-module refactor) it failed with asyncio CancelledError while _verify_background_pending_close_releases_round_without_an_active_turn's app shut down in run_test: Textual's Widget._message_loop_exit, gathering the child tasks of ConsoleTranscript (#console-native-transcript) during the screen prune, raised it. Textual's own modules cancel only timer, worker and Pilot wait tasks, never a widget's message-loop task, so a child's task, or the transcript's own, was most likely cancelled from outside Textual. A failure on unchanged code counts against a pull request in the merge queue, and two in a row evict it. Split from TASK-35443, whose 'worker manager is closed' failure had a different cause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task that is cancelled at shutdown, and what cancels it, is identified, in the test or the product
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
