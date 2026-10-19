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
Tests/UI/test_console_session_tab_close.py::test_session_close_pending_journeys runs in a required lane (UI Fast Lane (8) and the Console P0 gate). In run 39151121023 (attempt 1, 2026-10-18, chore/task-33132-libcollections-migration, a DB-module refactor) it failed with asyncio CancelledError while _verify_background_pending_close_releases_round_without_an_active_turn's app shut down in run_test: Textual's Widget._message_loop_exit, gathering the child tasks of ConsoleTranscript (#console-native-transcript) during the screen prune, raised it. No .cancel() in Textual's own modules targets a widget's message-loop task (they cancel timer, worker, Markdown-stream, command-search and Pilot-wait tasks, and a call_later handle), so a child's task, or the transcript's own, was most likely cancelled from outside Textual. A failure on unchanged code counts against a pull request in the merge queue, and two in a row evict it. Split from TASK-35443, whose 'worker manager is closed' failure had a different cause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task that is cancelled at shutdown, and what cancels it, is identified, in the test or the product
- [ ] #2 The test passes 20 consecutive pull-request runs in CI
- [ ] #3 If the product was at fault, a test that fails deterministically without the fix pins it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-19 UTC, first look (not a cause yet). Python 3.12's gather re-raises a cancelled child's original CancelledError (Future._make_cancelled_error returns the saved _cancelled_exc), so the traceback keeps the frames where it was first raised. This one has no frame below ConsoleTranscript's own `await gather(...)` in _message_loop_exit. So either the transcript's own message-loop task was cancelled while it waited for its children, or a child's task was cancelled before it ran any frame. A CancelledError raised inside a child's Compose, Mount or Unmount would also end that child's task cancelled (Textual's _pre_process catches only Exception, and _process_messages guards only the message loop), but it would leave the child's frames in the traceback. Locally on dev 69771c1811 the test passed 5 of 5 sequential runs (75-107 s each).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
