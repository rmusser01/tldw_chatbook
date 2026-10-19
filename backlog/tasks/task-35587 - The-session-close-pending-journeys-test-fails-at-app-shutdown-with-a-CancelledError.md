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
dependencies:
- TASK-34982
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
2026-10-19 UTC, first look (not a cause yet). The traceback has no frame below ConsoleTranscript's own `await gather(...)` in _message_loop_exit, which runs in _process_messages' finally after the loop's `except CancelledError: pass`. On the venv's CPython 3.12.11 (scratch script, three cases): gather over a child task cancelled before it ran keeps the child's frame; cancelling the awaiting task keeps the child's frames; only a child task whose cancellation an earlier awaiter already consumed gives a frameless CancelledError, because Future._make_cancelled_error hands out the saved exception once and builds a fresh one after that. So a direct child of the transcript most likely had a task that was cancelled and already awaited before shutdown.

TASK-34982 describes such a path: the transcript poll paints inside its timer's task, and stopping the timer while a paint awaits a row removal cancels the removed rows' tasks. Textual's Timer._stop_all then awaits the timer task and swallows its CancelledError, and the removal's gather has already consumed each row's. If those rows are still the transcript's children, a later prune's gather gets a fresh, frameless CancelledError, as in the CI run. That this is what happened here is inferred, not shown. Next step: TASK-34982's deterministic timer-stop test, then whether its fix covers this shutdown.

Locally on dev 69771c1811 the test passed 5 of 5 sequential runs (75-107 s each).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
