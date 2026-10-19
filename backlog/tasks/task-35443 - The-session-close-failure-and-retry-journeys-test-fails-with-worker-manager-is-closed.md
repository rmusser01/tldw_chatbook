---
id: TASK-35443
title: The session-close failure-and-retry journeys test fails with "worker manager is closed"
status: Done
created_date: 2026-10-18 10:10
updated_date: '2026-10-19 04:10'
assignee:
- '@claude'
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
- [x] #1 The cause of the "worker manager is closed" failure is identified, in the test or the product
- [x] #2 The test passes 20 consecutive pull-request runs in CI
- [x] #3 If the product was at fault, a test that fails deterministically without the fix pins it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read both failing CI logs to the first exception, not the wrapper's summary line.
2. Check whether dev already changed the failing code, and whether the failing heads contain that change.
3. Count the pull-request attempts since that change in which the test ran.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

2026-10-18, after #3152 split this file's journeys test. Data: every completed attempt of the pull_request runs of derived-artifacts.yml created from #3152's merge (2026-10-18 08:50Z) to 21:36Z: 36 runs, 35 completed attempts, every head containing the merge; 25 of them finished all ten UI shards, on 11 branches. The failure-and-retry test failed in none of them. The other half of the split, test_session_close_pending_journeys, failed once with a different teardown error: 39151121023/1 (chore/task-33132-libcollections-migration, a DB-module refactor) raised asyncio CancelledError from Textual's Widget._message_loop_exit (its gather over child tasks, at ConsoleTranscript) while _verify_background_pending_close_releases_round_without_an_active_turn's app shut down in run_test. Both failures happen at app shutdown in this file; that they share a cause is not shown.

2026-10-19 UTC, cause (AC #1, in the test). "worker manager is closed" is not a Textual error: it is the text the test's own stub raised. _verify_close_flow_that_cannot_start_tells_the_user replaced the screen's run_worker with a function that refused every call. While it was in place, a real deferred callback (chat_screen.py, the workspace tray recompose) submitted its console-workspace-context-legacy-aliases worker; the stub refused that too, the RuntimeError escaped into Textual's message loop and stopped the app, and the later _await_tabs timed out with the tab ids already equal ('tab strip never settled: {x} != {x}'). Both logs (39050477414/1, 39049115705/1) show that same stub frame and _kwargs. The product was not at fault, so AC #3 needs no product pin.

The fix was already on dev before this task was filed: 623df0136e ("isolate close worker failure from workspace refresh", merged with #3146 on 2026-10-17 18:33Z) makes the stub refuse only the exact console-close-session:<id> group and delegate every other call, and forces the recompose callback inside the patch so the delegation is exercised. It added the lesson "Worker failure injection must leave unrelated work running". Neither failing head (514c9000c6, 47d54057d3) contains it.

AC #2: every completed attempt of derived-artifacts.yml pull_request runs created from 2026-10-17 18:00Z to 2026-10-19 01:00Z (65 runs, 78 attempts), kept where the head contains 623df0136e and both jobs that run this file (UI Fast Lane (8), Console P0 regression gate) finished: 55 attempts (52 distinct runs) on 19 branches, from 2026-10-18 01:56Z to 2026-10-19 00:56Z. The failure-and-retry test failed in none of them. The one failure of this file in that window is the pending-journeys CancelledError above, now split into its own task, which cites this one.

Files: this task file, backlog/docs/lessons-testing-evidence.md (no code change).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Test fault, already fixed on dev by 623df0136e (#3146): the test's run_worker stub refused an unrelated deferred worker and its own RuntimeError stopped the app. 55 covered PR attempts (52 distinct runs) since, no failure. The separate pending-journeys CancelledError is split into its own task.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
