---
id: TASK-36233
title: >-
  Console P0 changed-source replay test: a timer tick can run the replay FULL in
  the held timer task
status: To Do
assignee: []
created_date: '2026-10-22 23:32'
labels:
  - console
  - tests
  - flaky
  - ci
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while re-verifying PR #3187 (TASK-33620.15.6) on 2026-10-22, separate from the two modes TASK-35713 tracks. Tests/UI/test_console_preparing_current_sources.py::test_preparing_changed_source_after_original_tab_await_replays_full asserts that every custom core-state call after the lock release comes from a task other than the held timer task (assert calls and all(task is not held['task'] for task in calls)). The transcript poll runs FULL inside the timer's own task every 0.2 s, and once the custom callback is installed that FULL is no longer narrowed, so a tick that starts after the re-armed worker FULL completes and before the test's 20 ms predicate poll observes it calls the custom callback from the timer task. Local evidence under background QoS (taskpolicy -c background, 6 concurrent single-node runs): on a detached origin/dev worktree at 04c6c68dc8 the current test failed this way in 1 of 18 cases; the PR branch failed this way in 3 of 18 (not a significant difference); the pre-PR #3202 copy of the test failed this way on the branch in 2 of 9 and on dev in 0 of 9. A sys.monitoring trace of the natural variant (12 runs per arm) showed the same FULL and Character-check sequences on both arms. Under the same QoS the natural variant also failed once on the branch (0 of 12 on dev) at its 5 s setup budget ('No original natural Preparing display completed before the tab hold'). Without QoS clamping both arms passed 16 of 16.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test distinguishes the held invocation from later FULL passes that legitimately run in the timer task, without weakening the replay, ownership and custody assertions
- [ ] #2 The test passes 25 consecutive single-node runs on dev under background-QoS load (or an equivalent slow-runner reproduction)
<!-- AC:END -->
