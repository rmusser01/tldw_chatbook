---
id: TASK-35743
title: >-
  Resend hook-admission pilot asserts a streamed paint after a fixed 0.65 s
  pause
status: To Do
assignee: []
created_date: '2026-10-20 05:34'
labels:
  - console
  - tests
  - flaky
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_turn_resend_ui.py::test_resend_poll_survives_initial_hook_admission_read runs in a required lane: the file is in scripts/ui_pr_gate_census.txt. After its resend starts streaming, the test waits a fixed pilot.pause(0.65) and then asserts that 'probe incremental reply' is visible. On a loaded host the reply reaches the store but is not painted yet.

Found under TASK-35660. On 2026-10-19 the file was run once on the rebased TASK-35660 branch at host load 43-52 (macOS, plain command). This test failed in its private-profile child with "assert 'probe incremental reply' in ...", while the visible text still showed the run as Running / Preparing... / Stop. The store assertion just before it passed. Run alone at load 47-58, it then passed 4 of 4 on the branch and 4 of 4 on dev 8136af6f6f, alternating. Before the rebase it was red on both trees in a whole-file run at load ~100. TASK-35660 changes only what happens before the press, so this wait is not something that branch changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test waits for the streamed reply to be painted with a bounded condition wait, not a fixed pause, and fails naming what was visible if it never is
- [ ] #2 With the poll's paint of the streamed reply disabled (scratch mutation), the test still fails
<!-- AC:END -->
