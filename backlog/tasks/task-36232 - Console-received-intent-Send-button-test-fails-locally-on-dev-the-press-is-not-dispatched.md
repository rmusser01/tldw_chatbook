---
id: TASK-36232
title: >-
  Console received-intent Send-button test fails locally on dev: the press is
  not dispatched
status: To Do
assignee: []
created_date: '2026-10-22 23:09'
labels:
  - console
  - tests
  - ci
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while re-verifying PR #3187 (TASK-33620.15.6) on 2026-10-22. Tests/UI/test_console_received_intent_feedback.py::test_actual_send_receives_intent_before_original_configuration_read[send-button] fails on a developer machine under the plain pytest command, 4 of 4 single-node runs on a detached origin/dev worktree at 04c6c68dc8 and 4 of 4 on the PR branch, at line 92 ('stock Send did not reach original configuration Workspace SQL'). The [enter] variant passed those same 4 single-node runs on each arm, but in the local admission-sensitive Console P0 step (238 cases) both variants failed on both arms, identical by id. The app log shows ChatScreen.on_button_pressed for console-send-message followed by console_send_stage status=not_dispatched, so the press never reaches send preparation. The same node passes in hosted CI (Console P0 gate). This is the local-verification gap class of TASK-35781: a required Console P0 node that confirms nothing locally.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The [send-button] and [enter] variants pass under the plain pytest command on a developer machine, alone and inside the admission-sensitive Console P0 step
- [ ] #2 The cause of status=not_dispatched on the local Send-button press is recorded with evidence
<!-- AC:END -->
