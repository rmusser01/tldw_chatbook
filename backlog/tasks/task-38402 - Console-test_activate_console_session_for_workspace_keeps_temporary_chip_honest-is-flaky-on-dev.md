---
id: TASK-38402
title: >-
  Console:
  test_activate_console_session_for_workspace_keeps_temporary_chip_honest is
  flaky on dev
status: To Do
assignee: []
created_date: '2026-10-31 02:11'
labels:
  - console
  - tests
  - flaky
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_composer_menu.py::test_activate_console_session_for_workspace_keeps_temporary_chip_honest fails intermittently under the plain command at its session-switch assertions (store.active_session_id is not the workspace's session after _activate_console_session_for_workspace and one pilot.pause()), before it reaches the temporary-chip assertions. Measured 2026-10-30, each run in isolation: 2 of 6 failed on dev 7536ac3b87 and 2 of 4 on the TASK-37840 branch, at lines 811 and 819. The file is not in the UI PR gate, so the flake has gone unnoticed; it also makes a full-file head-vs-dev comparison of that file noisy. Found while gating TASK-37840.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes 10 of 10 isolated runs under the plain command, by waiting for the activation's outcome rather than a single pause, or the race it exposes in the activation path is fixed
- [ ] #2 The test still checks the temporary chip after each of its three activations, in both directions
<!-- AC:END -->
