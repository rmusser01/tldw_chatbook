---
id: TASK-17660
title: 'Settings: paste-collapse toggle persistence tests red on dev'
status: To Do
assignee: []
created_date: '2026-08-17'
labels:
  - settings
  - test-health
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two parameterizations of `Tests/UI/test_destination_shells.py::test_settings_console_paste_collapse_toggle_reflects_and_persists_config` (`[True-True-False]` and `[false-False-True]`) fail on clean origin/dev — verified 2026-08-17 in a detached baseline worktree at `8dc8c2a2c` with identical failures on the task-17653/17659 branch, which touches neither the Settings card's paste controls nor their persistence. Found during task-17653's footer-consumer sweep.

Needs a bisect against recent Settings/Console-Behavior merges (the status-row toggle, the selection-feedback arc, or another recent landing may have shifted the card's control order or the persistence seam the test drives).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The failing parameterizations are green on dev, either by fixing the regression they caught or by updating the test to the intended contract (decided by reproducing the toggle flow live first)
- [ ] #2 The task records which merge introduced the red
<!-- AC:END -->
