---
id: TASK-35883
title: >-
  Two composer-state pilots are red on dev: the Console's own sync overwrites
  the state they set by hand
status: To Do
assignee: []
created_date: '2026-10-21 00:45'
labels:
  - console
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_send_disabled_state.py::test_active_run_block_shows_wait_reason and ::test_idle_stop_button_has_no_unreachable_tooltip fail locally on dev 2d071d6d71. Each calls composer.sync_action_state(run_active=True, send_blocked=True, ...) on the mounted Console, pauses 0.1 s and then asserts what that call rendered. Instead Send is enabled ('assert False is True' at line 200) and Stop is hidden ('none' == 'block' at line 266), which looks like the Console's own control-bar sync re-pushing the real, idle state during the pause. Neither test calls the shared provider-selection helper. No task tracks them, and the file is not in scripts/ui_pr_gate_census.txt, so whether CI runs it green is not known.

Found under TASK-35660 on 2026-10-20 (macOS, plain command, load 16-22). Run alone as a pair, three times on each tree: dev failed both tests every time; the TASK-35660 branch, which changes neither the tests nor the code they drive, failed both in 2 of 3 runs. In whole-file runs, dev failed only the wait-reason test and the branch failed both. In TASK-35660's previous review pass, on dev 0fef489ece, neither was red.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause is named: what overwrites the hand-set composer state between the call and the assertions, and which dev change made it reachable
- [ ] #2 Both tests pass on dev under the plain command in five consecutive runs alone and in a whole-file run, or each is rewritten to drive the state through the product path it means to pin
- [ ] #3 If the overwrite is a product defect (a sync that drops a live run's state), it is fixed or filed separately with the evidence
<!-- AC:END -->
