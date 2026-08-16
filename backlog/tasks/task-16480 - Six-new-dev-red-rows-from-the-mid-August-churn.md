---
id: TASK-16480
title: Six new dev-red rows from the mid-August churn
status: To Do
assignee: []
labels:
  - test-health
priority: medium
---

## Description

Found 2026-08-15 while closing task-15741 (a 5-module verification run on
dev `2e115cf04`-era): six tests are red on untouched dev, none previously
inventoried -- fresh drift from the last two days' heavy merge traffic.

- `test_settings_configuration_hub.py::test_settings_provider_picker_initial_known_provider_enter_is_noop_for_drafts`
- `test_settings_configuration_hub.py::test_settings_provider_route_echoes_do_not_survive_widget_lifecycle[category_departure]`
- `test_settings_configuration_hub.py::test_settings_provider_route_echoes_do_not_survive_widget_lifecycle[recompose]`
- `test_console_workbench_contract.py::test_console_left_rail_keeps_session_and_moves_staged_context_out`
- `test_library_shell.py::test_library_note_keyboard_capability_matrix[create_discard-terminal_size0]`
- `test_library_shell.py::test_library_note_keyboard_capability_matrix[create_discard-terminal_size1]`

Per the standing 15512 discipline: attribute each to its causing commit
before adjusting any expectation; the route-echo pair touches the
task-15673/15740 echo family, so check whether a new repopulation site
missed the prevent()/nav-echo convention before calling the tests stale.
Another session may already be on some of these -- check `.worktrees/` and
open branches before starting (a task's status is not a lock).

Extra evidence for the capability-matrix pair (added 2026-08-16, burn-down
close-out): both still fail at dev `ee741cf10` (re-run this session). The
task-15774 review reproduced them standalone at its branch HEAD AND at a
fresh origin/dev-tip throwaway worktree with the identical assertion:
`#library-notes-new was not reachable with tab; focused=NavigationButton
(id='nav-lab', ...)` -- i.e. tab-order focus stranding on the nav-lab
button, not a notes-editor defect.

## Acceptance Criteria

- [ ] Each of the six is attributed to its causing commit
- [ ] Genuine product breaks are fixed rather than absorbed into expectations
- [ ] The three modules pass whole on dev
