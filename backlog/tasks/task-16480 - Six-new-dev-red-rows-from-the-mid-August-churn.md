---
id: TASK-16480
title: Six new dev-red rows from the mid-August churn
status: Done
assignee:
  - '@Robert'
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

## Implementation Plan (the how)

1. Verified all six still red on dev tip 5b4820931 (2026-08-16)
2. Cluster A (settings hub x3): reproduce, attribute (the route-echo pair per
   the task note: look for a repopulation site missing the prevent()/nav-echo
   convention), fix at the convention seam
3. Cluster B (workbench contract left-rail): reproduce, attribute, fix
4. Cluster C (library capability matrix x2): reproduce, attribute, fix
5. Each attribution via parent-check/bisect before touching expectations
6. Whole modules green + lint; closeout; PR to dev

ADR required: no
ADR path: N/A
Reason: test-health repairs of existing behavior


## Acceptance Criteria

- [x] Each of the six is attributed to its causing commit
- [x] Genuine product breaks are fixed rather than absorbed into expectations
- [x] The three modules pass whole on dev

## Implementation Notes

All six verified still red on dev tip 5b4820931 before work; each attributed
to its causing commit first (2026-08-16, TASK-16480).

- Settings hub x3: two of the three landed red at their OWN introducing
  commits (a1405b154 picker, 449fab8ca route-echo -- both Aug 13 "preserve
  provider ..." fixes whose fixes did not work). Fixed forward: (i) the
  picker now gets its highlight at COMPOSE time (factored
  `_apply_provider_picker_highlight`) because the commit's
  `call_after_refresh(_refresh_provider_picker)` fires before the category
  body mounts and its QueryError is swallowed -- fresh pickers kept the
  first-selectable default; (ii) `_sync_provider_credential_widget` re-arms
  the credential suppress queue when the value changes (inside the
  task-15740 prevent block) so navigation-applied credentials stay
  suppressible across the widget-lifecycle echo -- prevent() silences only
  the live write, and the recompose echo fires un-prevented.
- Workbench contract (left rail): bisected to 6476d84f0 (compact rail
  focus), which deliberately closes the left rail at 120 columns -- its own
  resize-reflow tests encode that. The task-400 contract test's 120x40
  premise is stale under that responsive design; retargeted to 140x42 with
  the attribution documented in the test docstring. Contract unchanged.
- Library capability matrix x2: bisected (2-run-majority predicate after a
  flake fooled the first bisect) to d5a08b0fe (reconcile retained Library
  transitions). GENUINE product bug: the discard coroutine's list
  reconcile runs while the mutation interlock is still held, so the
  rebuilt canvas freezes operation_running=True into the browse toolbar --
  New/Sort/Select permanently disabled (keyboard AND mouse) after
  discarding a freshly created note. Fixed by re-syncing the canvas after
  releasing the interlock in the discard finally, mirroring
  `_execute_library_notes_tree_mutation`'s own finally.
- Whole modules on this branch: settings hub + workbench contract 420
  passed, library shell 594 passed, 0 failures. Ruff findings on touched
  files all pre-exist on dev.
