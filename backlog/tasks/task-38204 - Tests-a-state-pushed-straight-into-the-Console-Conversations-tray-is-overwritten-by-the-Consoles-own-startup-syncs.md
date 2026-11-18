---
id: TASK-38204
title: >-
  Tests: a state pushed straight into the Console Conversations tray is
  overwritten by the Console's own startup syncs
status: To Do
assignee: []
created_date: '2026-10-30 08:38'
labels:
  - tests
  - console
  - flaky
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tray tests that push a hand-built state straight into ConsoleWorkspaceContextTray (sync_state) and then pause once flake under host load: the Console's own deferred startup syncs land after the push and replace the state, so the tray re-renders the real rows. Measured 2026-10-30 with a probe of Tests/UI/test_console_workspace_tray_recompose_guard.py::test_a_structural_change_still_recomposes (push sections=(), one pause, expect no rows): 2 of 32 runs failed on dev ff815f8559 and 3 of 32 on the TASK-33622.6 branch, both with a new row instance and the tray state back at 1 section. The overwriting calls were ChatScreen._sync_console_workspace_context via _sync_native_console_chat_ui (worker, idle callback, and the recovered-images loader) and the workspace files-availability snapshot refresh. The same family made test_zero_result_workspace_search_stays_editable_and_initializes_without_echo flaky in the TASK-33622.6 review. A flaky pin hides real regressions and costs reruns.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tray tests that push a state directly wait until the Console's startup syncs have settled (or suppress them) before pushing, so the pushed state is the one rendered
- [ ] #2 test_a_structural_change_still_recomposes passes 32 of 32 consecutive runs under the plain command on a loaded host
- [ ] #3 The settle helper is shared, not copied per test file
<!-- AC:END -->
