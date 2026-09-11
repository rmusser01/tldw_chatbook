---
id: TASK-32313
title: 'Fix stale controller and reconcile test fixtures (drift)'
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - tests
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seven tests were red on dev (verified on a pristine worktree): four resume
tests in `Tests/UI/test_console_workspace_controller.py` plus its
constructor-docstring test, and two in
`Tests/Workspaces/test_console_workspace_reconcile.py`. All fixture drift,
not product bugs: the resume flow gained a token-preparation step that reads
`screen._build_console_provider_selection` (missing from `_NoMountScreen`),
the workspace activation path renamed its defaults accessor to
`_blank_console_session_settings` and added
`_console_new_chat_default_generation` (missing from the reconcile stubs),
and the controller constructor gained an undocumented
`notify_character_navigation` dependency. Notably, the atomic-rollback
resume test was passing VACUOUSLY through the wrong except path because the
drift exception fired before every injected failure boundary.

ADR required: no — test-only fixture repair plus one docstring line.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both suites fully green on dev: 115 controller tests, 9 reconcile tests
- [x] #2 The cancellation/rollback resume test exercises its injected failure boundaries (no longer short-circuited by fixture drift)
- [x] #3 No production behavior changes (docstring addition only)
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Reproduce on pristine dev; capture the actual drift exception
   (`AttributeError: '_NoMountScreen' object has no attribute
   '_build_console_provider_selection'`) by spying the resume steps.
2. Teach `_NoMountScreen` a stable provider-selection snapshot; teach the
   reconcile stubs the renamed/added accessors; document the constructor
   dependency.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- `Tests/UI/test_console_workspace_controller.py`: `_NoMountScreen` gains
  `_build_console_provider_selection` returning a stable SimpleNamespace
  (equal-to-itself for the token-prep change guard).
- `Tests/Workspaces/test_console_workspace_reconcile.py`:
  `_RealActivateStub` gains `_blank_console_session_settings` (None,
  matching its historical settings shape) and
  `_console_new_chat_default_generation` (0).
- `tldw_chatbook/UI/Console_Modules/workspace.py`: docstring-only — the
  `notify_character_navigation:` Args entry.
- Evidence: controller suite 115 passed, reconcile suite 9 passed (were
  5 and 2 failures respectively).
<!-- SECTION:NOTES:END -->
