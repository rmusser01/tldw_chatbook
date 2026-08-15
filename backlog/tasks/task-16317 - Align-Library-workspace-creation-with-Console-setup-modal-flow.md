---
id: TASK-16317
title: Align Library workspace creation with Console setup-modal flow
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 02:33'
updated_date: '2026-08-15 03:29'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Library screen still creates bare unbound workspaces via the same identity helper the Console used before TASK-16316. Users now learn two different creation flows; aligning Library removes the inconsistency and gives every new workspace a folder binding at creation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The #library-create-local-workspace button opens the same setup modal (name prefilled, folder required, read-only default)
- [ ] #2 Cancel/Escape from the Library modal creates nothing
- [ ] #3 Confirmed creation binds the folder and activates the workspace, with the existing Library invalidation/refresh sequence
- [ ] #4 Bind-race at write time keeps the workspace and warns
- [ ] #5 Library workspace-creation tests updated for the modal flow and pass; cancel path covered
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no — direct reuse of ConsoleWorkspaceSetupModal (no Console-specific deps) and existing registry pre-checks; no schema or interface change. Steps: 1) Rewire LibraryScreen.create_local_workspace to push the setup modal with the same validator closure. 2) Confirm handler: identity at confirm time, create, bind, activate, existing invalidate/refresh. 3) Update Tests/UI/test_post_release_workspaces_library_depth.py creation tests to complete the modal; add cancel-path coverage. 4) Run suites + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- `LibraryScreen.create_local_workspace` (the `#library-create-local-workspace` button, its only entry point) now pushes the shared `ConsoleWorkspaceSetupModal` instead of creating immediately. The modal has no Console-specific imports, so it is reused as-is; the validator closure mirrors the Console one (`validate_workspace_name` -> `validate_folder_binding("workspace-pending", path)`).
- New `_confirm_library_workspace_create` follows the Console template from TASK-16316: identity computed at confirm time via `_next_local_workspace_identity`, create -> `add_folder_binding(path, allow_write)` -> `set_active_workspace`, then the Library's existing `_invalidate_library_workspace_depth_state()` / `_preserve_library_rail_scroll()` / `refresh(recompose=True)` and the TASK-713 activation-aware toast. Bind-race keeps the workspace and warns (no half-state re-prompt); Cancel/Escape creates nothing.
- Tests: creation tests in `Tests/UI/test_post_release_workspaces_library_depth.py` updated with a `_confirm_workspace_setup_modal` helper (fills the folder, pauses so the queued Input.Changed handler settles, then fires validation synchronously to bypass the 0.3s debounce). Added cancel-path test and a one-ro-binding assertion on the happy path. 13 passed.
- Files: tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/test_post_release_workspaces_library_depth.py.
- Known unrelated failures: `test_library_note_keyboard_capability_matrix[create_discard]` in Tests/UI/test_library_shell.py fails on the base branch too (verified via stash).
- PR: #1659 (branched off the TASK-16316 branch, PR #1657; merge order 1657 then 1659).
<!-- SECTION:NOTES:END -->
