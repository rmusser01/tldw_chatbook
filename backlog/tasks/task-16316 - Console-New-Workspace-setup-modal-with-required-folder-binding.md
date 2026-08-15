---
id: TASK-16316
title: Console New Workspace setup modal with required folder binding
status: Done
assignee: []
created_date: '2026-08-15 02:22'
updated_date: '2026-08-15 02:23'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
New Workspace in the Console created a bare workspace with no folder binding and no hint of what it maps to. Creation now opens a setup modal capturing name and a required, validated folder binding.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 New Workspace opens a setup modal (name prefilled, folder required, read-only default)
- [x] #2 Create only enables when name and folder pass the same validation rules as add_folder_binding
- [x] #3 Cancel/Escape creates nothing
- [x] #4 Bind-race at write time keeps the workspace and warns instead of failing silently
- [x] #5 Unit tests cover modal gating, dismiss results, and the confirm create+bind+activate sequence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no — direct implementation of existing workspace-registry and SafeModalDismissMixin patterns; no schema or interface change. Steps: 1) extract shared folder-candidate validation into _folder_binding_error + public validate_folder_binding/validate_workspace_name. 2) New ConsoleWorkspaceSetupModal (SafeModalDismissMixin, debounced validation). 3) Rewire ConsoleWorkspaceController._create_console_workspace to open the modal; confirm-time identity; bind-race keeps workspace. 4) Tests: registry parity, modal gating, controller flow. 5) Update modal inventory contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- `LocalWorkspaceRegistryService`: the guards inside `add_folder_binding` were extracted verbatim into `_folder_binding_error()` (returns the error string or None), now the single source of truth for the binding gate. New read-only pre-checks `validate_folder_binding()` and `validate_workspace_name()` (blank + case-insensitive duplicate) wrap it; the write path raises from the same rules so pre-check and write cannot drift.
- New `Widgets/Console/console_workspace_setup_modal.py`: `ConsoleWorkspaceSetupModal` (SafeModalDismissMixin, Esc=request_safe_cancel, backdrop-safe) with prefilled name, required folder path, read-only/read-write checkbox (ro default), inline debounced (0.3s) validation, Create disabled until valid; Enter in the path field re-validates before dismissing. Dismisses `ConsoleWorkspaceSetupResult(name, folder_path, allow_write)` or None on cancel.
- `ConsoleWorkspaceController._create_console_workspace` now only opens the modal (suggested name from `next_local_workspace_identity`). New `_confirm_console_workspace_create` computes the collision-free identity at CONFIRM time (stale-open race), creates, binds, activates, then runs the existing sync/notify sequence. A write-time bind race keeps the workspace, warns, and points at the Folders settings — no half-state re-prompt.
- Tests: `Tests/UI/test_console_workspace_setup_modal.py` (gating, cancel, result payload, rw toggle, submit-on-invalid), `Tests/UI/test_console_workspace_create_flow.py` (create+bind+activate, None path, bind-race on a real registry), registry parity tests appended to `Tests/Workspaces/test_workspace_folder_bindings.py`, and the modal-inventory contracts in `Tests/UI/test_console_modal_dismissal.py` updated (Task3 table 12→13, inventory 28→29, reachable 37→38).
- Files: tldw_chatbook/Workspaces/registry_service.py, tldw_chatbook/UI/Console_Modules/workspace.py, tldw_chatbook/Widgets/Console/console_workspace_setup_modal.py (new), plus the three test files.
- Follow-up candidate (not done here): the Library screen still creates unbound workspaces via the same identity helper; aligning it with this flow deserves its own task.
<!-- SECTION:NOTES:END -->
