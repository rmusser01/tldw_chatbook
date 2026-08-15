---
id: TASK-16317
title: Align Library workspace creation with Console setup-modal flow
status: In Progress
assignee:
  - '@robert'
created_date: '2026-08-15 02:33'
updated_date: '2026-08-15 02:33'
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
