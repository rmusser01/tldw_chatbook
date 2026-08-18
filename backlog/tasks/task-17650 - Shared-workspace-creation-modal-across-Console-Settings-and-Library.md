---
id: TASK-17650
title: >-
  Shared workspace creation modal across Console, Settings, and Library
status: In Progress
assignee: []
created_date: '2026-08-17 00:00'
labels:
  - workspaces
  - ux
priority: high
dependencies: []
---

## Description (the why)

Workspace creation today collects nothing (Console/Library) or a name only
(Settings), and folder binding — the thing that makes a workspace do anything
for agents — is a separate post-creation Settings-only action. Users are left
with a new "Workspace N" entry and no idea what it changes (task-713/task-714
complaints). Replace instant creation on all three surfaces with one shared
modal that collects a name + optional folder bindings and explains, truthfully,
what a workspace does.

Spec: `Docs/superpowers/specs/2026-08-17-workspace-create-modal-and-project-skills-design.md` §4.
Plan: `Docs/superpowers/plans/2026-08-17-workspace-create-modal.md`.

## Acceptance Criteria (the what)

- [ ] Creating a workspace from Console rail, Settings ▸ Workspaces, or Library opens the shared modal; escape cancels with nothing created
- [ ] Folder paths are validated inline as they are added (missing/home/root/sensitive/nested-overlap rejected before Create), via a validator shared with `add_folder_binding`
- [ ] Name collisions render inline and keep the modal open
- [ ] Console make_active path reproduces the full session-activation sequence including the TASK-713 toast; unchecked path only resyncs context
- [ ] Folders bind read-only per ADR-028
- [ ] User Guide pages updated and the 2026-07-26 settings-workspaces spec carries a supersession note

## Implementation Plan (the how)

Execute `Docs/superpowers/plans/2026-08-17-workspace-create-modal.md` (7 tasks:
validator extraction → browse-from-modal spike → full modal → Console/Settings/
Library wiring → docs + live verification).
