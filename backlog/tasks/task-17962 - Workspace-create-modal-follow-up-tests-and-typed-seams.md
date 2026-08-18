---
id: TASK-17962
title: >-
  Workspace create modal: follow-up tests and typed seams
status: To Do
assignee: []
created_date: '2026-08-17 21:00'
labels:
  - workspaces
  - testing
priority: medium
dependencies:
  - TASK-17650
---

## Description (the why)

The TASK-17650 final whole-branch review closed with a set of deliberately
deferred coverage and typing gaps (its Recommendations §3). They are known,
ledgered, and none blocks the shipped behavior — but PR B (TASK-17651) reads
more fields off `WorkspaceCreateResult`, so these seams should be pinned and
typed before that work builds on them.

## Acceptance Criteria (the what)

- [ ] Activation-failure path (`set_active_workspace` raising after a successful create) has a seam test on each surface: Console handler, Settings `_done`, Library `_done` (Library's reorder is already pinned; Console/Settings are not)
- [ ] The Enter-submit fast path (`Input.Submitted` on `#workspace-create-name` → `_create`) has a pilot test using a real keypress (not a direct method call), per the task-17961 blind-spot lesson
- [ ] A test produces `failed_folders` through the real TOCTOU path (folder deleted between Add and Create), not a synthetic result object
- [ ] `_remove_folder` and unchecked-checkbox-survives-recompose have pilot coverage
- [ ] The three `_done`/handler callbacks and `WorkspaceCreateResult.project_skills` carry real type annotations (`WorkspaceCreateResult | None`; a typed tuple for `project_skills`)
- [ ] Decide (and implement or explicitly reject) restoring per-surface `description` provenance ("Created from Console/Settings/Library") lost to the uniform modal description

## Notes

Source: final review of `feat/workspace-create-modal` (see
`Docs/superpowers/plans/2026-08-17-workspace-create-modal.md` and the spec's
§7). Related open defect: TASK-17961 (focused compact-widget rendering).
