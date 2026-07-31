---
id: TASK-1610
title: >-
  Run completion must not destroy a dirty bench editor
status: In Progress
assignee: []
created_date: '2026-07-31 15:10'
labels:
  - evals
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Whole-branch review of the bench-authoring program (task-1482), Important 1. The selection-yank guard added for run completion checks selection identity only: when a sample or bench run completes while the user is still ON the launched bench, the completion select() recomposes the screen and destroys every typed field and staged target in the editor. The sample-worker case is the sharpest: "Create sample bench" is always available in the rail, and its completion yanks to a run group belonging to a DIFFERENT bench than the one being edited. Fix shape: BenchEditor grows a dirty flag; `_selection_unmoved_since_launch` (or the workers' completion paths) consults it and degrades to the existing "— see the Runs section." toast when the mounted editor is dirty.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] A run completing while the mounted bench editor holds unsaved edits never recomposes the screen
- [ ] The clean-editor and moved-selection behaviors are unchanged
- [ ] Tests cover the dirty-editor case for both workers
<!-- AC:END -->
