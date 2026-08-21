---
id: TASK-19026
title: Preserve wide Library Notes browse continuity
status: Done
assignee: []
created_date: '2026-08-21 15:29'
updated_date: '2026-08-21 16:04'
labels:
  - library
  - ux
  - notes
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the Library rail visible for wide Notes browsing while giving note editing and Files workspaces a focused full-width task surface that returns to the exact prior browse context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Wide Notes browse retains the Library rail at the existing breakpoint.
- [x] #2 The note editor and Files workspace use a focused task surface with a persistent Library/Notes return cue.
- [x] #3 Back restores Notes source, scope, selected identity, scroll, rail position, and semantic focus.
- [x] #4 Dirty, sync, conflict, mutation, and Escape guards remain authoritative.
- [x] #5 Compact Notes remains navigation-first, and resizing does not reset draft or browse context.
- [x] #6 Only touched Notes and direct-owner tests are run; no repository-wide pytest claim is made.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See `Docs/superpowers/plans/2026-08-21-library-notes-wide-browse-continuity.md`.

ADR required: no

ADR path: `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`

Reason: this task implements ADR-076's approved responsive Notes presentation
inside the existing Library screen, Notes canvas, source, focus, and guard
owners. It changes no storage, sync/conflict policy, service contract, or
cross-module interface.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Kept the wide Database Notes navigator beside the Library rail, then derived
  one focused-task presentation for database editing/loading and the retained
  Files workspace. Wide tasks show one `‹ Library / Notes` cue and yield the
  rail width without changing the user's persisted rail preference; compact
  Notes keeps its existing one-stage source strip and inner Back control.
- Added one frozen browse-return receipt over the existing semantic Notes focus
  identity plus independent rail scroll. Admitted database-row and list-to-Files
  transitions retain filter, sort, placement/note identity, list scroll, rail
  scroll, focus, and rail collapse state. Fresh rail entries and direct/deep
  links clear stale receipts; editor-to-Files preserves the original receipt.
- Routed the cue through the existing guarded database-editor and Files exits;
  no parallel dirty, save, sync, conflict, reload, mutation, or Escape policy
  was added. Existing focus generations veto stale deferred restoration.
- Verification: the final exact new-path gate passed 10 cases. The final
  touched/direct-owner selector passed 43 cases with 774 deselected; only the
  existing Requests dependency and Python 3.13 audioop deprecation warnings
  remained. Ruff check passed the exact three changed Python files and
  `git diff --check` passed. Ruff format-check still reports the same three
  whole-file baseline failures present at the planning commit, so unrelated
  formatter churn was removed and no formatter-clean claim is made. The
  Impeccable layout detector returned zero findings. No CSS changed.
- Five required one-at-a-time inverses failed and were restored: discarding the
  receipt jumped list scroll 7→54; disabling focused-task derivation left the
  rail visible; bypassing the Files leave seam made its awaited guard count 0;
  ignoring the focus-generation guard stole focus from Collapse to note row 0;
  and recomposing on breakpoint crossing replaced the retained TextArea.
- Documentation now uses ASCII-only wide-browse, wide-task, and compact-layout
  diagrams. Per user direction, repository-wide pytest and a live/tmux profile
  were not run; only modified/touched Notes component and direct-owner gates are
  claimed.
<!-- SECTION:NOTES:END -->
