---
id: TASK-1974
title: 'Change review: per-file revert and Undo-all with user-edit guards'
status: To Do
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - console
  - change-review
  - safety
dependencies:
  - TASK-1970
  - TASK-1971
  - TASK-1973
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore-to-B actions from the Review screen: per-file (`u`) and whole-turn Undo-all (`U`), both confirmed. Modified/deleted restore via checkout from B; CREATED files un-create via explicit guarded delete (`checkout B -- path` errors on a path absent from B); renames restore old and remove new. Guard: before any revert, each target's disk state is compared to E — files that differ (user or later turn edited them) are listed BY NAME in the confirm before anything is overwritten. Every revert takes a fresh snapshot and updates the row's `reverted` field. Runs under the per-root lock; per-file outcomes reported individually — partial failure is never silent.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Round-trip tests: create/modify/delete/rename each reverted back to exact B state on disk
- [ ] #2 Un-create removes the file without touching neighbors; revert of a B-absent path never calls checkout
- [ ] #3 Editing a file after E then reverting lists that file by name in the confirm dialog (sabotage: removing the guard fails the test)
- [ ] #4 Undo-all on a turn whose file was ALSO changed by a later turn warns before clobbering
- [ ] #5 After revert, a fresh snapshot exists and `reverted` reflects what happened
- [ ] #6 A revert failure on one file reports that file and completes the rest
- [ ] #7 Revert refuses with 'finish or stop the run first' while ANY run is active on the root (sabotage: removing the guard fails the test)
- [ ] #8 Undo-all on a multi-root turn restores every root's files
<!-- AC:END -->
