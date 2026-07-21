---
id: TASK-415
title: Skills editor - confirmation or undo for Delete
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 15:18'
updated_date: '2026-07-21 16:04'
labels:
  - skills
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P1 from the 2026-07-21 Skills UX/NNG review (verified live). Delete in the skill editor is a single click with no confirmation and no undo and it sits directly beside Save with equal visual weight at the bottom of a scrolled form. A misclick permanently removes the skill directory including supporting files. Inconsistent with how carefully the same panel gates trusting a skill. NNG heuristic 5 (error prevention).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Deleting a skill requires an explicit confirmation step or offers a working undo,Confirmation communicates what will be removed including supporting files count when present,Save and Delete are visually differentiated (destructive styling or separation),Covered by tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Inline two-step delete mirroring the notes/media confirming-delete pattern (not a modal - house style): first Delete press arms _library_skill_confirming_delete, snapshots live field values into the editor state via new _snapshot_library_skill_live_fields (the confirm recompose would otherwise lose unsaved typed text - skills-specific hazard the autosaving notes editor doesn't have), disarms dirty-tracking across the recompose (bootstrap-trust pattern; dirty flag itself preserved), and renders skill_delete_confirm_copy ('Delete "name"? This removes the skill's directory and N supporting files and cannot be undone.') above a Delete/Cancel row replacing the normal toolbar. Only #library-skill-delete-confirm deletes; Cancel returns to the normal row. Bonus scope-consistent fix: create mode renders NO Delete button (the old one was a silent no-op there - nothing on disk to delete). Visual differentiation AC: Delete keeps library-media-action-danger (4-cell separation margin + muted) and task-449's Discard now sits between Save and Delete. 6 new tests (pure copy helper, confirm-row render, no-delete-in-create, arm/confirm/cancel handlers) watched fail first; 2 end-to-end flow tests updated to drive the confirm step. Suites: skills canvas 59 passed, Tests/Skills 129 passed.
<!-- SECTION:NOTES:END -->
