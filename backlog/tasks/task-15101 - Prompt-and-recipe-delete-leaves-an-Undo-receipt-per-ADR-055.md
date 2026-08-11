---
id: TASK-15101
title: Prompt and recipe delete leaves an Undo receipt per ADR-055
status: To Do
assignee: []
created_date: '2026-08-11 06:20'
labels:
  - library
  - ux
  - adr-054
dependencies: []
priority: medium
---

## Description

ADR-055 (task-14901) sets one reversibility rule across Library destructive
actions: soft-deleting persisted user data leaves a receipt at the point of
action offering Undo, with the durable recovery story named. Prompt/Recipe
deletion is a soft delete (`local_prompt_service.delete_prompt` →
`Prompts_DB.soft_delete_prompt`, versioned, with retained version history per
ADR-049), so the deleted row is recoverable at the store level — but the UI
today goes modal-confirm-then-silence: after `_delete_library_prompt`
succeeds, the editor resets to the list with no receipt and no way back. The
modal copy now states permanence honestly ("This cannot be undone from
Library.", task-14901), which ADR-055 accepts only as the interim state; the
owed pattern is the media one (task-4022): a "✓ deleted" receipt with
Undo/Dismiss in the prompts list, restoring through a service-level un-delete
seam (to be added — `Prompts_DB` has no undelete today; the version-history
machinery of ADR-049 is the natural substrate). Covers both the single-item
modal delete and any selection-based bulk delete the modal already fronts.
Update the modal copy to promise the Undo once it exists. The PR-1473 one-flag
lesson applies to every mutator of the shared prompts list/count/receipt
state.

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A successful prompt/recipe deletion leaves an in-place receipt in the prompts list naming what was deleted, with Undo and Dismiss, matching the media receipt grammar
- [ ] #2 Undo restores the artifact through a service seam (no raw SQL) and the list and rail Prompts count update in place
- [ ] #3 Undo and any concurrent prompt delete cannot race on shared list/count/receipt state (one shared in-flight interlock, PR-1473 pattern)
- [ ] #4 The delete modal copy promises exactly what exists (Undo), replacing "This cannot be undone from Library."
<!-- AC:END -->
