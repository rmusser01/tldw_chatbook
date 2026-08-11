---
id: TASK-15102
title: Collection delete leaves an Undo receipt per ADR-055
status: To Do
assignee: []
created_date: '2026-08-11 06:20'
labels:
  - library
  - ux
  - adr-055
dependencies: []
priority: low
---

## Description

ADR-055 (task-14901) sets one reversibility rule across Library destructive
actions: soft-deleting persisted user data leaves a receipt at the point of
action offering Undo, with the durable recovery story named. Deleting a local
Library Collection is a soft delete (`library_collections_service.
delete_collection` sets `deleted_at`; members are untouched), so the grouping
is recoverable at the store level — but the UI today goes
two-step-confirm-then-silence: after `confirm_library_collection_delete`
succeeds the panel just refreshes, with no receipt and no way back. The
confirm tooltip now states the consequence honestly (members survive, the
deletion cannot be undone from Library — task-14901), which ADR-055 accepts
only as the interim state; the owed pattern is the media one (task-4022): a
"✓ deleted" receipt in the Collections panel with Undo/Dismiss, restoring
through a service-level un-delete seam (to be added — clearing `deleted_at`
behind the service, never raw SQL from the screen). Lower priority than the
notes/prompts siblings: a Collection is a recreatable grouping, not primary
content. Update the tooltip to promise the Undo once it exists.

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A successful Collection deletion leaves an in-place receipt in the Collections panel naming the deleted Collection, with Undo and Dismiss
- [ ] #2 Undo restores the Collection and its membership through a service seam (no raw SQL) and the panel list/count update in place
- [ ] #3 The confirm affordance copy promises exactly what exists (Undo), replacing "the deletion cannot be undone from Library"
<!-- AC:END -->
