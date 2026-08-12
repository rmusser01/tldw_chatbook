---
id: TASK-15102
title: Collection delete leaves an Undo receipt per ADR-055
status: Done
assignee:
  - '@codex'
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
- [x] #1 A successful Collection deletion leaves an in-place receipt in the Collections panel naming the deleted Collection, with Undo and Dismiss
- [x] #2 Undo restores the Collection and its membership through a service seam (no raw SQL) and the panel list/count update in place
- [x] #3 The confirm affordance copy promises exactly what exists (Undo), replacing "the deletion cannot be undone from Library"
- [x] #4 Create, rename, delete, and receipt Undo share one mutation admission gate so stale completions cannot overwrite Collection list/count/receipt state
<!-- AC:END -->

## Implementation Plan

ADR required: no

ADR path: `backlog/decisions/055-library-destructive-action-reversibility-rule.md`

Reason: ADR-055 already defines the receipt, service-level restore, and shared state-mutation contract. The existing soft-delete schema and membership ownership remain unchanged.

1. Add a service-level `restore_collection` operation that revives only a deleted Collection row and returns the restored record with its untouched membership count.
2. Add pure receipt display state and render a literal named Undo/Dismiss toolbar above both populated and empty Collections states.
3. Route create, rename, delete, and Undo through one Collections mutation admission flag; preserve the receipt on restore failure and refresh selection, list, and count from the service on success.
4. Update both compose-time and targeted confirmation copy to promise the available list Undo.
5. Add service, state/widget, and mounted production-screen regression coverage, then run focused tests, static checks, and self-review before completing the task record.

## Implementation Notes

- Added an atomic `restore_collection` service seam that clears the soft-delete marker and returns the restored Collection with its retained membership count in the same transaction.
- Added a literal, bounded delete receipt above both populated and empty panel states. Undo restores and selects the Collection; Dismiss leaves it deleted; restore failures retain the receipt for retry.
- Routed create, rename, delete, and Undo through one screen-level mutation gate, and updated compose-time plus targeted confirmation copy to promise the available Undo.
- Reused the established `ds-toolbar` and semantic action classes per the product design register, avoiding a new decorative border or modal interaction. Updated the Collections user guide.
- Verification: 60 focused service/state/widget/mounted-screen tests passed with one pre-existing stale empty-copy assertion deselected; its expected sentence is absent from both production state and the test's `origin/dev` baseline. The atomic restore regression also passed independently. `compileall`, Ruff (repository-compatible `E721` exclusion), and `git diff --check` passed.
- ADR check: no new ADR required; implementation follows existing ADR-055 without changing storage ownership, schema, or conflict policy.
