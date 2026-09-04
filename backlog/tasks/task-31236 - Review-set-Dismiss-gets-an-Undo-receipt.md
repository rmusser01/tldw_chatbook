---
id: TASK-31236
title: Review-set Dismiss gets an Undo receipt
status: Done
assignee: []
created_date: '2026-09-04 01:50'
updated_date: '2026-09-04 03:05'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P2: every picker row is [open][Dismiss] side by side; Dismiss fires immediately (soft-delete), toasts "Review set dismissed.", closes the dialog, and no reopen path exists anywhere in the UI. A mid-walk set with many done-marks dies to one mis-click — a hidden recovery state, the product's stated anti-reference. USER RULING (critique #3 close): undo on the toast, mirroring the media delete-receipt pattern — no extra click on the happy path. The rows are already soft-deleted (deleted_at), so undo is an un-tombstone, not a rebuild.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dismissing a set surfaces an immediate Undo affordance in the receipt
- [x] #2 Undo restores the set with its cursor, done-marks, and active/inactive state intact
- [x] #3 The undo window closing (timeout or next action) leaves the current soft-delete semantics unchanged
<!-- AC:END -->

## Implementation Plan

1. RED: service undismiss contract (restore + one-active yield), picker-dismiss receipt arming, undo worker, canvas receipt row
2. GREEN: ReviewSetService.undismiss; screen receipt state + handlers + worker; canvas row mirroring the bulk-delete receipt
3. Live tmux verify of dismiss → receipt → Undo → DB restore

## Implementation Notes

Service: undismiss(set_id, reactivate) clears deleted_at and re-activates ONLY when no other live set became active since (one-active invariant outranks the undo; single transaction). Screen: the PICKER_DISMISS branch captures (set_id, name, was_active) from the picker rows into _library_media_review_dismiss_receipt and no longer toasts — the receipt IS the confirmation, mirroring the delete-receipt precedent; Undo runs _review_dismiss_undo_worker (group library_review_set, exit_on_error=False, error notice on failure per the task-30042 silent-failure ruling). Canvas: "✓ dismissed · name" + Undo/Dismiss row after the bulk-delete receipt, review_dismiss_receipt_name threaded through both state builders. Trap re-hit and pinned in tests: the viewer-scoped sync seam does NOT recompose the canvas — both the dismiss branch and the undo worker need their own _sync_library_canvas(self, "media") or the receipt never mounts/clears (live-verified both ways). DB-verified: restored set has deleted_at NULL with cursor and done marks intact; a set dismissed while inactive comes back inactive.
