---
id: TASK-14901
title: One reversibility story across Library destructive actions
status: Done
assignee:
  - '@claude'
created_date: '2026-08-10 17:20'
updated_date: '2026-08-11 06:28'
labels:
  - library
  - ux
  - recritique-2026-08-09
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from task-4023's cross-task observation (2026-08-09, task-4022 review round 2)
and the re-critique's heuristic #4 score of 1: the Library ships three different
reversibility stories on one screen. Blank/session notes are silently GC'd with no
undo; bulk media delete gets an in-place Undo receipt (task-4022); single media
delete gets nothing at all (confirm, then silence). Prompt/skill drafts discard,
notes persist. One consistent contract — what is undoable, for how long, and what
receipt destruction leaves — needs a design decision that spans notes, media
(single and bulk), prompts, and skills, so it could not ride task-4023's
grammar/copy batch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A written rule states which Library destructive actions are undoable and what receipt each leaves; the rule is recorded in backlog/docs or a decision file
- [x] #2 Single media delete and bulk media delete follow the same receipt/undo pattern
- [x] #3 Notes deletion/GC behavior matches the written rule (or the rule explicitly names notes as the exception, with the reason)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit every Library-surface destruction path (media single/bulk, notes delete + blank-note GC, prompts delete, skills delete, collections delete, draft discards) — what happens today, soft vs hard, what recovery exists
2. Write THE RULE as ADR-055 in backlog/decisions/ with the inventory table and per-surface dispositions
3. TDD: single media delete adopts the bulk receipt/Undo seam (shared _library_media_bulk_delete_in_flight flag, shared worker group, _library_media_delete_receipt_ids, _undo_library_media_bulk_delete) — failing tests first in Tests/UI/test_library_multiselect_media.py, then implementation; PR-1473 race-test family must stay green
4. One-line copy fixes where copy is all that is missing: single-delete confirm copy now promises the undo (viewer), prompt delete modal says the saved artifact cannot be restored from Library, collections confirm tooltip states the consequence
5. File (not build) structural conformance gaps: notes delete receipt, prompts delete receipt, collections delete receipt — ids leapfrogged past the cross-worktree max (15020)
6. Docs stamp: Docs/User_Guide/library/media-and-conversations.md single-delete row + verification stamp
7. Targeted tests + collect-only sweep; live tmux verification of delete-to-receipt-to-Undo; commit
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**The rule is ADR-055** (`backlog/decisions/055-library-destructive-action-reversibility-rule.md`,
indexed in the decisions README): four patterns picked by two questions —
(A) soft-deleting persisted user data owes a receipt + Undo at the point of
action with Trash as the durable story (task-4025 implements the surface for
media); (B) hard deletes must state permanence in the confirm copy; (C) the
blank-note GC is the one NAMED exception, silent only because all of its
guards hold (session-blank id match, no destructive op admitted, seed-title
provenance via `_library_note_title_user_edited`, body/keywords blank,
version-checked best-effort); (D) unsaved-draft discard is
confirm-not-receipt. The ADR carries the full audited inventory table with
per-surface dispositions.

**AC#2 — single media delete is now one-item bulk.** The viewer confirm
handler claims the SAME `_library_media_bulk_delete_in_flight` flag and
exclusive worker group as the bulk delete/Undo pair (a press while either is
in flight is refused); the arm press supersedes any stale receipt; on
success `_delete_library_media_item` sets
`_library_media_delete_receipt_ids = (media_id,)` and the list renders the
existing "✓ deleted · 1 item" receipt whose Undo/Dismiss handlers are
untouched — no second undo path, and the flag clears in a `finally`. The
viewer confirm copy now promises the undo, mirroring the bulk copy's shape.
Live-verified end-to-end in tmux (fresh sdd_lq3 profile, seeded chunked
item): delete → receipt + rail count 2→1 → Undo → row back, count 1→2,
receipt cleared; no probe input leaked into the live config (grepped).

**AC#3 + conformance sweep.** Notes: real deletion is Pattern A, honest
interim copy today, receipt+Undo filed as task-15100; blank-note GC is the
rule's named exception with its guards recorded. Prompts: soft delete —
modal copy now states "This cannot be undone from Library." on all three
variants (one-line copy fix); receipt filed as task-15101. Skills: hard
delete already states "cannot be undone" — conforms as Pattern B, no change.
Collections: soft delete — the "Confirm delete" tooltip now states the
consequence in both the compose and the in-place patcher (recompose
discipline); receipt filed as task-15102. Follow-up ids leapfrogged past the
cross-worktree/origin-dev max (15020 observed).

**Tests**: 5 new tests (3 handler-level, 2 real file-backed DB incl. a
chunked-row full delete→Undo cycle) + 3 tightened modal-copy pins; RED
observed before implementation. Full battery green: 46
test_library_multiselect_media (incl. the PR-1473 race family), 20
test_prompt_delete_confirmation_modal, 205 test_library_prompts_canvas (run
in 4 chunks — the whole file legitimately exceeds the 240s tool timeout), 56
Library state/service, 11 collections maturity = 338 passing.

**Files**: `tldw_chatbook/UI/Screens/library_screen.py`,
`Widgets/Library/library_media_viewer.py`,
`Widgets/Library/prompt_delete_confirmation_modal.py`,
`Widgets/Library/library_collections_panel.py`,
`Tests/UI/test_library_multiselect_media.py`,
`Tests/UI/test_prompt_delete_confirmation_modal.py`,
`backlog/decisions/054-…` + README index, tasks 15100/15101/15102, user
guide pages media-and-conversations/prompts/collections (stamped).
<!-- SECTION:NOTES:END -->
