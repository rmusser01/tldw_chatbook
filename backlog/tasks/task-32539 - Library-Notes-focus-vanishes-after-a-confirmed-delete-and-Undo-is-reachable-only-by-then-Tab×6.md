---
id: TASK-32539
title: >-
  Library Notes: focus vanishes after a confirmed delete, and Undo is reachable
  only by / then Tab×6
status: Done
assignee: []
created_date: '2026-09-13 06:46'
updated_date: '2026-09-14 19:25'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor B, personas Sam and Jordan, Edit/delete workflow. D7.

**What happened.** Info → Delete → Tab → Enter (delete confirmed) → "✓ deleted · Jordan first note" receipt with Undo / Dismiss and "Recently deleted (1)". Then: no focus mark anywhere; twelve Tabs boxed nothing; Enter did nothing. Undo was reached only by `/` (focus the filter) then Tab×6 — eight keystrokes to the one recovery action (B 19, 20, 21, 22). A's delete → Undo round trip used the mouse (A 19–21). Captures: B 19–22.

**Cause.** INFERRED: the post-delete focus is not parked on the receipt or a list row; task-32132 / 32268 fixed the prompt's placement and Tab cycle, task-32255 the restored row's reveal, none of them the focus target after confirmation. The docs describe the receipt but not where focus goes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Delete is confirmed, focus lands on the receipt's Undo (or, if the receipt is absent, the next list row) with a visible shape-based cue and a footer chip naming it
- [x] #2 Undo from the post-delete state costs at most two keystrokes
- [x] #3 A test pins the post-delete focus target and the chip
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce (done: after a confirmed delete _sync_library_canvas runs with no then=, focus is nowhere, Tab1 restarts at 'New', Undo is 7 tabs away).
2. RED pins: focus lands on #library-notes-delete-undo; the navigator footer names it.
3. Fix: pass then=partial(_focus_library_note_control, '#library-notes-delete-undo') on the post-delete canvas sync; add _LIBRARY_NOTES_NAVIGATOR_ENTER_LABELS and append the focus chip to the navigator tier.
4. GREEN + live.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced live at dev 2f97a42c9a (235x52): Info → Delete → Tab → Enter left
the receipt on screen with NOTHING focused; the next Tab restarted at the
toolbar's "New", putting Undo seven stops away
(`editor-00-32539-post-delete-no-focus-235x52`). The delete's own canvas sync
carried no `then=` at all.

**Two rounds, because the obvious fix passed its test and did nothing live.**

Round 1 added `then=partial(_focus_library_note_control,
"#library-notes-delete-undo")` and the navigator footer's label table
(`_LIBRARY_NOTES_NAVIGATOR_ENTER_LABELS`, consulted by
`_library_focus_enter_label`, appended by the shared chip helper). The
harness pin went green. Live it changed nothing.

Round 2 traced it with a patched `queue_after_recompose`/`recompose` pair.
`queue_after_recompose` REPLACES, and two later syncs each evicted the
intent: `_delete_library_note`'s own trailing sync (so the intent moved
there, where the guard already means "a delete succeeded and its receipt is
showing"), and then the Trash reload worker every delete starts, which ends
in a target-less `_sync_library_canvas`. The harness fake carries no
`list_deleted_notes`, so that worker never ran in tests — which is why the
first round looked fixed.

**Root-cause fix, at the seam.** (a) `canvas_sync`: a sync with no `then` of
its own carries only a DEFAULT identity restore, so it no longer overwrites
a follow-up another sync queued. The media branch has guarded its own
default that way since task-31567; this generalises the rule to the single
place every kind queues through, and an explicit `then` still supersedes.
(b) the receipt's Undo/Dismiss gained portable semantic roles
(`delete-undo` / `delete-dismiss`) in the paired
`_library_notes_semantic_role` / `_library_notes_role_target` tables —
without them the identity capture could not express the one recovery control
on screen, so any recompose lost focus on it.

**Tests.** `::test_focus_lands_on_undo_after_a_confirmed_delete` (focus, the
`library-canvas-action` shape class, the "enter undo delete" chip),
`::test_undo_costs_one_enter_after_delete` (AC#2, asserted on the service's
own `restore_calls`), and
`::test_a_background_sync_does_not_evict_the_post_delete_focus_intent`, which
gives the fake a `list_deleted_notes` for one test so the race runs on the
real route. All RED on detached origin/dev — the last one with the exact live
symptom, "focus is 'library-notes-filter'". Live green at 235x52 and 100x30,
and one Enter restored the note.

Modified: `tldw_chatbook/UI/Screens/library_screen.py`,
`tldw_chatbook/UI/Library_Modules/library_notes_controller.py`,
`tldw_chatbook/UI/Library_Modules/canvas_sync.py`, the pin file, the guide.
**Review round (Minors 1-3): the seam COMPOSES, it does not skip.** The first
shape of (a) skipped the default follow-up whenever a callback was already
pending, which also dropped task-32106's Items-pane scroll restore (the notes
editor-owned default, which touches no focus at all) and — since only
`recompose()` clears `_post_recompose_callback` — let a pending callback that
never got a recompose suppress every later default for that canvas.
`PostRecomposeCallback.queue_default_after_recompose` now folds the default in
AHEAD of whatever is pending, the same way `preserve_same_id_focus_after_
recompose` already did, so the default's non-focus work runs and the pending
intent still runs last and wins on focus. The media branch's local skip
(`canvas_sync.py:885`) was the same rule spelled twice and is deleted; its
restore no-ops unless focus is missing or on a pane grip, and
`test_media_focus_restore_never_clobbers_a_queued_follow_up` still passes at
both sizes. Pinned by `::test_a_pending_follow_up_does_not_cost_the_list_its_
scroll_offset` and `::test_a_stuck_pending_callback_does_not_suppress_later_
defaults`, both RED against the skip shape.

**AC#1's second branch is unreachable.** "If the receipt is absent, focus the
next list row" cannot happen: the focus intent rides the branch guarded by
`view == "list" and delete_receipt is not None`, so a successful delete always
has a receipt. Recorded rather than implemented — code for a state production
cannot produce is untestable by construction.
<!-- SECTION:NOTES:END -->
