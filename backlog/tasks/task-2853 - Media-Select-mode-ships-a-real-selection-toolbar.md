---
id: TASK-2853
title: Media Select mode ships a real selection toolbar
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 01:10'
updated_date: '2026-08-07 11:07'
labels:
  - library
  - media
  - ux
  - uat-2026-08-06
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 (LIB-05, observed at dev `6ffa56516`). Owner ruling 2026-08-07: ship a
selection toolbar (not remove Select).

Media (3) → "Select" enters a mode offering only checkboxes, an "N selected" count, and "Done".
No action consumes the selection; pressing Done discards it; the bottom preview pane meanwhile
keeps showing a previously selected different item. The control advertises bulk capability — the
power user's #1 need (bulk export/delete) — and delivers a no-op, poisoning trust in every other
control's promise.

Scope per ruling: a selection toolbar with real actions. Export selection first (Export canvas +
context-scoped export already exist — wire selection in as a scope). Delete-selected with confirm.
"Add to collection" only when collection item adapters exist (do not block on them).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 While items are selected, a toolbar offers at least "Export selection" and "Delete selected" with the selection count
- [x] #2 Export selection produces a bundle containing exactly the selected items (verified against the zip)
- [x] #3 Delete selected asks for confirmation naming the count, then soft-deletes and updates list + rail counts
- [x] #4 Leaving Select mode without acting discards the selection explicitly (copy states it) and the preview pane never shows an item outside the current selection context
- [x] #5 Live TUI verification of both actions end-to-end
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify the UAT repro at HEAD (found: Export selected/Select all/Clear already wired via an earlier task in this program; only Delete selected + the discard notice + preview-hide were genuinely missing).
2. Add Delete selected to the Media canvas's select-mode toolbar (TDD): armed in-place confirm (Delete selected -> confirm copy naming the count + Delete/Cancel), mirroring the existing single-item viewer delete pattern.
3. Wire the confirm to a new _delete_library_media_selection worker that loops the existing media_reading_scope_service.delete_media_item seam (mark_as_trash) per id, updates _local_source_records/_local_source_counts, reconciles the selection, and exits Select mode on full success.
4. Fix the adjacent preview-pane defect (hide preview entirely while Select mode is active).
5. Add the explicit 'Selection discarded (N items)' notice on every select-mode exit path.
6. Live-verify end-to-end in tmux; fix any defect live verification surfaces before considering the task done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Re-verified at HEAD before implementing: Export selected/Select all N shown/Clear were
already wired (an earlier task in this 8-task program had built the id-scoped export
plumbing generically for media/conversations/notes). Genuinely missing: Delete selected,
the explicit discard notice on Select-mode exit, and the preview-pane defect.

Shipped for Media only (per AC scope):
- "Delete selected" button next to "Export selected", disabled/tooltip mirroring the
  export button's F-018 pattern. Press arms an in-place confirm ("Delete N selected
  items? This moves them to trash." + Delete/Cancel) -- the SAME two-step armed-button
  idiom the single-item viewer Delete already uses (chosen over IngestGuardrailModal,
  which is reserved for pre-flight warn-but-continue import summaries, not a delete
  confirm). Confirming calls the existing media_reading_scope_service.delete_media_item
  seam (mark_as_trash) per id -- no raw SQL, no new DB method.
- Row checkboxes are frozen while the confirmation is showing so the confirmed count
  can never drift from what actually gets deleted.
- On completion: succeeded ids drop from the in-canvas list AND
  _local_source_counts["media"] (the rail's "Media N" count) is decremented; a partial
  failure keeps Select mode active with only the failed ids still checked and a quiet
  warning naming the failure count.
- Preview pane hidden entirely while Select mode is active (was showing a stale
  pre-select-mode item -- the UAT's LIB-05 finding).
- Every Select-mode exit path (Done toggle, type-filter cycle) now routes through one
  shared helper that clears the pending bulk-delete confirm flag too and surfaces
  "Selection discarded (N items)." when the cleared selection was non-empty.

Live-verification-found-and-fixed defect (not introduced by this task, but blocking
AC1's "real toolbar" -- reproduced against the pre-task-8 baseline commit too): the
select-mode toolbar's "N selected" Static had no explicit width, and inside the
ds-toolbar Horizontal it silently claimed ~1700 columns on a 170-column terminal,
pushing every sibling Button off-screen -- present in the DOM (headless pilot tests
never caught it) but genuinely invisible live. Fixed with an explicit
`.styles.width = "auto"` on that Static. Also found+fixed: the bulk-delete completion
used the canvas-scoped `_sync_library_canvas` (which explicitly skips the rail per its
own docstring), so the rail's "Media N" count went stale after a delete even though the
underlying count was correctly decremented -- switched the completion tail to
`self.refresh(recompose=True)`, matching the single-item delete's own precedent; a new
test pins the exact `{"recompose": True}` call.

Tests (TDD throughout, RED confirmed before each GREEN): pure-state passthrough test in
test_library_media_state.py; canvas-render tests (button disabled/tooltip states,
confirm-row swap, toolbar controls absent while confirming, preview hidden in Select
mode) and screen-handler tests (arm/cancel/confirm, row-press guarded during confirm,
discard-notify on every exit path) in test_library_multiselect_media.py; two REAL
file-backed-DB tests (full success + partial failure) driving the actual
delete_media_item -> mark_as_trash seam end to end, asserting is_trash=1 in the DB
(never mocked); a REAL (non-mocked) export-roundtrip test in
test_library_export_roundtrip.py building an actual zip from an id-scoped selection and
asserting the exact 2 selected media items are present (and the one left out is not).
169 targeted tests pass; Tests/Library collect-only sweep 1079 (up from 1077 baseline,
both new tests accounted for); the 4 pre-existing failures in test_library_screen.py/
test_library_shell.py were independently reproduced against the pre-task-8 baseline
commit (0d01a67ff) via a throwaway worktree, confirming they predate this task.

Live TUI verification (tmux, scratch profile): ingested fixtures, entered Select mode,
selected 2 of 3/2 items, Export selected -> real zip unzipped and inspected (exactly 2
media content files, correct text, the unselected item absent); Delete selected ->
confirm banner named the count -> Cancel preserved the selection -> re-armed -> Delete
-> rows AND the rail's "Media N" count both dropped to 0 in place (no navigation away
and back needed) -> underlying sqlite rows confirmed is_trash=1, deleted=0 (soft
delete, not hard delete).

Files changed: tldw_chatbook/Widgets/Library/library_media_canvas.py,
tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/Library/library_media_state.py,
tldw_chatbook/Library/library_shell_state.py, Docs/User_Guide/library/media-and-conversations.md,
Tests/UI/test_library_multiselect_media.py, Tests/Library/test_library_media_state.py,
Tests/Library/test_library_export_roundtrip.py.

Concern for the controller: library_screen.py grew by ~250 net lines in this task alone,
on top of the pre-existing ~19k-line file already flagged for a split (task-1378/1379).

Review round 2 fixes (Important x2): (1) the unbounded-width toolbar-count Static bug
was fixed only for Media as a Python one-off, leaving the byte-identical Conversations
counter (and Notes -- same pattern, found while fixing) still broken; replaced with a
shared `.library-toolbar-count { width: auto; }` CSS class (css/components/
_agentic_terminal.tcss, bundle regenerated) applied to all three canvases' counters --
one declaration, not three near-duplicate one-offs. New headless test in
test_library_multiselect_conversations.py mounts the REAL LibraryScreen + REAL CSS
bundle and asserts the Conversations counter's rendered region width stays bounded;
proved it actually catches the regression via a manual RED/GREEN check (stripped the
class, watched the test fail with the exact `1701 < 30` symptom, restored it). (2) the
bulk-delete completion never re-armed keyboard entry focus on its full-success exit
from Select mode, unlike `_exit_library_media_viewer`'s established task-2856 AC1
convention -- fixed (arms only on full success; a partial failure keeps Select mode
active, which is not a "return to a list" transition, so it does not arm). The
identical, pre-existing gap in the single-item viewer delete (`_delete_library_media_item`)
was fixed too rather than just flagged, since it is the exact same one-line pattern.
Both gaps now covered by tests (bulk success arms, bulk partial-failure does not,
single-item success arms), each verified RED before the fix.

Minor (no code change, recorded per reviewer): the media-type filter cycle silently
cancels an armed bulk-delete confirmation (routes through the same
`_exit_library_media_select_mode` exit helper as "Done", using generic "Selection
discarded" copy rather than delete-specific wording) -- accepted as the safest
behavior for an armed destructive action interrupted by an unrelated control.
<!-- SECTION:NOTES:END -->
