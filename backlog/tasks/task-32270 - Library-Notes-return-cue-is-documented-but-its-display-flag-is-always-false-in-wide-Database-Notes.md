---
id: TASK-32270
title: >-
  Library Notes return cue is documented but its display flag is always false in
  wide Database Notes
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 16:48'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - docs
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The guide claims: "When Library navigation is closed, one stable cue names the return destination: `< Library / Notes`". PROVEN absent at 235 columns -- `library_browse_route_swap.py:133` sets `task_return.display = wide_focused_task`, and `wide_focused_task` is False whenever `adaptive_database_notes` is true, which is always, in wide Database Notes. So the control the guide describes can never render in the state the guide describes.

Either the cue should render there, or the guide should stop promising it; both are small, and the choice belongs with whoever owns the back-cue grammar (task-32139 shipped the compact half).

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cue renders in wide Database Notes with navigation closed, or the guide no longer claims it does
- [x] #2 Covered by a test asserting the cue's presence or absence in the state the guide describes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Prove the display flag: library_browse_route_swap's wide_focused_task requires not adaptive_database_notes
2. Take the AC's documentation branch -- the cue is a compact control (task-32136/32139)
3. Correct the three guide sentences that promise it on a wide terminal
4. Pin both the absence in wide Database Notes and the guide's own wording
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cause confirmed as stated. `build_library_notes_source_strip` computes `wide_focused_task = not adaptive_database_notes and not _file_notes_active() and not compact and _library_notes_focused_task_active()`. `adaptive_database_notes` is true for every wide Database-Notes canvas kind, and `_library_notes_focused_task_active()` needs either Folder files (killed by the second clause) or Database Notes (killed by the first), so the flag is unreachable at wide sizes. `_sync_library_notes_source_controls` computes the same thing; the one site that does NOT guard it (`_update_library_notes_responsive_state`) returns early for every adaptive-reader route, which is all of Database Notes.

Took AC#1's documentation branch rather than adding a fourth wide control: task-32136 already decided (user decision, 2026-09-09) that wide Folder files keeps both source switches instead of collapsing to the cue, so the cue is a COMPACT control and the wide claim was the drift. Three guide sentences corrected -- the "when Library navigation is closed" paragraph (superseded sentence named, not silently dropped), the note-work-area bullet, and the guarded-return paragraph, which now names the editor's own `‹ Notes`.

Pinned both ways: `test_the_return_cue_stays_off_the_strip_in_wide_database_notes` (the control's real state) and `test_the_notes_guide_does_not_promise_a_cue_the_wide_route_cannot_paint` (the guide's own wording).

Modified: `Docs/User_Guide/library/notes.md`, `Tests/UI/test_library_notes_w3_layout.py`.
<!-- SECTION:NOTES:END -->
