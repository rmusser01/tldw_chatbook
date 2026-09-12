---
id: TASK-32268
title: >-
  Library Notes delete prompt renders six rows below Delete and outside the
  Info border, contra the guide
status: Done
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Residual of task-32132, which fixed the pane-snap: the prompt now stays on Info instead of throwing the pane to Edit. What remains is that it renders six rows below the Delete button that raised it and **outside** the Info border, so the guide's claim that the confirmation "renders where Delete was pressed" with Info staying open is overstated, and the prompt is detached from the control at the moment of highest anxiety for a first-timer.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The prompt renders adjacent to the Delete control and inside the Info border
- [x] #2 The guide's claim matches the live surface
- [x] #3 Covered by a test asserting the prompt's position relative to the control that raised it
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live: measure the prompt's rows against the Info border and the Delete button.
2. RED test asserting containment and adjacency.
3. Compose the prompt inside Info's Danger section; keep Info the surface while confirming.
4. Handle the scroll the move introduces; GREEN; verify live at 235x52 and 100x30; guide; stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Cause PROVEN live at dev 4a14b3f36f (`wave3-caps/editor-keys/22-delete-prompt.txt`): Info's border closed at row 31, Delete sat at row 27 inside it, and the prompt painted at rows 32-34 — outside the box, five rows below the button. `_compose_editor` mounted `#library-note-delete-confirmation` as a sibling of every region, after the (permanently hidden) wide utilities and the conflict callout.

Fix: the prompt composes as the next child of Info's Danger section, immediately after `#library-note-context-delete`. `apply_session_state` now also states the invariant that placement depends on — Info is the surface whenever `confirming_delete` is set — instead of leaving it incidental to Delete being Info-only (task-32132's ruling).

The move puts the prompt inside Info's scroll, which focusing Cancel then drags. `LibraryNotesState.delete_origin_scroll` remembers the offset the reader was on when the prompt opened, and cancelling restores it (with `immediate=True`, because the default defers the scroll past the next refresh where the still-running focus animation wins) and focuses Delete without a second scroll. That keeps `test_library_note_delete_captures_context_origin_before_gated_flush`'s promise, which is what "restore the origin" has always meant there.

Verified live at 235x52 (prompt at row 28, directly under Delete at row 27, both inside a border that closes at row 49) and at 100x30.

Files: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`, `tldw_chatbook/UI/Library_Modules/library_notes_controller.py`, `tldw_chatbook/UI/Library_Modules/library_notes_state.py`, `Tests/UI/test_library_notes_wave_editor_keys.py`, `Tests/UI/test_library_shell.py`, `Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
