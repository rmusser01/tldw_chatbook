---
id: TASK-32249
title: >-
  Library Notes Markdown preview is capped at 20 rows at 235x52 while the
  compact layout gets 1fr, and its paging keys are inert on arrival
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
  - css
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cause PROVEN. `css/components/_agentic_terminal.tcss` carries `#library-note-preview-region { height: auto; min-height: 12; max-height: 20; }`, and the **very next rule** gives `#library-shell-grid.library-notes-compact #library-note-preview-region { height: 1fr }` -- so a 100-column terminal reads a long note better than a 235-column one. The lines are byte-identical at `c4a7b1911f`: pre-existing, newly found. No test pins the cap.

Confirmed live: the preview box closes at screen row 33 with 14 blank rows beneath it, the footer promises `pgup/pgdn scroll`, and PageDown from the landed state changes nothing (`R/caps/03`, `04`); one click inside the region and the same key reveals the code block, the table, the callout and the wikilink (`R/caps/05`). Preview is the reading surface for the researcher persona, and a reader's reasonable conclusion from a 20-row window with blank space under it is that content is missing.

Two riders on the same surface, both P3: the preview leaks the Obsidian callout marker `> [!note]` instead of rendering the callout, and the status line still reads "Next: Keep editing; changes save automatically." while Preview is showing.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The preview uses the available height in the wide layout (`1fr`, as the compact layout already does); no fixed 20-row cap at 235x52
- [x] #2 Activating Preview focuses the region, so the footer's `pgup/pgdn scroll` promise is true on arrival without a click
- [x] #3 An Obsidian callout renders as a callout rather than leaking its `[!note]` marker into the text
- [x] #4 The status line does not advertise editing behaviour while Preview is showing
- [x] #5 Covered by a test pinning the preview height rule in the wide layout
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live at 235x52 on a seeded profile (preview height, paging keys, callout marker, status line)
2. Confirm the 20-row cap was already deleted by task-32217; pin it at the critique's width
3. Focus the preview region on activation, mirroring Info's handler
4. Rewrite Obsidian callout headers into plain blockquote headers before the Markdown widget sees them
5. Stop the status line advertising autosave while Preview is showing
6. RED->GREEN tests in Tests/UI/test_library_notes_w3_layout.py; guide + stamp
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1/#5 were already true at HEAD: task-32217 deleted `max-height: 20` from both the component source and `LibraryScreen.BUNDLED_CSS`, and `test_the_note_preview_takes_the_same_height_the_body_does` pins it at 170x48. Added a second pin at the critique's own 235x52 rather than re-deleting anything (`test_the_wide_preview_fills_the_work_pane_at_the_critique_width`); measured live, the box now runs to the pane floor.

Three things were still true there and are fixed:

- Activating Preview focuses `#library-note-preview-region`, mirroring the Info handler two functions away. The region is the sole scroll owner, so the footer's "pgup/pgdn scroll" line was false until the reader clicked inside the box.
- `render_obsidian_callouts` (`Utils/markdown_parsing.py`) rewrites `> [!note] Title` into `> **Note: Title**` before the Markdown widget parses it. Textual has no callout extension, so the marker rendered as literal text INSIDE the quote bar the callout was already drawing; keeping the bar and bolding the words is the whole fix. Applied at both the compose and the in-place sync site, and the sync's staleness comparison moved to the rewritten source so a callout note does not re-render on every sync.
- The status line no longer offers "Keep editing; changes save automatically" on a read-only surface: `state.presentation == "preview"` now yields "Next: Press Edit to change this note."

Live at 235x52 on a seeded profile: `wave3-caps/layout/02-preview-before.txt` (leaked `[!note]`, "Keep editing", focus ring on the Preview button) -> `11-preview-after.txt` ("▌ Note: Obsidian callout", "Next: Press Edit to change this note.").

Modified: `tldw_chatbook/Utils/markdown_parsing.py`, `tldw_chatbook/Widgets/Library/library_notes_canvas.py`, `tldw_chatbook/UI/Library_Modules/library_notes_controller.py`, `Tests/UI/test_library_notes_w3_layout.py`, `Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
