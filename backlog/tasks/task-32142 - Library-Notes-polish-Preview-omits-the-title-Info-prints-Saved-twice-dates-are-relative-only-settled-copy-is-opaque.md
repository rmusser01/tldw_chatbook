---
id: TASK-32142
title: >-
  Library Notes polish: Preview omits the title, Info prints Saved twice, dates
  are relative only, settled copy is opaque
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 06:58'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Minor observations from both assessors: the Preview mode shows the body without the title; the Info view repeats 'Saved' in the header and above the panel; 'Created 3m · Modified now' is the only date anywhere for a note; 'All planned items settled.' ends the import; the delete receipt survived an entire import journey. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preview shows the title above the rendered body
- [x] #2 Saved appears once per view
- [x] #3 Info shows an absolute timestamp beside the relative one
- [x] #4 Receipts are dismissed when the user leaves the list for another workflow
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1: added a new Static (#library-note-preview-body-title) inside the Preview VerticalScroll, immediately above the Markdown body -- the existing header-row title Static was already technically present but shared a crowded strip with the mode buttons and scrolled away from the body; this one reads as the document's own title, kept in sync via the existing title-update loop. AC#2: #library-note-context-status was a literal duplicate of the always-visible #library-note-status (same content_recovery text) -- set its display permanently False instead of show_context. AC#3: extended build_library_note_editor_state's Created/Modified parts with an absolute local timestamp reusing the SAME parsed source _parse_browser_timestamp already feeds to format_console_relative_age (no new raw-ISO exposure), formatted with the codebase's established strftime('%Y-%m-%d %H:%M') convention: 'Created 2026-09-08 21:14 · 3m ago'. AC#4: a delete receipt is scoped to the Notes list session -- added a one-line clear (_library_note_delete_receipt = None) to the two handlers that leave the list for another workflow (_show_library_file_notes for Folder files, handle_library_notes_add_from_files for Add from files); left the broader _supersede_library_notes_navigation function (called from 4 other, unrelated sites) untouched since the concern is specific to these two named workflows. Files: tldw_chatbook/Widgets/Library/library_notes_canvas.py, tldw_chatbook/Library/library_notes_state.py, tldw_chatbook/UI/Library_Modules/library_notes_controller.py, Docs/User_Guide/library/notes.md. Tests: Tests/UI/test_library_notes_wave_editor_keys.py (5 new tests), Tests/Library/test_library_notes_state.py (updated one existing assertion for the new absolute-timestamp format). Live-verified at 235x52 and 60x24: Preview title above body, Info shows Saved once, Properties show 'Created 2026-09-08 22:35 · 1h ago · Modified ... · v1 · 8 words'.
<!-- SECTION:NOTES:END -->
