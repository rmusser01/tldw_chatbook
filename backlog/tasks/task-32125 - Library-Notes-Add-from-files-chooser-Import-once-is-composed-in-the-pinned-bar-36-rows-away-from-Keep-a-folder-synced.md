---
id: TASK-32125
title: >-
  Library Notes Add from files chooser: Import once is composed in the pinned
  bar 36 rows away from Keep a folder synced
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 06:36'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
  - layout
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PROVEN: `_compose_phase` (phase 'choose') yields only the keep-synced button under the two descriptions; `_compose_pinned_actions` yields 'Import once' beside 'Back to Notes' at the bottom of the canvas. A top-down reader sees a one-option choice, and the header already reads 'Lasting sync · Choose how files should relate…' before anything was chosen. Both assessors flagged it independently. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Import once and Keep a folder synced render as sibling buttons directly under their own descriptions, in description order
- [x] #2 The pinned bar holds only Back to Notes on the choose phase
- [x] #3 The header does not name a relationship before one is chosen
- [x] #4 Covered by a compose test
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Move Import once out of the pinned bar into the choose body, each button under its own description.
2. Stop the canvas header naming Lasting sync before a relationship is chosen.
3. Compose test in Tests/UI/test_library_notes_wave_import_ux.py.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved the Import once button out of `_compose_pinned_actions` into `_compose_phase`'s choose branch so both relationship buttons render directly under their own description; the pinned bar now holds only Back to Notes. `LibraryNotesCanvas._authority_copy` names 'Add from files' (not 'Lasting sync') while the sync snapshot phase is 'choose', with 'Next: Choose Import once or Keep a folder synced.'

Files: Widgets/Library/library_notes_add_from_files_canvas.py, Widgets/Library/library_notes_canvas.py, Tests/UI/test_library_notes_wave_import_ux.py, Docs/User_Guide/library/notes.md. Live-verified on the fresh profile (caps 01/03).
<!-- SECTION:NOTES:END -->
