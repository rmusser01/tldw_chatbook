---
id: TASK-32131
title: Library Notes slash accelerator inserts itself into the filter it focuses
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 06:57'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - keyboard
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PROVEN live (parent + evidence assessor): pressing `/` in the Notes list focuses the filter and types '/', so 'Reading' becomes '/Reading' and the query is wrong. The footer advertises '/ find note' and the guide lists it as 'Focus the note filter'. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pressing / focuses the filter and leaves its content unchanged
- [x] #2 Covered by a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause differed from the initial theory: /, when the notes filter is NOT yet focused, was already handled correctly (screen-level on_key redirects and stops the event before it reaches the Input). The real leak only reproduced with the filter ALREADY focused -- pressing / a second time hit the early isinstance(Input) return and typed a literal '/', the same class of bug the rail search box's LibraryRailSearchInput already solved (task-1584). Fix: reuse LibraryRailSearchInput for #library-notes-filter instead of a plain Input -- its _on_key already re-arms (select-all) on a second '/'. Files: tldw_chatbook/Widgets/Library/library_notes_canvas.py. Tests: Tests/UI/test_library_notes_wave_editor_keys.py (2 new tests, both RED before / GREEN after).
<!-- SECTION:NOTES:END -->
