---
id: TASK-32136
title: >-
  Library Notes Folder files is a mode of Notes: keep the rail and the source
  strip, and offer the configured sync folder — user decision 2026-09-09
status: In Progress
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 05:49'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - file-notes
  - layout
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The user decided Folder files is a mode of Notes, not a separate screen. Today switching to it replaces the whole canvas: the Library rail, the Notes list and the 'Library notes | Folder files' strip itself disappear, leaving a '‹ Library / Notes' cue; the empty state ('Choose a notes folder.') does not explain the mode or offer the already-configured `[notes] sync_directory`. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At wide sizes the Library rail and the source strip stay visible inside Folder files
- [x] #2 The empty state explains in one line what Folder files does and offers the configured sync folder when one is set
- [x] #3 The file-notes guide's layout tour matches
- [x] #4 Covered by a compose test at 235x52
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing compose test at 235x52: Folder files keeps the rail and the Library notes | Folder files strip.
2. Stop treating wide Folder files as a focused task that collapses the source strip (compose + _sync_library_notes_source_controls).
3. Empty state: one-line explanation plus a Use <folder> button for [notes] sync_directory.
4. Docs layout tour + stamps.
<!-- SECTION:PLAN:END -->
