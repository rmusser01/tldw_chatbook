---
id: TASK-32130
title: >-
  Library Notes Import once copy: config and empty files reported as Failed, CSV
  fan-out undercounted in the review, receipt lists no skipped files
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
  - copy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed by the evidence assessor on the 71-file vault: `.obsidian/*.json` and an empty `Untitled.md` both read 'This source could not be imported safely.' (a config file is not a failed import and an empty file is not unsafe); `notes.csv` is reviewed as 'Content: create 1 new note' and imports two; the receipt '61 imported · 0 updated · 11 skipped · 0 failed' names none of the skipped paths or reasons, and 'All planned items settled.' is opaque. Independent of task-32129: these are honest-copy problems for any import. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Empty and whitespace-only files are reported as 'Empty, nothing to import' and config-like files as skipped with the reason
- [x] #2 The review's per-item plan states the number of notes a structured file will create
- [x] #3 The receipt has a 'Skipped (N)' disclosure listing each path and reason
- [x] #4 The completion copy says what happened in plain words
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Parser: EMPTY classification for empty/whitespace-only sources, SKIPPED for structured documents that hold no note record.
2. Planner: keep the parser's specific reason instead of one generic failure sentence.
3. Receipt: derive a Skipped (N) disclosure from the reviewed plan and replace 'All planned items settled.' with a plain completion line.
4. Pin the CSV fan-out count (already correct) with a test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two new `ImportClassification` members carry outcomes that were being reported as failures: EMPTY (`empty_source` -> 'Empty file — nothing to import.') for a zero-byte or whitespace-only source of any extension, and SKIPPED (`not_a_note` -> 'Not a note file (app configuration).') for a well-formed JSON/YAML document that holds no note record (no content/body key, or non-mapping records). Malformed data keeps `invalid_content`.

Root cause of the generic copy was in the planner, not the parser: `_issue_item` discarded `issue.user_message` and substituted one of two constants, so 'too large', 'not valid UTF-8' and 'could not be parsed' all read 'could not be imported safely'. It now uses the parser's own message, which fixes every reason code at once.

Receipt: `LibraryNoteImportSnapshot` gained `skipped_count`/`skipped_items`, derived from the reviewed plan's SKIP items (bounded to 50 rows, with a truthful 'Showing the first N' line beyond that), rendered as a `Skipped (N)` Collapsible. `_receipt_detail` replaces 'All planned items settled.' with 'Import finished · 61 notes created · 11 files skipped'.

AC#2 did not reproduce: a 2-row CSV already planned 'create 2 new notes' (`_effect_summary` counts payloads). Pinned with a test instead of changed.

Files: Notes/note_import_plan_models.py, note_import_parsers.py, note_import_planner.py, Library/library_note_import_state.py, Widgets/Library/library_note_import_canvas.py, Tests/Notes/test_note_import_planner.py, Tests/Library/test_library_note_import_state.py, Tests/UI/test_library_notes_wave_import_ux.py. Live-verified end to end on the 71-file vault (caps 06/07/08).
<!-- SECTION:NOTES:END -->
