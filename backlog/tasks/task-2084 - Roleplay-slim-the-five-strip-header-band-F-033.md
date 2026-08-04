---
id: TASK-2084
title: 'Roleplay: slim the five-strip header band (F-033)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 09:39'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nav + title/subtitle/Ready + purpose + count + mode strip; 'Characters' appears 3x in 3 lines; library repeats the count. ~23% of screen at 100x30 before content. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Purpose and status merge into one line,Count is shown once,Header band loses at least 2 lines at 170x50,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests first: merged purpose+count line, no #personas-status-row in DOM, library count empty when unfiltered, workbench top edge moves up at 170x50. 2. PersonasScreen: drop the status-row Static; new _purpose_line_text() (descriptor + count from _character_total/_profiles/_dictionaries_cache/_lore_books_cache) replacing _status_row_text/_update_status_row at all call sites. 3. PersonasLibraryPane.update_rows: unfiltered non-recovery count line renders empty (count now lives in the header line); filtered/recovery/pagebar states unchanged. 4. Update tests pinning the old strips/copy. ADR required: no - UI composition/copy consolidation; no schema, boundary, or contract change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Merged purpose+count into one line ('Characters — who the AI plays · 2') via new _purpose_line_text() (counts from _character_total / len(_profiles) / _dictionaries_cache / _lore_books_cache); removed the #personas-status-row strip from compose and replaced _status_row_text/_update_status_row with _update_purpose_line at all 12 call sites. Library pane count line now renders empty for unfiltered lists (count shown once, in the header line) and keeps filtered/recovery/pagebar states. Band loses one strip (workbench top edge moves from y=11 to y=10 at 170x50, pinned by test) and the library list gains the duplicate-count row. Files: tldw_chatbook/UI/Screens/personas_screen.py, tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py; tests in Tests/UI/test_personas_{workbench,library_pane,library_pane_paging}.py. Verified: 58 targeted workbench tests + 30 pane/layout tests + 48 server-isolation race tests pass; dict/lore/scale suites green in the pre-commit gate run (10 race-test failures found and fixed via updated count expectations). ruff clean. ADR: not required (UI strip consolidation; no schema/boundary change).
<!-- SECTION:NOTES:END -->
