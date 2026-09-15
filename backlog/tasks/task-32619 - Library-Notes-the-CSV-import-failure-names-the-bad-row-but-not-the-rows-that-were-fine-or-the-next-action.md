---
id: TASK-32619
title: >-
  Library Notes: the CSV import failure names the bad row but not the rows that
  were fine or the next action
status: To Do
assignee: []
created_date: '2026-09-15 06:42'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor B D3, persona Riley, Obsidian workflow. A pinned design decision whose message, not whose atomicity, is the finding.

What happened. notes.csv holds a header plus 5 data rows; row 6 has an empty content cell. The review reports 'Failed (1) · vault/notes.csv · This source could not be parsed as notes. Row 6 could not be read as a note' (B cap 23) and four valid notes are silently dropped. Naming the offending row number rather than printing a stack trace is already better than most importers -- the gap is that the message never says how many rows were fine, which cell is wrong, or what to do next.

Cause PROVEN: Notes/note_import_parsers.py:848-856 re-raises a parse failure on any single row and aborts the whole file, pinned deliberately by Tests/Notes/test_note_import_planner.py::test_csv_uses_recognized_columns_and_rejects_invalid_rows_atomically. The atomicity is not in question here.

Adjacent open rider 32573 covers the generic 'Next: Review the error' clause that several Notes failures fall back to; this is the same rule applied to one message.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A structured source that fails names how many rows parsed, which row and cell failed, and what to do about it
- [ ] #2 The atomicity stays as pinned, and the pin stays green
- [ ] #3 The message is asserted by a test against its exact sentence
<!-- AC:END -->
