---
id: TASK-32619
title: >-
  Library Notes: the CSV import failure names the bad row but not the rows that
  were fine or the next action
status: Done
assignee:
  - '@robert'
created_date: '2026-09-15 06:42'
updated_date: '2026-09-15 17:45'
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
- [x] #1 A structured source that fails names how many rows parsed, which row and cell failed, and what to do about it
- [x] #2 The atomicity stays as pinned, and the pin stays green
- [x] #3 The message is asserted by a test against its exact sentence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read Notes/note_import_parsers.py's CSV path (_csv_payloads, _payload_from_mapping, _keywords) end to end to find every invalid_content raise site reachable from a CSV row.
2. Thread a stable field-role token through _ParseFailure.detail at each site (row/title/content/keywords/template) -- non-breaking for the JSON/YAML path, which already discards and replaces that detail with its own "Record N of M" message.
3. In _csv_payloads, map each role token to the row's own as-typed header text and build one message naming rows parsed so far, the failing row number, the cell (or the row, for a shape failure), and the next action.
4. Update the one pinned exact-message test (test_a_mostly_valid_structured_source_names_the_record_that_failed's CSV case) to the new sentence; leave the atomicity pin (test_csv_uses_recognized_columns_and_rejects_invalid_rows_atomically) untouched since it doesn't assert message text.
5. Prove the new pin red against a reverted copy of the fix, then restore.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: _csv_payloads (Notes/note_import_parsers.py) caught any
invalid_content _ParseFailure from _payload_from_mapping and re-raised it
with only "Row N could not be read as a note." -- naming the row but not
how many rows had already parsed OK, which cell was wrong, or what to do.

Fix: _payload_from_mapping (and _keywords, which it calls) now tag every
invalid_content raise with a stable role token ("row" for a whole-record
shape problem, else the mapping key CSV populated from a column: "title",
"content", "keywords", "template") via _ParseFailure's existing `detail`
parameter. This is a no-op for the JSON/YAML structured path
(_structured_payloads), which already discards the original detail and
substitutes its own "Record N of M could not be read as a note." sentence
-- unaffected, unchanged, still covered by its own pinned tests.

_csv_payloads maps each role token to the row's OWN as-typed header text
(from `headers`, not the casefolded `normalized_headers` used for role
detection) and raises through one new `_row_failure()` helper shared by
both the whole-row shape check and the per-cell _payload_from_mapping
failure, so the message is built in exactly one place. New message shape:
"<N> row(s) imported so far. Row <n>: fix the "<Column>" cell, then import
again." (or "fix that row" for a whole-row shape failure). Example (header
+ 5 data rows, row 6 = the 5th data row has an empty Content cell): "This
source could not be parsed as notes. 4 row(s) imported so far. Row 6: fix
the "Content" cell, then import again."

Atomicity (AC#2) is untouched -- the CSV loop still aborts and reports the
whole file FAILED on the first bad row;
test_csv_uses_recognized_columns_and_rejects_invalid_rows_atomically (which
asserts classification/reason_code, not message text) passes unmodified.

Test (AC#3): updated the CSV case of
test_a_mostly_valid_structured_source_names_the_record_that_failed
(Tests/Notes/test_note_import_planner.py) to assert the exact new sentence
for its fixture (title,content header; row 2 valid; row 3 has an empty
content cell) -- '1 row(s) imported so far. Row 3: fix the "content" cell,
then import again.' Proved red: reverted note_import_parsers.py to its
pre-fix HEAD content in a scratch backup-and-restore (backup taken first,
restored immediately after), reran the test, saw it fail, restored the
fix.

Full suite: Tests/Notes/test_note_import_planner.py +
test_note_import_obsidian.py + Tests/UI/test_library_notes_wave_import_ux.py
= 470 passed, 0 failed.

Modified: tldw_chatbook/Notes/note_import_parsers.py,
Tests/Notes/test_note_import_planner.py.
<!-- SECTION:NOTES:END -->
