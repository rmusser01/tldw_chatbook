---
id: TASK-32176
title: >-
  Library Notes: Import once copy edges leave content-less records and a
  triplicated classification set
status: Done
assignee: []
created_date: '2026-09-09 09:14'
updated_date: '2026-09-09 17:15'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - import
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the task review of task-32125/task-32130. Three copy edges survived the
Import once fix-up: a JSON array containing one content-less record reads
the whole file as "Not a note file (app configuration)" instead of
importing the valid records and reporting which one failed; the review's
**Skip all** group action is scoped to the current review page but never
says so; and the set of non-importable classifications is spelled out in
three separate places in code
(`_NON_IMPORTABLE_CLASSIFICATIONS`, `_NON_IMPORTABLE`,
`_CLASSIFICATION_LABELS`), which is one extra place to forget when the set
changes.

Added 2026-09-09 (integration of PR #2549, reproduced twice): choosing
**Update existing** on an *Unchanged repeat* row aborts the whole import.
The row reads "Content: replace existing content · Folder placement:
unchanged"; `_execute_item` then raises
`ImportReceiptTransitionError("Membership receipt authority does not match
the approved plan.")` (`note_import_executor.py`, the
`len(membership_effects) != len(item.memberships)` guard), no receipt is
produced, and the user gets "Import needs attention." — the live `1 failed`
seen during the task-32135 pilot. **Pre-existing, not caused by this wave**:
a probe that imports a folder, re-imports it unchanged, presses **Update
existing** and imports fails identically on the merged wave tree and on the
wave base `f054f35ae1` (the guard and its message are byte-identical on
both). Changing the file first — a *Changed repeat* — updates cleanly
(`updated=1, failed=0`) on both trees, so the defect is specific to the
unchanged-repeat-plus-update combination the review lets a user pick.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A mostly-valid structured import file reports which specific
  record failed, instead of classifying the whole file as not a note file
- [x] #2 The **Skip all** (and equivalent group) action's label says "on
  this page"
- [x] #3 The non-importable classification set has one source of truth,
  used everywhere it is currently duplicated
- [x] #4 **Update existing** on an unchanged repeat either imports or is
  refused in the review, never aborts the run with no receipt
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: pin the Update-existing abort on an unchanged repeat at the executor seam (memberships present, add_membership False).
2. Fix the guard/zip in _execute_item to use the memberships the plan authorized (the receipts ledger only records effects when add_membership is true).
3. One source of truth: export NON_IMPORTABLE_CLASSIFICATIONS from note_import_plan_models; canvas derives its string set from it; ImportParseIssue uses it; test pins the label map covering every enum member.
4. Group action labels say 'on this page'.
5. Structured sources: name the failing record instead of one generic whole-file sentence.
6. Docs stamp, live verify, backlog notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Four fixes, one per AC.

AC#4 (the reported abort) was a mismatch between two authorities: the receipt
ledger records a membership effect only when `item.add_membership` is set
(note_import_receipts.py:679), while `_execute_item` compared the effect count
against every membership the parser had proposed. Update existing on an
*unchanged* repeat is the one review choice that sets replace_content without
add_membership (the item still carries its proposed membership), so the guard
raised ImportReceiptTransitionError and the run ended with no receipt. The
executor now zips against `item.memberships if item.add_membership else ()`.
Create new is unaffected (it must approve membership) and Skip returns earlier.
Pinned by test_update_existing_on_an_unchanged_repeat_updates_without_new_placement,
which reproduces the exact message before the fix.

AC#3: `_NON_IMPORTABLE_CLASSIFICATIONS` is now the public
`NON_IMPORTABLE_CLASSIFICATIONS`; the canvas derives its string set from it and
the parser's ImportParseIssue contract uses it instead of a fourth hand-written
copy. A test also pins that every enum member has a group label, which is the
other thing that was easy to forget.

AC#1: `_ParseFailure` gained an optional `detail` that `_message` appends, so a
structured document names the record that failed ("Record 2 of 3 has no note
content." / "could not be read as a note.") and a CSV names the row. Applied at
the shared `_payload_from_mapping` callers, so JSON, YAML and CSV all benefit.
Deviation from the brief: the valid records are NOT imported around the failing
one. A source is either one parsed source or one issue in this pipeline; a
partial import would silently drop a record with no row in the review and no
line in the receipt, which is worse than refusing a file the user can fix and
re-import. The AC asks for the report, and the file is refused with the record
named.

AC#2: the group buttons read "Skip all on this page" / "Create all on this
page". They still fit the 60-column viewport test (the heading is 1fr).

Files: note_import_executor.py, note_import_parsers.py,
note_import_plan_models.py, library_note_import_canvas.py, the three test files
and Docs/User_Guide/library/notes.md.
<!-- SECTION:NOTES:END -->
