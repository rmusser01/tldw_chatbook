---
id: TASK-32258
title: >-
  Library Notes import receipt repeats itself three times and review progress
  and receipt use three different denominators
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 16:13'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On one 71-file vault import, within a single journey: the review counted **66 items**, progress reported **67 complete**, and the receipt reported **59 created + 8 skipped**. Three numbers for one import, none of them wrong on its own terms, all of them visible to the same user in the same minute. The receipt itself then states its outcome three times in three formats.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One denominator is used across review, progress and receipt, or each surface states what it is counting
- [x] #2 The receipt states its outcome once
- [x] #3 Covered by a test asserting the three surfaces agree for a single import
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run one real import end to end and compare review/progress/receipt numbers.\n2. Name the denominator on every surface and state the receipt outcome once.\n3. RED/GREEN test asserting the three surfaces agree for one import.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
REPRODUCED exactly, then traced to a real difference in units.

Running the 71-file vault through the production chain gave the reported numbers: review "66 items", progress "67 of 67 complete", receipt "59 imported + 8 skipped". PROVEN cause: they are two denominators, not one wrong one. The review counts SOURCES (`len(plan.items)`); the receipt ledger counts PLANNED CHANGES -- `note_import_receipts._outcome_count` charges one row per NOTE a source creates, so the two-row `notes.csv` is worth two. 66 sources, 67 planned changes.

A second, smaller defect fell out of the same reading: `begin_importing` seeded the progress bar with `len(plan.items)` -- the reviews denominator -- and the first real progress message from the ledger then moved it under the reader (4 -> 5 in the regression test). The planned-change unit now has ONE definition (`note_import_plan_models.planned_change_count`) that the ledger and the progress seed share, so the bar cannot start on the wrong scale.

AC#1 is met by its second branch (each surface states what it is counting) rather than by forcing one denominator, because the ledgers unit is what the executor actually settles and the review row is what the user actually decides: "Review 66 sources before import.", "67 of 67 planned changes complete", and the receipt naming notes/files/links.

AC#2: the outcome is stated once. It used to appear three times -- "Import completed." in the header, "59 imported · 0 updated · 8 skipped · 0 failed", then "Import finished · 59 notes created · 8 files skipped". Now the header carries the session STATE, the receipt line carries the OUTCOME once, and the quiet line beneath reconciles the denominators out loud: "67 planned changes from 66 reviewed sources."

Verified live in the shell, end to end on the real vault (capture `31-live-receipt-235x52.txt`): "Import completed." / "59 notes created · 8 files skipped · 54 links resolved" / "67 planned changes from 66 reviewed sources."

Files: `tldw_chatbook/Library/library_note_import_state.py`, `tldw_chatbook/Notes/note_import_plan_models.py`, `tldw_chatbook/Notes/note_import_receipts.py`, `tldw_chatbook/Widgets/Library/library_note_import_canvas.py`, `Tests/UI/test_library_notes_wave_import_ux.py`, `Tests/Library/test_library_note_import_state.py`, `Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
