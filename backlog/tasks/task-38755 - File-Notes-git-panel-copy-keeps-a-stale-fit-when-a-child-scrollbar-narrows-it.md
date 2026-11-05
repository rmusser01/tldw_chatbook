---
id: TASK-38755
title: "File Notes git panel copy keeps a stale fit when a child scrollbar narrows it"
status: To Do
assignee: []
created_date: 2026-11-05 07:11
labels:
- follow-up
- library
- file-notes
priority: low
references:
- tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py
- Tests/UI/test_library_file_notes_git.py
dependencies:
- TASK-38693
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 7 of the 2026-10-23 follow-up burn-down. Checked on dev bb80905eb3, 2026-11-04.

The File Notes git panel (`Widgets/Library/library_file_notes_git_panel.py`) fits the copy in its fixed regions (the repository, status, last-action and selected-note lines) to each `Static`'s current width, in `_fit_fixed_regions` (:1909). Apart from state changes, the only geometry trigger for a refit is the panel's own `on_resize` (:1837), which schedules one after the next refresh. A child's width can change while the panel's does not, for example when a list's scrollbar appears and takes a cell. The copy then keeps its earlier fit until something else refits it.

TASK-38693's notes (PR #3405) record this at 40x20. The commit summary was first fitted to 28 cells. Then the list surface's scrollbar took a cell, the panel did not resize, and the summary showed the stale 28-cell fit until it later settled at 27. Round 6 saw that stale window last up to about 0.5 s. During it the line is clipped or elided for the wrong width.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When a fixed-region line's content width changes and the panel's does not (for example, a scrollbar appears), its copy is refitted to the new width by the next refresh
- [ ] #2 A test makes a child scrollbar appear at a narrow size and checks that the copy equals `_fit_two_line_copy` at the new width on the next refresh; it is red before the change
<!-- AC:END -->
