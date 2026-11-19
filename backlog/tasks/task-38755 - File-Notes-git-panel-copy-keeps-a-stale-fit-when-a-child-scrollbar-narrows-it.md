---
id: TASK-38755
title: "File Notes git panel copy keeps a stale fit when a child scrollbar narrows it"
status: Done
assignee:
- '@claude'
created_date: 2026-11-05 07:11
updated_date: 2026-11-05 20:58
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
- [x] #1 When a fixed-region line's content width changes and the panel's does not (for example, a scrollbar appears), its copy is refitted to the new width by the next refresh
- [x] #2 A test makes a child scrollbar appear at a narrow size and checks that the copy equals `_fit_two_line_copy` at the new width on the next refresh; it is red before the change
<!-- AC:END -->

## Implementation Plan

1. Add a red test in `Tests/UI/test_library_file_notes_git.py`: at 40 columns, mount the panel with long last-action copy and no list scrollbar, render enough rows that `#file-notes-git-list-surface` shows its scrollbar (the panel keeps its size), wait one refresh, and assert the summary equals `_fit_two_line_copy` at its new width.
2. Make the four fixed-region lines a `Static` subclass that keeps its source copy and refits on its own `Resize`, and have `_fit_fixed_regions` hand each line its copy.
3. Run the covering tests and append the new node at the census tail.

## Implementation Notes

- The four fixed-region lines (repository, status, last action, selected note) are now `_FixedRegionLine`, a `Static` that keeps its source copy and refits itself on its own `Resize`. `_fit_fixed_regions` hands each line its copy through `show_copy`, with the panel's width as the fallback for a hidden line, as before. A scrollbar that narrows a line without resizing the panel now refits it at once.
- Test: `test_fixed_region_copy_refits_when_a_child_scrollbar_narrows_it` (`Tests/UI/test_library_file_notes_git.py`). At 40x15 the empty state fits the list surface. Three rows bring the surface's scrollbar, so the summary goes from 40 to 39 cells while the panel stays at 40. After one `pilot.pause()` the summary must equal `_fit_two_line_copy` at 39. Red before the change: the summary still showed the 40-cell fit ("...unrelated \n...s; unrelated..."). Green after.
- Covering runs: `test_library_file_notes_git.py` and `test_library_file_notes_git_push.py`, 306 passed (`-n 4`, load 17).
- The new node is appended at the UI PR gate census tail. The `library_file_notes_git_panel.py` module-size row goes 4265 -> 4278 under the owner's size waiver.
- Files: `tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py`, `Tests/UI/test_library_file_notes_git.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `Tests/Architecture/test_module_size_ratchet.py`.

