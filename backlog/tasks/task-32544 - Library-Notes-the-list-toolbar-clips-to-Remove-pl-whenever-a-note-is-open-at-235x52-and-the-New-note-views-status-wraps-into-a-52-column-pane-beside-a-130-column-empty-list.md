---
id: TASK-32544
title: >-
  Library Notes: the list toolbar clips to "Remove pl" whenever a note is open
  at 235x52, and the New note view's status wraps into a 52-column pane beside a
  130-column empty list
status: Done
assignee: []
created_date: '2026-09-13 06:46'
updated_date: '2026-09-14 17:06'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor A, everyone, Edit and Create workflows. P2 #9 + minor.

**What happened.** With a note open at 235x52 the list's second toolbar row reads "New folder  Add to folder  Move note  Remove pl" (A 05, 09, 36) — task-32127 pinned two full rows (`test_notes_toolbar_fits_two_rows_with_every_action_visible`) only for the list with NO note open. In the New note view the status "Ready · Next: Press Blank note, or choose a template." wraps inside a 52-column work pane while the empty list keeps ~130 columns (A 04). Captures: A 04, 05, 09, 36.

**Cause.** INFERRED (a width threshold below the pinned case; the guide already describes a third-row wrap for narrow panes). Docs contradicted: "Nothing is ever painted as half a word."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With a note open at 235x52 every list toolbar label paints whole (third row, overflow menu or shorter labels — the guide's narrow-pane rule applied at this width)
- [x] #2 The New note view gets the work-pane width its status line needs; the empty list does not keep more than half the canvas while the New note view is the task in hand
- [x] #3 Tests pin both widths through the production layout resolver
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live at 235x52 (note open) and in the New note view.
2. Pack the folder-actions row into as many rows as the pane can paint (shared toolbar_action_rows helper).
3. Make reader_has_item true while the create view owns the work pane (the view is carried by _library_selected_row_id, not _notes_state.view).
4. RED->GREEN pins in Tests/UI/test_library_notes_w4_layout.py through the production canvas and the production resolver.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two defects, two causes, both proven by tracing and reproduced live.

AC#1 -- the folder-actions group is the widest row this toolbar composes: with a NOTE selected it is 'New folder / Add to folder / Move note / Remove placement', 64 cells of buttons plus the .ds-toolbar row's own 1-cell indent, against the 64-column Items pane the production resolver gives the list at 235 columns beside an open note. Textual toolbars do not wrap, so the last button was laid out at x=48..68 and painted 'Remove pl'. Every existing toolbar pin selected a FOLDER (46 cells), which is why they were green. New pure helper `toolbar_action_rows(labels, pane_width)` greedy-packs any action group into the rows its pane can paint, in reading order; `_compose_tree_actions` now builds its buttons through `_tree_action_buttons` and emits one Horizontal per packed row. The row indent it budgets for was MEASURED off the mounted regions, not assumed -- without it the group still ran exactly one cell off the pane. The packer subsumes the old `stacked` column for this group (below 48 cells it already puts one action per row).

AC#2 -- the New note view fills the work pane, but it is not a `view`: the canvas switches to create mode on `_library_selected_row_id == LIBRARY_ROW_CREATE_NOTE` while `_notes_state.view` stays 'list' (see `_build_library_notes_state`'s mode ladder and `_library_notes_focus_region`'s first branch). `_sync_library_notes_reader_layout_from_shell` passed `reader_has_item=self._notes_state.view != \"list\"`, so task-31979's 'an empty work pane hands its width to the list' rule fired on a pane that was full: 138 columns of empty list beside a 48-column create view at 235. One clause added to that expression. The same root cause is task-32547's.

Live: 235x52, note open -> 'New folder  Add to folder  Move note' / 'Remove placement', whole (wave4-caps/layout/layout-10-remove-placement-after, before: layout-00-remove-pl). New note view -> the status line 'Ready · Next: Press Blank note, or choose a template.' on ONE line in a ~122-column work pane beside a 64-column list (layout-11-new-note-235-after, before: layout-01-new-note-52).

Files: Widgets/Library/library_notes_canvas.py, UI/Screens/library_screen.py, Tests/UI/test_library_notes_w4_layout.py.
<!-- SECTION:NOTES:END -->
