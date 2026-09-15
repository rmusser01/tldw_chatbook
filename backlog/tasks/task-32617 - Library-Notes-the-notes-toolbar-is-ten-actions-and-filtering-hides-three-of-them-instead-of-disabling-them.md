---
id: TASK-32617
title: >-
  Library Notes: the notes toolbar is ten actions, and filtering hides three of
  them instead of disabling them
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-15 06:41'
updated_date: '2026-09-15 18:39'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A P2 and assessor B D10, persona Alex, all workflows. Heuristic 8 contributor; the cognitive-load checklist fails chunking on this row.

What happened. Once a sync root exists the list toolbar carries ten visible actions across two rows -- New, Sort, Select, Add from files…, Export, Manage sync folders / New folder, Add to folder, Move note, Remove placement -- with no grouping rule a reader can infer (A cap 40). A counted the decision points on the whole surface: toolbar 10, Info 7, the import review 8 group headers plus 2 bulk controls plus 2 per row, sync setup 9, Folder-files Manage 6. Four of the five exceed four choices; only the Add-from-files chooser passes.

And with a filter active, Add to folder, Move note and Remove placement simply vanish -- only New folder remains -- while Sort alone gets an explanation ('Sort unavailable — clear the filter') (B cap 08 against cap 18). Hidden controls teach the user the feature does not exist; the screen already has the right pattern one control to the left.

Cause PROVEN by capture. Wave 4's layout work (tasks 32544/32549, PR #2684) stopped the toolbar clipping mid-word and gave Sort its reason; the breadth and the hiding were not in scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actions unavailable under a filter are disabled with their reason on screen, in the grammar Sort already uses, rather than removed
- [x] #2 The toolbar groups its actions by a rule a reader can state, and the secondary group is reachable without scanning ten peers
- [x] #3 The count of simultaneously visible actions in the notes list is stated as a deliberate number somewhere a later change will see it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the vanishing by driving the canvas directly -- filter on and off, selection present and absent -- instead of trusting the critique's attribution.
2. Disable the unavailable actions with the reason on screen, in Sort's grammar.
3. Name the second group so the reader can state the rule.
4. Write the action count down where a later change trips over it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**The critique's attribution was wrong, and the fix is better for it.** Driving `LibraryNotesCanvas` directly across the four combinations (filter on/off x selection present/absent) shows the three placement actions are dropped whenever `tree_selected_placement_id` resolves to no row -- filter or no filter. A filter is simply the usual way to end up with no selection. Fixing the SELECTION case therefore fixes the filtered case and every other one; fixing 'the filter' would have left the rest.

AC#1: `_tree_action_buttons`'s note branch used to require `selected.kind == 'note'`; it is now the `else` of the folder branch, and with no row selected the three actions compose disabled, marked with the shared '○' (`library_disabled_action_label`) and explained on one shared line -- 'Note actions unavailable — select a note in the list' -- through `library_disabled_reason_line`, the same helper and the same sentence shape as Sort's 'Sort unavailable — clear the filter' one control to the left. One line for the group rather than three, because every row of this pane is a measured budget. The visible count does not grow: the folder and note branches remain mutually exclusive, so this group is at most four actions in every state.

AC#2: the group gets a `destination-section` heading, 'Folders & placement' -- the same heading grammar Folder files uses for 'File actions' / 'Session Git' / 'Danger' one mode over. The rule a reader can now state: everything above it acts on the LIST, everything under it on the folder tree and the row selected in it. Suppressed in the compact shell, whose list rows are a measured budget (task-32123/32261) and where a label costs more than it explains.

AC#3: `NOTES_LIST_VISIBLE_ACTION_BUDGET = 13`, derived by COMPOSING the canvas's widest reachable state and counting the buttons (filter + selected note + restorable folder + import receipt + sync root), not by hand. The pin asserts equality, not '<=': a new action has to displace one, move behind a disclosure, or move the number on purpose.

**Modified:** `Widgets/Library/library_notes_canvas.py`, `Tests/Widgets/Library/test_library_notes_w5_toolbar.py` (new), `Docs/User_Guide/library/notes.md`.

**Red first:** both parameters of the disabled-with-a-reason pin failed with the buttons absent; the heading pin failed with no heading composed. A positive control (a selected note enables them and the reason line goes) guards the other direction.
**Fix round 1 (caught by an existing pin, not by review).** At 60x20 the
compact shell lost THREE rows of the notes list: the three blocked actions
wrap its 50-cell toolbar onto two more rows and their reason line takes a
third. `test_library_note_60x20_navigator_state_allocation` reads those rows
straight off `#library-notes-list` and failed 4-of-7 and then 6-of-7 before
the gate went in. The compact shell now keeps the unselected state exactly as
it was -- same call as the heading, and as task-32261's hidden select counter.
Checked against dev afterwards: that test is a BASELINE red on both sides with
the identical value (`Region(x=5, y=13, width=50, height=6)`, 3 failed /
1 passed on dev @ c0d3d4dad7 and on this branch), so the parity is by value,
not just by name.
<!-- SECTION:NOTES:END -->
