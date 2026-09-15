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
