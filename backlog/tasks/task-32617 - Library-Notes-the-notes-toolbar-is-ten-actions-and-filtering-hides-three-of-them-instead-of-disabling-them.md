---
id: TASK-32617
title: >-
  Library Notes: the notes toolbar is ten actions, and filtering hides three of
  them instead of disabling them
status: To Do
assignee: []
created_date: '2026-09-15 06:41'
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
- [ ] #1 Actions unavailable under a filter are disabled with their reason on screen, in the grammar Sort already uses, rather than removed
- [ ] #2 The toolbar groups its actions by a rule a reader can state, and the secondary group is reachable without scanning ten peers
- [ ] #3 The count of simultaneously visible actions in the notes list is stated as a deliberate number somewhere a later change will see it
<!-- AC:END -->
