---
id: TASK-32215
title: >-
  Library Notes: the Sort chooser exists only on the empty list; folder verbs
  are ungated
status: Done
assignee: []
created_date: '2026-09-10 14:53'
updated_date: '2026-09-11 02:00'
labels:
  - library
  - notes
  - ux
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The empty Notes toolbar reads `New · Sort: Newest · ○ Select`; with seven notes it reads `New · Select · Add from files… · Export` + `New folder · Add to folder · Move note · Remove placement` and Sort is gone, while three of the eight controls are meaningless without a selection and carry no `○` or reason. Media, Prompts and Skills keep their sort. (The Notes wave-2 branch may touch this toolbar; coordinate.) Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 12.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sort is available on a populated Notes list in the same slot as its siblings
- [x] #2 Selection-scoped folder verbs appear only with a checked row or are gated with the inline-reason grammar
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify Sort presence on a populated list (PR #2558) -- do not write a second Sort control
2. Verify whether the selection-scoped folder verbs are gated
3. Pin + fix whatever half still stands
4. Docs stamp + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both halves were already delivered by the Notes wave; this task contributes the missing pins and NO new controls.

AC#1 -- verified live at dev 1077ac2dad on the seeded 7-note profile: the populated toolbar reads `New · Sort: Newest · Select`, in the same slot as its siblings. PR #2558 (task-32172) composed Sort unconditionally with a Newest default and #2565 re-pins its presence; no second Sort control was written, and nothing here re-pins Sort in either direction.

AC#2 -- the snapshot measured 02374bf66a, before the folder-tree mode landed. At the current tip `_compose_tree_actions` composes `Add to folder`, `Move note` and `Remove placement` ONLY when the selected tree row is a note placement (folder verbs likewise only for a folder row); with nothing selected the toolbar carries just `New folder`, which is not selection-scoped. Verified live in all three states (captures in crit9/wave/notes/caps) and now pinned: Tests/UI/test_library_crit9_notes.py::test_selection_scoped_folder_verbs_need_a_selected_row, parametrized over selected/not, which passes on dev and is the evidence the claim is real rather than incidental.

Files: Tests/UI/test_library_crit9_notes.py (new).
<!-- SECTION:NOTES:END -->
