---
id: TASK-32503
title: >-
  Library Notes: a saved note's row keeps its stale age until the next refetch
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Saving an in-canvas note edit and pressing Back updates that row's TITLE in
the folder tree but not its AGE: the row still reads the age it had before
the edit until the authoritative refetch lands.

Reproduction (dev ff2dc03145 + the task-32201 fixture repair, which is what
first let this assertion run):
`Tests/UI/test_library_shell.py::test_library_shell_note_save_then_back_refreshes_list_title_and_age`
->

    assert ['Reading list (edited)', '9w'] == ['Reading list (edited)', 'now']

The title half is right, so the in-place patch runs. `_patch_library_note_
list_from_session` passes `modified_at=baseline.modified_at` into both
`patch_note_records_after_save` and `patch_notes_tree_branches_title`, so the
stamp it patches WITH is the session baseline's -- which the save path does
not advance, since the local save reply carries no new modified time. The
row therefore re-renders with the seeded stamp and the age label is
unchanged.

Left unfixed by task-32185 deliberately: the row label (and the projection
age that feeds it) is owned by the Library ▸ Notes list/tree work stream,
and a wave-3 group was editing that code in parallel.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Saving an in-canvas edit and returning to the list shows that note's age as just-saved, without waiting for the refetch (`test_library_shell_note_save_then_back_refreshes_list_title_and_age` passes)
- [ ] #2 The stamp the row shows comes from the save that actually happened (the client's own save time or a modified time the save path returns), not from a baseline the save leaves untouched
<!-- AC:END -->
