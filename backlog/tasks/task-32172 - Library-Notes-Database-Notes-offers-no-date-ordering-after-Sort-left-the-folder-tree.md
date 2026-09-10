---
id: TASK-32172
title: >-
  Library Notes: Database Notes offers no date ordering after Sort left the
  folder tree
status: To Do
assignee: []
created_date: '2026-09-09 09:10'
updated_date: '2026-09-09 09:10'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - list
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the task review of task-32128. task-32128 correctly gated the flat list's
Sort control off the folder tree — the tree's order is the repository's
paging contract, and a Sort control there would lie about what it controls.
That left Database Notes with no date-ordering control anywhere once any
folder exists (which is always true in practice, because Agent_Lessons is
seeded): the repository's folder paging and the deep-link locator both hard-
code `ORDER BY title COLLATE NOCASE`, with no ORDER BY parameter through
`page_note_placements` or the locator to plumb Newest/Oldest through. Re-
offering Sort in the tree requires that plumbing to exist first, or the
control would be decorative again.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Choosing Newest or Oldest reorders folder placements by the note's
  modified date, consistently across paged results within a folder
- [ ] #2 The deep-link locator honours the same order parameter, not a fixed
  title order
- [ ] #3 A Sort control is re-added to the folder tree only once both above
  hold true
- [ ] #4 The behavior is pinned in a test
<!-- AC:END -->
