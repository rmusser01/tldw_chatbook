---
id: TASK-1040
title: >-
  Creating a source leaves the sources table showing the old list
status: To Do
assignee: []
created_date: '2026-07-28 02:00'
labels:
  - watchlists
  - bug
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`_create_source` never calls `_load_sources()`, so after a source is created successfully the `#sources-table` still shows the list from before. The new source only appears after leaving the Sources section and coming back.

Confirmed live during the task-1035 fix.

A user creating their first source is told nothing happened: the form closes and the table is still empty. The obvious next move is to create it again, which either duplicates the source or hits a uniqueness error for something they cannot see.

The Feeds region *does* update, because it reads `scoped_source_rows()` off a different path — so the screen contradicts itself.

**The tree's counts are stale in the same way**, verified live immediately after creating a source through the fixed form:

```
│ All sources  0           ││  Feeds in All sources (1)
│ Unassigned  0            ││  AI News RSS  (rss)
```

The rail says zero, the centre says one, and they are describing the same thing. So this is not only the sources table: **creating a source refreshes only the view that happens to read it directly**, and every count and list derived from it stays behind. Fix them together, or the next one found will be filed separately again.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A source created through the form appears in `#sources-table` without leaving the section
- [ ] #2 The Feeds region and the sources table agree immediately after creation
- [ ] #3 The reload happens off the UI thread and does not block the form closing
- [ ] #4 A test creates a source and asserts the table contents, proven to fail against current code
- [ ] #5 Deleting a source refreshes the table, the tree counts and Feeds the same way
- [ ] #6 The tree's `All sources` and `Unassigned` counts match the Feeds heading immediately after a create or delete
- [ ] #7 A test asserts the tree count and the Feeds heading agree after creating a source, proven to fail against current code
<!-- AC:END -->
