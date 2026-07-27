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

Note the Feeds region *does* update, because it reads `scoped_source_rows()` off a different path — so the screen contradicts itself, with the new source visible on the left and absent from the table in the centre.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A source created through the form appears in `#sources-table` without leaving the section
- [ ] #2 The Feeds region and the sources table agree immediately after creation
- [ ] #3 The reload happens off the UI thread and does not block the form closing
- [ ] #4 A test creates a source and asserts the table contents, proven to fail against current code
- [ ] #5 Deleting a source refreshes the table the same way
<!-- AC:END -->
