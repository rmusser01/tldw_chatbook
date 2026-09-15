---
id: TASK-32640
title: >-
  Library Notes: editor header answers "where does this note live?"
status: To Do
assignee: []
created_date: '2026-09-15 10:35'
labels:
  - library
  - notes
  - critique-4
  - idea
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 idea 1, ACCEPTED in task-32627. One row under the note title
reading which of the three worlds this note belongs to and, for a synced
note, its file path and when it was last written to disk.

This is the cheapest answer to a theme every critique has raised: the three
worlds (database-only, managed folder, edit-in-place) are not answerable from
inside a note, so a user cannot tell whether what they are typing will reach a
file. It is also the place a stale or refused write becomes visible without a
trip to Manage — related to task-32633, which owns the refusal signal itself.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 An open note shows, without tabbing or opening Info, which world it lives in, in the user's words rather than an internal name.
- [ ] #2 A synced note also shows its file path and when the file was last written; a database-only note shows neither and does not pretend to.
- [ ] #3 The row costs at most one line and degrades at 100x30 — it truncates the path, never the world.
- [ ] #4 The line is derived from live state, not from what the note looked like when it was opened.
<!-- AC:END -->
