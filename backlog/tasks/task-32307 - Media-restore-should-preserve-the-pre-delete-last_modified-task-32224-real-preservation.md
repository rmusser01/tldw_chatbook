---
id: TASK-32307
title: >-
  Media restore should preserve the pre-delete last_modified (task-32224 real
  preservation)
status: To Do
assignee: []
created_date: '2026-09-11 00:55'
labels:
  - library
  - media
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-32224 closed on its docs clause: the guide now says restore re-dates the item, because `mark_as_trash` stamps last_modified too (measured on PR #2583), so a restore-only change cannot stop the re-dating. Real preservation needs the pre-delete stamp carried through the delete receipt and written back through the media DB's shared sync contract in `DB/Client_Media_DB_v2.py`. Optional: the reviewer judged no follow-up required.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Undo of a bulk delete restores each item's pre-delete last_modified and its list position
<!-- AC:END -->
