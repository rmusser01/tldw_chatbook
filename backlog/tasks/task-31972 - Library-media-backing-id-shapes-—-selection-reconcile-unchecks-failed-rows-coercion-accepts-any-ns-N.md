---
id: TASK-31972
title: >-
  Library media: backing-id shapes — selection reconcile unchecks failed rows;
  coercion accepts any ns:N
status: To Do
assignee: []
created_date: '2026-09-07 20:27'
labels:
  - library
  - media
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two id-shape leftovers from PRs M and O. (1) The bulk-delete SELECTION reconcile in `library_screen.py` (~24239) feeds backing ids from `_source_record_id` into `_library_media_row_selection.reconcile()`, whose retained set holds canvas row ids (`local:media:<n>`), so on a partial failure the intersection empties and the rows the user must retry are unchecked — defeating task-3020 AC3. Task-31943 fixed the same mismatch at the prune site only. (2) The one coercion helper `library_media_int_backing_id` (task-31962) is looser than the spelling it replaced: `foo:12` now coerces to 12 where the old helper required the `local:media:` prefix; reachable only through `_review_cursor_for_display` with a `server:media:`-shaped id, which does not exist today.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a partial bulk-delete failure the failed rows stay selected (pinned over a real DB)
- [ ] #2 The selection reconcile compares one id spelling, shared with the prune site
- [ ] #3 The coercion helper refuses ids without the `local:media:` prefix, pinned
<!-- AC:END -->
