---
id: TASK-32224
title: >-
  Library Media: Undo of a bulk delete re-dates the restored items to now and
  reorders the list (guide: 'restore never rewrites')
status: Done
assignee: []
created_date: '2026-09-10 14:55'
updated_date: '2026-09-10 17:40'
labels:
  - library
  - media
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After Undo the restored items read 'now' and jump to the top of Newest; the guide says restore never rewrites the item. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 22.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Restore preserves the item's modified time and position, or the guide states the real behaviour
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Test: seed two items with old last_modified, delete, Undo, assert stored time unchanged.
2. Trace the restore write; if it is a dedicated statement, drop last_modified from its SET list; if a shared helper stamps it for all writers, correct the guide sentence instead.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Took AC#1's SECOND clause (the guide states the real behaviour), because the measurement contradicted the assumed root cause.

Measured against a real MediaDatabase before choosing: seeded last_modified 16:57:59.638, after mark_as_trash 16:58:00.746, after restore_from_trash 16:58:01.853. The DELETE already stamps last_modified with the current time -- `mark_as_trash` and `restore_from_trash` are both optimistic-locking writes that bump version, stamp last_modified and log a sync event. So dropping the stamp from the restore alone would have left the item carrying the DELETE's 'now' and still sorting to the top of Newest: it would not have satisfied AC#1's first clause at all. Genuinely preserving the pre-delete time would mean capturing it per id before the delete, carrying it through the receipt, and writing it back through the media DB's shared sync contract -- a new cross-cutting write path for a P3 ordering nicety, on a column that is that row's sync clock rather than an edit marker.

The guide's 'restore never rewrites the item' now reads 'Restore brings the item back and marks it changed now, so it returns at the top of a Newest sort', and says the same of Undo on a delete receipt; 'title, content and analysis untouched' keeps the part that was true. A test pins the measured behaviour through a real MediaDatabase AND the guide sentence, so neither can drift alone.

Live at 235x52: selected two items, confirmed the delete ('✓ deleted · 2 items · in Trash'), pressed Undo -- both came back reading 'document · now' / 'audio · now' where they had read '2h', at the top of the Newest list.

Files: Docs/User_Guide/library/media-and-conversations.md, Tests/UI/test_library_crit9_media_reader.py.
<!-- SECTION:NOTES:END -->
