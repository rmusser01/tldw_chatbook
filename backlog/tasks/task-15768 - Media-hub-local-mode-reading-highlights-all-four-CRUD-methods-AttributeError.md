---
id: TASK-15768
title: 'Media hub local mode: reading-highlights CRUD all AttributeError against the real service'
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - bug
  - media
priority: high
---

## Description

Found and confirmed during task-15467 (input-latency burn-down), explicitly
left unfixed as out of scope for that task's off-loop-threading AC:
`MediaReadingScopeService` calls `list_reading_highlights`,
`create_reading_highlight`, `update_reading_highlight`, and
`delete_reading_highlight` in local mode, but `LocalMediaReadingService` only
implements the unprefixed `list_highlights`/`create_highlight`/
`update_highlight`/`delete_highlight`. All four calls `AttributeError`
against a real local service — confirmed directly
(`getattr(service, method_name)` raises for all four).

Every local-mode media-item click already hits this: loading highlights is
swallowed by `_load_media_item_detail`'s broad `except Exception` and
silently presents zero highlights, and every local-mode highlight
create/update/delete action hits the identical `AttributeError`. The
Library screen's own, separate call sites already use the correct unprefixed
names and work fine — this is specifically the Media hub's scope-service
bridge, which never matched `LocalMediaReadingService`'s actual method
names.

## Acceptance Criteria

- [ ] `MediaReadingScopeService`'s local-mode calls for list/create/update/delete
      reading highlights reach `LocalMediaReadingService`'s real methods (no
      `AttributeError`), by renaming one side to match the other
- [ ] A local-mode media item with existing highlights shows them in the
      Media hub item detail (regression test against a real
      `LocalMediaReadingService`, not a mock that hides the name mismatch)
- [ ] Creating, updating, and deleting a highlight from the Media hub in
      local mode works end-to-end (tests)
- [ ] The broad `except Exception` around the detail load in
      `_load_media_item_detail` is narrowed or logged so a future
      naming/contract drift is visible instead of silently presenting empty
      state
