---
id: TASK-28240
title: 'Review sets - Phase 1: persistence and pure model'
status: To Do
assignee: []
created_date: '2026-09-02 22:27'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Foundation for Library review sets (design: backlog/docs/design-library-review-sets.md, approved via task-28024). A review set = pinned ordered local media ids + cursor + per-item done marks + completion state, persisted so review resumes across restarts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 New v4 tables review_sets + review_set_items in Library_Collections_DB (bump _CURRENT_SCHEMA_VERSION 3->4, idempotent DDL in _initialize_schema, gated on schema_version); a partial unique index enforces at most one active set
- [ ] #2 A ReviewSet service exposes create/get/list/advance-cursor(with tombstone skip)/mark-done/complete/reopen/dismiss; cursor is an absolute position, progress is computed over LIVE items only
- [ ] #3 Tombstone detection is a runtime resolve against the Media DB (no cross-DB FK); title_snapshot is used when an item is gone; an all-tombstoned set reports empty, not complete
- [ ] #4 Pure cursor/progress logic is unit-tested with in-memory SQLite incl. tombstone, all-done, and all-tombstoned cases
<!-- AC:END -->
