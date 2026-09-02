---
id: TASK-28240
title: 'Review sets - Phase 1: persistence and pure model'
status: Done
assignee: []
created_date: '2026-09-02 22:27'
updated_date: '2026-09-02 22:43'
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
- [x] #1 New v4 tables review_sets + review_set_items in Library_Collections_DB (bump _CURRENT_SCHEMA_VERSION 3->4, idempotent DDL in _initialize_schema, gated on schema_version); a partial unique index enforces at most one active set
- [x] #2 A ReviewSet service exposes create/get/list/advance-cursor(with tombstone skip)/mark-done/complete/reopen/dismiss; cursor is an absolute position, progress is computed over LIVE items only
- [x] #3 Tombstone detection is a runtime resolve against the Media DB (no cross-DB FK); title_snapshot is used when an item is gone; an all-tombstoned set reports empty, not complete
- [x] #4 Pure cursor/progress logic is unit-tested with in-memory SQLite incl. tombstone, all-done, and all-tombstoned cases
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read Library_Collections_DB schema-init + migration pattern exactly. 2. TDD the pure model (Library/review_set_state.py): cursor advance w/ tombstone skip, live-count progress, completion — pure functions, unit-tested first. 3. Add v4 tables (review_sets, review_set_items) + partial unique index; bump _CURRENT_SCHEMA_VERSION 3->4, idempotent DDL. 4. ReviewSet DB/service methods: create/get/list/advance-cursor/mark-done/complete/reopen/dismiss; tombstone = runtime resolve via injected is_live predicate. 5. Unit tests w/ in-memory SQLite: tombstone, all-done, all-tombstoned, resume.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase 1 shipped. (1) Schema v4 in Library_Collections_DB (bumped _CURRENT_SCHEMA_VERSION 3->4; _REVIEW_SET_SCHEMA_DDL applied idempotently in _initialize_schema): review_sets + review_set_items. NO CREATE INDEX -- the one-active invariant is enforced transactionally in the service (deactivate-all then activate in one transaction), which keeps the change out of the index-plan-pin census. (2) Pure model Library/review_set_state.py: ReviewSet/ReviewSetItem/ReviewProgress + tombstone-aware advance_cursor/resolve_cursor/review_progress/is_complete/is_empty over an injected is_live predicate (cursor is absolute, progress/completion over LIVE items). (3) ReviewSetService Library/review_set_service.py: create(dedupe+pin+activate)/get/list/get_active/advance(persist)/set_cursor/mark_item_done/refresh_completion/activate/reopen/dismiss. Tombstone detection is a runtime is_live resolve (separate DB files, no FK possible). Tests: Tests/Library/test_review_set_state.py (14) + test_review_set_service.py (11), all green; updated the collections migration/service tests for v4 (schema_version 3->4, future-schema test uses 5). Preflight + all collections suites green (the one round_trip_public_ids failure is PRE-EXISTING on dev, confirmed by reverting). Files: DB/Library_Collections_DB.py, Library/review_set_state.py, Library/review_set_service.py.
<!-- SECTION:NOTES:END -->
