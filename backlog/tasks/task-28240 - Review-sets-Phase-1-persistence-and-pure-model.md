---
id: TASK-28240
title: 'Review sets - Phase 1: persistence and pure model'
status: Done
assignee: []
created_date: '2026-09-02 22:27'
updated_date: '2026-09-03 00:38'
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
Phase 1 shipped + hardened after Qodo review on PR #2323. Schema v4 in Library_Collections_DB: review_sets + review_set_items + a partial UNIQUE index review_sets_one_active (active=1 AND deleted_at IS NULL) that enforces the one-active invariant in the schema AND matches get_active_review_set's WHERE clause (census row added, pre-convention). Pure model Library/review_set_state.py: tombstone-aware advance/resolve/progress/complete over an injected is_live, computed on a SINGLE live snapshot (no double-evaluation crash). Service Library/review_set_service.py: create (validates origin allow-list + non-empty name + non-empty items) / get / list(limit) / get_active / advance (atomic read+write) / set_cursor / mark_item_done / refresh_completion (atomic) / activate (guards a missing/dismissed id so it never clears the active set) / reopen / dismiss. Reads share one snapshot via _read_review_set. Tests in :memory: (Tests/Library/test_review_set_state.py + test_review_set_service.py). Files: DB/Library_Collections_DB.py, Library/review_set_state.py, Library/review_set_service.py, scripts/index_plan_pin_census.tsv.
<!-- SECTION:NOTES:END -->
