---
id: TASK-982
title: Do not retain moved-from File Notes paths as restorable deletions
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-27 18:50'
updated_date: '2026-07-27 18:52'
labels:
  - bug
  - notes
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Chatbook-initiated File Notes move currently tombstones the source path after publishing the destination. That makes the old path appear under Recently deleted and allows Restore to recreate a stale duplicate. Because Chatbook knows this operation is a move, the source replica projection should be removed without recording a deletion; genuine Chatbook deletes and externally detected missing files must remain recoverable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A successful Chatbook move leaves only the destination active and searchable in the replica
- [ ] #2 The moved-from source is absent from list_deleted() and the Recently deleted UI
- [ ] #3 Actual Chatbook deletions and externally detected missing files remain tombstoned and restorable
- [ ] #4 Disk and session reporting continue to record the operation as moved source to destination
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused failing replica, service, and mounted-workspace regressions showing a successful Chatbook move does not create a source tombstone while the destination remains active/searchable and the session action remains `moved`.
2. Add one transactional replica operation that removes a moved-from current row and FTS projection without touching revisions or recording a tombstone.
3. Call that operation only after the moved destination has been successfully published to the replica; preserve the existing warning and recovery behavior when replica refresh fails.
4. Run the focused File Notes replica, service, and mounted-workspace tests plus targeted Ruff and diff checks, then self-review.

ADR required: no
ADR path: backlog/decisions/029-file-notes-disk-authority.md (existing)
Reason: This corrects move behavior to match the accepted distinction between moves and deletion tombstones; it changes no schema, authority rule, or cross-module interface.
<!-- SECTION:PLAN:END -->
