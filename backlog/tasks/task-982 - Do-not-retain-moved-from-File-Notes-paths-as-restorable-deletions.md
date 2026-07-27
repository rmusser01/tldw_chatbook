---
id: TASK-982
title: Do not retain moved-from File Notes paths as restorable deletions
status: Done
assignee:
  - '@codex'
created_date: '2026-07-27 18:50'
updated_date: '2026-07-27 19:17'
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
- [x] #1 A successful Chatbook move leaves only the destination active and searchable in the replica
- [x] #2 The moved-from source is absent from list_deleted() and the Recently deleted UI
- [x] #3 Actual Chatbook deletions and externally detected missing files remain tombstoned and restorable
- [x] #4 Disk and session reporting continue to record the operation as moved source to destination
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused failing replica, service, and mounted-workspace regressions showing a successful Chatbook move does not create a source tombstone while the destination remains active/searchable and the session action remains `moved`.
2. Add one atomic replica operation that publishes the destination row/FTS and removes only the active moved-from row/FTS in the same transaction, without touching revisions or genuine tombstones.
3. Route the service move through that single operation so a replica failure rolls back destination publication and preserves the source recovery copy.
4. Run the focused File Notes replica, service, and mounted-workspace tests plus targeted Ruff and diff checks, then self-review.

ADR required: no
ADR path: backlog/decisions/029-file-notes-disk-authority.md (existing)
Reason: This corrects move behavior to match the accepted distinction between moves and deletion tombstones; it changes no schema, authority rule, or cross-module interface.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented an atomic replica move that publishes destination bytes and removes only the active source projection without creating a source tombstone. Added session-local retry handling for transient replica failures, including safe source-path reuse and destination deletion so retained bytes keep the correct recovery identity. Updated the mounted workspace regression to verify the moved-from path never appears under Recently deleted while genuine deletions remain restorable. ADR: existing backlog/decisions/029-file-notes-disk-authority.md applies; no new ADR was required. Verification after rebasing on current dev: 48 focused replica/service tests passed, the mounted create/move/delete/protect/restore UI test passed, targeted Ruff passed, production modules compiled, and git diff --check passed. Modified File Notes replica, service, focused tests, mounted workspace test, and this task record.
<!-- SECTION:NOTES:END -->
