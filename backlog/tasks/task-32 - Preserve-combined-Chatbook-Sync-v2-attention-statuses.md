---
id: TASK-32
title: Preserve combined Chatbook Sync v2 attention statuses
status: Done
assignee:
  - Codex
created_date: '2026-05-10 19:16'
updated_date: '2026-05-10 19:18'
labels:
  - sync-v2
  - tldw-chatbook
  - local-first
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A local-first Sync v2 pass can retain outbound work due to push rejection or conflict while also recording inbound apply conflicts. Preserve both attention signals in the Sync v2 profile status so the client can surface all user-visible sync issues from the pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When a sync pass has both push partial failures and apply conflicts, the profile last_error includes both statuses.
- [x] #2 Existing push-only and apply-only profile last_error behavior remains unchanged.
- [x] #3 A focused regression test covers the mixed push partial plus apply conflict case.
- [x] #4 Focused Sync_Interop tests and security/diff checks pass for the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused regression test that combines one retained outbox push failure with one inbound note apply conflict, then asserts the Sync v2 profile last_error includes both attention statuses.
2. Run the new test first and confirm the current service hides the apply conflict in that mixed case.
3. Replace the current push-status-or-apply-status selection in LocalFirstSyncService with a small helper that joins all present successful-sync attention statuses while preserving exact push-only and apply-only strings.
4. Run the focused test file, broader Sync_Interop tests, Bandit on touched production code, and git diff --check before finalizing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified the new mixed-status regression fails before the service change with last_error limited to push_partial_failure: stale_base. Implemented a helper that joins push partial and apply conflict messages with '; ' while preserving existing single-status strings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated LocalFirstSyncService so final successful-sync attention status preserves both outbound push partial failures and inbound apply conflicts. Added a focused regression test that combines a retained stale-base outbox entry with a remote encrypted note conflict, while existing push-only and apply-only tests continue to assert their previous exact strings. Verified the focused local-first file, full Sync_Interop directory, Bandit on the touched service, and git diff --check.
<!-- SECTION:FINAL_SUMMARY:END -->
