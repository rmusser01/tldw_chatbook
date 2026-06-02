---
id: TASK-70.4
title: Add Sync v2 conflict review and recovery plan
status: Done
labels:
- sync
- sync-v2
- conflicts
- recovery
priority: high
parent_task_id: TASK-70
documentation:
- Docs/superpowers/specs/2026-05-26-chatbook-sync-v2-completion-roadmap-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Sync v2 conflict and partial-failure states actionable enough for manual Notes and Chat sync users to inspect, recover, retry, or defer without losing local-first control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conflicts display domain, item label, cause, local summary, remote summary, and safe recovery options.
- [x] #2 Partial push failures remain durable and visible until resolved.
- [x] #3 Retry, keep-local, accept-remote, duplicate/fork, and defer-later states are modeled or explicitly marked unavailable.
- [x] #4 Tests cover conflict persistence, user-facing mapping, and no cursor advancement after failed apply.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing Sync v2 regressions for durable conflict-review records, user-facing conflict mapping/actions, partial failure visibility, and cursor preservation on failed apply.
2. Add a small conflict review/recovery model/service that converts raw push/apply conflicts and retained outbox failures into safe user-facing rows.
3. Extend SyncStateRepository with durable Sync v2 conflict record helpers scoped to server profile/dataset/domain without exposing plaintext.
4. Wire LocalFirstSyncService and ManualSyncControlService to persist and surface conflict/recovery records after manual sync results.
5. Update TASK-70.4 completion notes and run focused Sync_Interop/UI verification plus git diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `SyncV2ConflictReviewService` and `SyncV2ConflictReviewItem` to convert durable conflict rows and retained outbox failures into safe user-facing review rows.
- Added `sync_v2_conflict_reviews` schema v3 storage plus repository helpers for scoped conflict persistence, listing, summary counts, and profile cleanup.
- Wired local-first sync to persist push/apply conflict reviews, and wired manual sync results to include review rows even when a run fails after retaining outbox work.
- Updated Settings manual sync result rows to show the first conflict summary and explicit recovery action availability.
- Added regressions for durable conflict rows, retained push-failure visibility, manual sync conflict-review propagation, Settings display copy, and schema migration. Existing local-first failed-apply tests continue to verify cursors do not advance after failed apply.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Sync v2 conflict review and recovery state is now durable, user-facing, and surfaced through manual sync results without exposing encrypted payload plaintext.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
