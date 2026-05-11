---
id: TASK-35
title: Attach identity to Sync v2 restore rejections
status: Done
assignee:
  - Codex
created_date: '2026-05-10 19:30'
updated_date: '2026-05-10 19:31'
labels:
  - sync-v2
  - tldw-chatbook
  - restore
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore apply rejections currently expose the adapter error code but not the envelope or entity identity that failed. Add envelope identity to top-level restore_selection rejected entries so clients can present targeted recovery actions without parsing raw results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Top-level restore_selection rejected entries include the rejected envelope client_envelope_id.
- [x] #2 Rejected entries also include enough domain/entity context for client-side targeting.
- [x] #3 Raw per-envelope results remain available and existing applied/conflict behavior is unchanged.
- [x] #4 Focused restore tests and security/diff checks pass for the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update the existing restore rejection regression to require client_envelope_id plus domain/entity context in top-level rejected entries.
2. Run that focused test before production edits and confirm current behavior only returns status/error_code.
3. Update SyncRestoreService.restore_selection to validate pulled envelopes once, apply them, and build enriched top-level rejected entries by pairing each rejected result with its source envelope while preserving raw results.
4. Run focused restore tests, full Sync_Interop tests, Bandit on restore_service.py, and git diff --check before finalizing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified the strengthened rejection traceability test failed before the service change because top-level rejected entries only contained status and error_code. Updated restore_selection to pair validated envelopes with apply results and enrich only top-level rejected entries, preserving raw results.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated SyncRestoreService.restore_selection to attach source envelope identity to top-level rejected restore entries, including client_envelope_id, domain, entity_id, stable_key, and operation. The raw per-envelope results remain unchanged for compatibility. Strengthened the restore rejection regression test to assert both the enriched rejected list and unchanged raw results. Verified focused restore tests, full Sync_Interop suite, Bandit on restore_service.py, and git diff --check.
<!-- SECTION:FINAL_SUMMARY:END -->
