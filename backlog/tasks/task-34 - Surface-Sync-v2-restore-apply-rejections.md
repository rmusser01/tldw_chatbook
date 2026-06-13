---
id: TASK-34
title: Surface Sync v2 restore apply rejections
status: Done
assignee:
  - Codex
created_date: '2026-05-10 19:26'
updated_date: '2026-05-10 19:27'
labels:
  - sync-v2
  - tldw-chatbook
  - restore
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Selective Chatbook restore can receive envelopes that domain adapters reject, for example workspace link envelopes whose source reference is missing. Restore should expose those rejected apply results in the structured restore_selection response instead of requiring callers to inspect raw results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 restore_selection includes a top-level rejected list for adapter-rejected envelopes.
- [x] #2 The rejected list preserves the adapter error_code so the client can present actionable restore feedback.
- [x] #3 Applied and conflict restore response behavior remains unchanged.
- [x] #4 Focused restore tests and security/diff checks pass for the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused restore_service regression test with a workspace link envelope that the WorkspacesSyncAdapter rejects, then assert restore_selection exposes that rejection in a top-level rejected list with the adapter error_code.
2. Run the new test before production edits and confirm the current restore response only exposes the rejection inside raw results.
3. Update SyncRestoreService.restore_selection to collect rejected apply results into a structured rejected list while preserving applied, conflicts, next_cursor, has_more, and raw results behavior.
4. Run the focused restore tests, full Sync_Interop suite, Bandit on restore_service.py, and git diff --check before finalizing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified the regression failed before the service change with KeyError: 'rejected'. Added top-level rejected apply results while leaving raw results, applied counts, and conflict reporting intact.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated SyncRestoreService.restore_selection to include a top-level rejected list containing adapter-rejected apply results. Added a regression test using the real workspace adapter missing-source rejection path, preserving the adapter error_code for client-facing restore feedback. Verified the focused restore tests, full Sync_Interop suite, Bandit on restore_service.py, and git diff --check.
<!-- SECTION:FINAL_SUMMARY:END -->
