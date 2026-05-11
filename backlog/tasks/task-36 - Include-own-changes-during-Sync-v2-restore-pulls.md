---
id: TASK-36
title: Include own changes during Sync v2 restore pulls
status: Done
assignee:
  - Codex
created_date: '2026-05-10 19:34'
updated_date: '2026-05-10 19:36'
labels:
  - sync-v2
  - tldw-chatbook
  - restore
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Selective Chatbook restore rebuilds local state from server-held Sync v2 envelopes. Unlike incremental sync, restore pulls should include envelopes authored by the same device ID so a reinstall or repaired client can recover its own previously-synced content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 restore_selection requests pull_v2_envelopes with include_own_changes enabled.
- [x] #2 Incremental local-first sync continues to request pulls with include_own_changes disabled.
- [x] #3 A focused restore regression test covers the restore pull flag behavior.
- [x] #4 Focused Sync_Interop tests and security/diff checks pass for the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update the focused restore pull regression so restore_selection expects include_own_changes=True when calling pull_v2_envelopes.
2. Run that restore test before production edits and confirm the current restore service still sends False.
3. Change SyncRestoreService.restore_selection to request include_own_changes=True, leaving LocalFirstSyncService incremental pull behavior unchanged.
4. Run focused restore tests, local-first test coverage for the unchanged incremental flag, full Sync_Interop tests, Bandit on restore_service.py, and git diff --check before finalizing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified the focused restore pull test failed before the service change because restore_selection sent include_own_changes=False. Changed only SyncRestoreService.restore_selection to send True; test_local_first_sync_service.py still passes and verifies incremental local-first pulls remain False.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Changed selective Sync v2 restore pulls to request include_own_changes=True so restore can rebuild a device from all selected server-held envelopes, including prior envelopes authored by the same device ID. Updated restore pull expectations for local-key and recovery-key restore paths. Verified focused restore tests, local-first sync tests for unchanged incremental behavior, full Sync_Interop suite, Bandit on restore_service.py, and git diff --check.
<!-- SECTION:FINAL_SUMMARY:END -->
