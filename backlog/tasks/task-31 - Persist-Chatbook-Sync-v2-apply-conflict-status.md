---
id: TASK-31
title: Persist Chatbook Sync v2 apply conflict status
status: Done
assignee:
  - Codex
created_date: '2026-05-10 19:11'
updated_date: '2026-05-10 19:15'
labels:
  - sync-v2
  - tldw-chatbook
  - local-first
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When tldw_chatbook runs local-first Sync v2, remote envelopes can apply as recorded local conflicts while the sync profile still reports a clean last_error. Persist a concise profile status for apply conflicts so the client can surface that attention is needed while still allowing cursor advancement after the conflict is durably recorded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A pulled envelope that applies as a local conflict stores an apply_conflict last_error on the Sync v2 profile.
- [x] #2 The sync cursor still advances after a local apply conflict is recorded, so already-recorded conflicts are not replayed forever.
- [x] #3 A focused regression test covers the profile status and cursor behavior for local apply conflicts.
- [x] #4 Focused Sync_Interop tests and security/diff checks pass for the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused regression test in Tests/Sync_Interop/test_local_first_sync_service.py that pulls an envelope which conflicts with a local note, then asserts both profile last_error and cursor advancement.
2. Run the new test first to confirm the current behavior fails.
3. Update tldw_chatbook/Sync_Interop/local_first_sync_service.py so successful sync final status preserves recorded apply conflicts while keeping cursor persistence behavior unchanged.
4. Re-run the focused Sync_Interop tests, Bandit on touched code, and git diff --check before closing the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a local-first apply conflict regression using the real notes adapter divergence path. The test is run through pytest with parakeet_mlx and lightning_whisper_mlx pre-marked unavailable because this macOS host aborts during optional STT auto-detection before Sync_Interop tests collect.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated LocalFirstSyncService to compute recorded apply conflicts before persisting the final profile state. Successful syncs now keep cursor advancement behavior but store apply_conflict: <type> when pulled envelopes were durably recorded as local conflicts, preserving existing push_partial_failure precedence. Added a focused regression test for encrypted note content divergence and verified the full Sync_Interop test directory plus Bandit and git diff --check.
<!-- SECTION:FINAL_SUMMARY:END -->
