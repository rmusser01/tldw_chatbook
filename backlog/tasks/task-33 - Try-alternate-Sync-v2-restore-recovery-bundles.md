---
id: TASK-33
title: Try alternate Sync v2 restore recovery bundles
status: Done
assignee:
  - Codex
created_date: '2026-05-10 19:21'
updated_date: '2026-05-10 19:23'
labels:
  - sync-v2
  - tldw-chatbook
  - restore
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Chatbook restore can receive multiple Sync v2 key recovery records for a dataset, such as rotated keys or records from different devices. Restore should try available candidate bundles until one unwraps successfully so a stale or incompatible first record does not block a valid later recovery record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Restore key resolution tries later recovery bundle candidates when an earlier candidate fails to unwrap.
- [x] #2 A successful later recovery bundle allows restore_selection to pull and apply selected envelopes.
- [x] #3 Existing explicit key_record_id filtering and recovery failure behavior remain intact.
- [x] #4 Focused restore tests and security/diff checks pass for the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused restore_service regression test with two recovery records: the first unwraps with a different secret and the second unwraps with the provided secret. Assert restore_selection pulls and applies after the later bundle succeeds.
2. Run that test before production edits and confirm current restore key resolution fails on the first bad candidate.
3. Update SyncRestoreService._resolve_dataset_key to iterate candidate recovery records, returning the first successful unwrap while preserving key_record_id filtering and the existing failure message when none work.
4. Run the focused restore tests, full Sync_Interop suite, Bandit on restore_service.py, and git diff --check before finalizing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified the new restore regression failed before the service change because _resolve_dataset_key stopped after the first recovery bundle. Updated resolution to try each filtered candidate and preserve the existing Failed to recover dataset key error when none unlock.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated SyncRestoreService recovery key resolution to iterate available recovery bundle candidates after key_record_id filtering and return the first bundle that unwraps with the provided recovery secret. Added a regression test where an old bundle fails and a later current bundle unlocks the dataset, allowing restore_selection to pull and apply envelopes. Existing recovery failure behavior remains covered by the restore tests. Verified the focused restore file, full Sync_Interop suite, Bandit on restore_service.py, and git diff --check.
<!-- SECTION:FINAL_SUMMARY:END -->
