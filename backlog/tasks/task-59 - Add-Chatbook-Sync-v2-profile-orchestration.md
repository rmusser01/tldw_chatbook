---
id: TASK-59
title: Add Chatbook Sync v2 profile orchestration
status: Done
labels:
- sync
- orchestration
- chatbook
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next Chatbook sync-engine slice after the already-present Sync v2 API client methods. Add explicit sync profile modes, dataset/device/cursor state, outbox/inbox/conflict summaries, and server_sync_service orchestration for capabilities, register, enroll, push, pull, and readiness without changing local-only behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chatbook accepts the canonical `local_first_sync` Sync v2 profile mode from the server sync-engine plan.
- [x] #2 Existing `local_first` profile records continue to work as a compatibility alias.
- [x] #3 Local-first sync and key recovery services recognize both canonical and legacy profile-mode spellings.
- [x] #4 Focused Sync_Interop and tldw_api sync client tests pass, with verification recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verified that the broad Chatbook Sync v2 API client/profile/encryption/local-first/restore substrate was already present on current `origin/dev`: `Tests/tldw_api/test_sync_client.py` passed with 9 tests and `Tests/Sync_Interop` passed with 122 tests before this patch.

Aligned profile mode handling with the server sync-engine plan by adding canonical `local_first_sync` support while preserving existing `local_first` records as a compatibility alias. The scope router now returns the requested canonical mode for `prepare_sync_v2_profile_mode`, and local-first sync plus key-recovery services accept both spellings.

Verification: `../../.venv/bin/python -m pytest Tests/Sync_Interop/test_sync_state.py Tests/Sync_Interop/test_sync_state_repository.py Tests/Sync_Interop/test_sync_scope_service.py Tests/Sync_Interop/test_local_first_sync_service.py Tests/Sync_Interop/test_key_recovery_service.py -q` passed with 66 tests.
Verification: `../../.venv/bin/python -m pytest Tests/Sync_Interop Tests/tldw_api/test_sync_client.py -q` passed with 136 tests.
Verification: `git diff --check` passed.
Review follow-up: addressed PR #320 Qodo/Gemini feedback by giving `is_local_first_sync_profile_mode` a Google-style public docstring, passing normalized profile modes from `SyncScopeService` into `ServerSyncService.run_v2_dry_run`, defaulting direct dry-runs to canonical `local_first_sync`, preserving explicit legacy `local_first` requests, and preserving stored canonical mode during key recovery. Verification: `../../.venv/bin/python -m pytest Tests/Sync_Interop/test_sync_state.py Tests/Sync_Interop/test_sync_state_repository.py Tests/Sync_Interop/test_sync_scope_service.py Tests/Sync_Interop/test_server_sync_service.py Tests/Sync_Interop/test_local_first_sync_service.py Tests/Sync_Interop/test_key_recovery_service.py -q` passed with 92 tests; `../../.venv/bin/python -m pytest Tests/Sync_Interop Tests/tldw_api/test_sync_client.py -q` passed with 137 tests; `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added canonical `local_first_sync` profile-mode support for Chatbook Sync v2 while keeping `local_first` compatibility. Profile state, repository persistence, scope preparation, local-first sync execution, and key-recovery setup now recognize the canonical mode used by the sync-engine plan.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant or not required
- [x] #4 Final summary added
- [x] #5 Known skips or blockers documented
<!-- DOD:END -->
