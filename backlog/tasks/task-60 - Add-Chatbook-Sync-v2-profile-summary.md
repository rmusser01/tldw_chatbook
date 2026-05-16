---
id: TASK-60
title: Add Chatbook Sync v2 profile summary
status: Done
labels:
- sync
- sync-v2
- client-contract
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a Sync v2 profile summary contract that lets Chatbook report local-first/server-front-end status, dataset/device identity, pending outbox counts, conflict counts, cursor state, key availability, and last error without parsing low-level repository rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `SyncStateRepository` exposes a Sync v2 profile summary for configured and missing profile states.
- [x] #2 The summary includes profile mode, device/dataset identity, cursor state, outbox counts, identity map counts, conflict counts, last mirror report, and an overall status.
- [x] #3 `SyncScopeService` exposes the summary through the existing state-repository boundary.
- [x] #4 Focused Sync interop and API-client tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added `get_sync_v2_profile_summary` to `SyncStateRepository` so callers can read one stable status contract instead of inspecting sync profile rows, outbox entries, identity mappings, conflict reports, remote cursors, and mirror reports separately.

The summary status is intentionally presentation-oriented: missing profiles report `not_configured`, server-front-end profiles report `server_frontend`, local-only profiles report `local_only`, pending outbox work reports `pending`, and any last error or durable conflict report reports `attention_required`.

Added `SyncScopeService.get_sync_v2_profile_summary` as the public service seam for app callers that already receive a scope service.

Verification:
- `../../.venv/bin/python -m pytest Tests/Sync_Interop/test_sync_state_repository.py::test_sync_v2_profile_summary_aggregates_state_counts_and_status Tests/Sync_Interop/test_sync_state_repository.py::test_sync_v2_profile_summary_reports_missing_profile Tests/Sync_Interop/test_sync_scope_service.py::test_sync_scope_service_returns_sync_v2_profile_summary Tests/Sync_Interop/test_sync_scope_service.py::test_sync_scope_service_requires_repository_for_sync_v2_profile_summary -q` passed with 4 tests.
- `../../.venv/bin/python -m pytest Tests/Sync_Interop Tests/tldw_api/test_sync_client.py -q` passed with 143 tests.
- `../../.venv/bin/python -m compileall tldw_chatbook/Sync_Interop` passed.
- `git diff --check` passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -q -r tldw_chatbook/Sync_Interop/sync_state_repository.py tldw_chatbook/Sync_Interop/sync_scope_service.py -f json -o /tmp/bandit_chatbook_sync_profile_summary.json` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a Sync v2 profile summary contract for Chatbook local-first/server-front-end status. The repository now aggregates profile metadata, remote cursor state, outbox counts, identity map status counts, conflict counts, and the last mirror report; the scope service exposes that aggregate for app use.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests and verification recorded
- [x] #3 Documentation updated in task record
- [x] #4 Final summary added
- [x] #5 No known blockers
<!-- DOD:END -->
