---
id: TASK-25
title: Register Chatbook Sync v2 runtime-policy actions
status: Done
assignee: []
created_date: '2026-05-10 16:26'
updated_date: '2026-05-10 16:33'
labels:
  - sync
  - client
  - runtime-policy
dependencies: []
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime policy registry includes all Sync v2 action IDs enforced by ServerSyncService
- [x] #2 Sync v2 actions require server source and deny in local mode through normal policy decisions
- [x] #3 ServerSyncService policy-gated Sync v2 methods are covered with the real ServicePolicyEnforcer rather than only mocks
- [x] #4 Focused runtime-policy and Sync_Interop tests Bandit and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing runtime-policy tests for all ServerSyncService Sync v2 action IDs.
2. Add failing ServerSyncService tests using ServicePolicyEnforcer to prove registered actions are allowed in authenticated server mode and denied in local mode before dispatch.
3. Register the Sync v2 resources/actions in runtime_policy.registry without changing existing legacy sync.changes actions.
4. Run focused runtime-policy and Sync_Interop tests plus Bandit and diff checks.
5. Update Backlog and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added Sync v2 runtime-policy action registration under sync_transport and real ServicePolicyEnforcer coverage for ServerSyncService. Verified focused Sync v2 policy/service tests: 10 passed; full ServerSyncService module: 12 passed; production Bandit on tldw_chatbook/runtime_policy/registry.py: 0 findings; git diff --check: clean. Full audited registry snapshot remains blocked by pre-existing kanban_boards_tasks/server_skills separated-source expectation drift.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Registered the Sync v2 runtime-policy actions used by ServerSyncService under the existing remote-only sync_transport capability, preserving legacy sync.changes actions. Added tests proving every Sync v2 action is known to the policy engine, denied in local mode with wrong_source, allowed in authenticated server mode, and enforced by ServerSyncService before dispatch.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run on touched production code
- [x] #4 Implementation notes added
<!-- DOD:END -->
