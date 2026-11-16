---
id: TASK-41213
title: Keep fixed-target cleanup visible to the SQL census
status: Done
assignee:
  - '@codex'
created_date: '2026-11-15 21:51'
updated_date: '2026-11-16 00:05'
labels:
  - ci
  - personal-context
dependencies: []
references:
  - >-
    https://github.com/rmusser01/tldw_chatbook/actions/runs/43750308318/job/132962026035
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The current dev semantic mutation inventory rejects fixed-target cleanup loops added by Personal Context recovery. The first is damaged-record removal; after it is corrected the same guard exposes restored sync-identity cleanup. Preserve both transaction behaviors while making their six fixed SQL targets visible to the existing guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The live SQL mutation inventory and immutable scan-cache checks pass without adding a census exception or weakening the guard.
- [x] #2 Damaged-record removal and interrupted profile re-encryption retain their tested behavior.
- [x] #3 Reviewed restored sync-identity clearing still removes the intended state and refuses changed review counts before changing data.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md (existing SQL census policy)
Reason: preserve existing cleanup transactions while replacing fixed-target dynamic SQL; no storage, security or ownership boundary changes.
1. Trace the two mutation-inventory failures from CI run 43750308318 to TASK-40140; retain the subsequent local scan exposing TASK-40139 SyncStateRepository.clear_restored_identity as a second fixed-target loop.
2. Replace only the two three-table DELETE loops with literal parameterized SQL where values exist, preserving order and transaction scope. Keep the census guard and exceptions unchanged.
3. Run both inventory checks, the damaged-record/interrupted-reencryption cases and restored sync-identity behavioral coverage.
4. Independently review and record evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced two fixed three-table DELETE loops with literal SQL in Personal_Context/repository_keys.py and Sync_Interop/sync_state_repository.py. Table order, predicates, bound values, surrounding transactions, outbox shredding and final object deletion are unchanged. The SQL census and its exceptions are unchanged.

Validation: both complete SQL inventory/cache checks pass; ten damaged-record and interrupted-reencryption cases pass. A real in-memory SQLite smoke check seeded all three restored-identity stores, verified stale review counts refuse without changing data, and verified reviewed clearing empties all three stores. Two broader restart tests fail before reaching the changed operation on this Windows host because their fixture writes an unescaped Windows path into TOML; that limitation is retained in the evidence rather than treated as a product failure. Scoped Ruff and independent review pass.

ADR required: no; existing ADR-337 census policy applies. No storage, ownership or security boundary changes.
<!-- SECTION:NOTES:END -->
