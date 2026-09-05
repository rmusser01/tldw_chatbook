---
id: TASK-31757
title: Preserve persisted parent identity in live compaction lineage snapshots
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 22:06'
updated_date: '2026-09-05 22:15'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Manual summary after real accepted sends reads the stale cached parent_message_id field, which remains None on checkpoint-persisted messages. The durable ordered-parent-chain guard correctly rejects these snapshots.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Live compaction snapshots use durable IDs consistently without weakening lineage guards
- [ ] #2 Distinct native and persisted IDs and a real edited branch are covered by regression evidence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; routine correction within the existing durable lineage contract. 1. Reproduce the real-send rewind failure and assert the durable parent chain. 2. Reuse ConsoleChatStore.durable_parent_for_message, already used by accepted-turn persistence, rather than reading the stale cached parent field. 3. Run complete rewind and parent-persistence checks with unchanged ownership and content guards; record fixture modernization in TASK-31753. Initial diagnosis of native-versus-persisted IDs was refined by the exact snapshot probe: every cached parent was None.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reused the existing durable_parent_for_message resolver in one snapshot field, matching the accepted-turn persistence owner. The real edited-send regression failed on the ordered-parent-chain assertion before the correction; summary persistence and restored no-leak behavior now pass in the 203-test rewind/settings selection. The parent fault injection follows the authoritative native tree. Final whole summary-file verification is pending. ADR not required: routine repair with unchanged guards.
<!-- SECTION:NOTES:END -->
