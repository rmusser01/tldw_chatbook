---
id: TASK-31757
title: Preserve persisted parent identity in live compaction lineage snapshots
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 22:06'
updated_date: '2026-09-05 22:37'
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
One snapshot field now uses the existing durable_parent_for_message resolver instead of an unset cached parent. The real edited-send parent-chain regression failed before the change; the full rewind integration and all 73 summary tests pass after rebasing. The parent fault case mutates the authoritative native tree. Pre-rebase combined 78-test run also passed; its existing aggregate descriptor warning remains tracked. No new ADR; ownership guards are unchanged.
<!-- SECTION:NOTES:END -->
