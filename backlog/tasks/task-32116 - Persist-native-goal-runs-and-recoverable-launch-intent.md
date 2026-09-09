---
id: TASK-32116
title: Persist native goal runs and recoverable launch intent
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 04:16'
updated_date: '2026-09-09 04:21'
labels:
  - agents
  - console
dependencies: []
references:
  - backlog/decisions/141-native-console-goal-runs.md
documentation:
  - Docs/superpowers/plans/2026-09-08-gnhf-inspired-goal-runs.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users need a durable objective and a single recoverable conversation when starting or reopening autonomous work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Identical repeated Start delivery resolves to the same goal, conversation identity and allowance; conflicting payloads are refused.
- [ ] #2 Interrupted conversation and workspace provisioning can be reconciled without duplicate history or premature model/tool execution.
- [ ] #3 Goal requests reject invalid types, oversized UTF-8 payloads and unbounded limits; stored goal policy and scope remain immutable.
- [ ] #4 Reopening real SQLite preserves goal state and the existing automatic-work history; migration retains legacy fleet semantics.
- [ ] #5 Goal and chain creation roll back together on failure; concurrent writers respect revision and launch identity constraints.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/141-native-console-goal-runs.md (Accepted)
Reason: adds durable goal records and cross-store launch provisioning under the reviewed runtime boundary.
1. Add meaningful failing tests for strict request bounds and immutable launch identity, real-SQLite rollback/migration, and recoverable conversation provisioning.
2. Implement task 1 of Docs/superpowers/plans/2026-09-08-gnhf-inspired-goal-runs.md with one AgentRunsDB transaction for goal and chain, and idempotent chat/workspace provisioning.
3. Run targeted tests and migration regressions, lint/format changed files, self-review and obtain independent task review.
4. Record exact validation and update the task; do not claim the later execution or UI slices are implemented.
<!-- SECTION:PLAN:END -->
