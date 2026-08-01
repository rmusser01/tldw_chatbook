---
id: TASK-1602
title: Define production local apply-store ownership for manual Sync v2
status: To Do
assignee: []
created_date: '2026-07-31 14:03'
updated_date: '2026-07-31 14:06'
labels:
  - architecture
  - sync
  - data-ownership
dependencies: []
references:
  - >-
    backlog/tasks/task-1601 -
    Bind-application-Sync-graph-to-the-runtime-server-context-provider.md
  - backlog/decisions/033-application-session-state-ownership.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the verified gap that leaves the visible manual Sync v2 workflow permanently blocked because TldwCli has no production local apply store. Define truthful domain data owners, mutation, tombstone, conflict, transaction, and privacy boundaries before enabling application composition.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The required local mutation contracts for every domain exposed by manual Sync are verified against current production repositories and adapters.
- [ ] #2 A canonical ADR defines local data ownership, transaction, deletion/tombstone, conflict, and failure invariants without using the in-memory verification store in production.
- [ ] #3 The design preserves memory-only dataset-key handling and prevents private payloads or keys from entering diagnostics or unrelated persistence.
- [ ] #4 The implementation boundary is decomposed into atomic, dependency-ordered Backlog tasks that can each be completed and verified in one pull request.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 All acceptance criteria are checked.
- [ ] #2 The canonical ADR and design documents are complete, internally consistent, and linked.
- [ ] #3 Resulting implementation tasks are atomic, dependency-ordered, and contain testable outcomes.
- [ ] #4 Self-review and relevant documentation checks pass before the task is marked Done.
<!-- DOD:END -->
