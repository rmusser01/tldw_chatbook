---
id: TASK-401.6
title: Add citation payload revocation retention and garbage collection
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 00:44'
updated_date: '2026-07-24 09:41'
labels:
  - rag
  - citations
  - privacy
  - retention
dependencies:
  - TASK-401.5
  - TASK-401.3
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-401
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enforce revocation-scoped deduplication, durable non-content tombstones, and reference-safe payload collection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Snapshot deduplication includes tenant or profile, authority, confidentiality policy, revocation scope, and governed exact-content identity.
- [ ] #2 Revocation and secure purge clear governed run, snapshot, and answer-attempt fields while retaining sealed run, attempt, evidence-reference, and payload identities plus only the permitted non-content tombstone.
- [ ] #3 Cache, import, and Sync replay cannot resurrect a tombstoned origin payload.
- [ ] #4 Garbage collection respects message and artifact owners, pending links, soft-delete retention, Sync tombstones, and policy retention windows.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing revocation-scoped dedupe, secure purge, tombstone, anti-resurrection, and identity-preservation tests.
2. Implement strict frozen lifecycle policy/tombstone contracts and atomic revoke transactions that clear every governed run/snapshot/attempt field without rewriting sealed trace structure.
3. Enforce tombstone checks on repository write, cache, imported, simulated Sync replay, and hydration seams so purged origins cannot resurrect.
4. Add reference-safe GC with message/artifact/pending-operation/Sync/policy-retention barriers and topologically safe graph deletion.
5. Run focused lifecycle/repository plus adjacent identity/persistence/migration/benchmark regressions, lint, diff, and both independent review gates.
6. Complete acceptance criteria and implementation notes only after approval.

ADR required: yes
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: This task implements ADR-024’s accepted revocation, tombstone, retention, anti-resurrection, and owner-safe collection policy using the fixed v25 schema.
<!-- SECTION:PLAN:END -->
