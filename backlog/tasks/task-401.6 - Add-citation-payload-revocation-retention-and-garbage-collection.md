---
id: TASK-401.6
title: Add citation payload revocation retention and garbage collection
status: To Do
assignee: []
created_date: '2026-07-24 00:44'
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
