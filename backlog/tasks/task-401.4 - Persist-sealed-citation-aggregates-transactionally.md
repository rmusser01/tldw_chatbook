---
id: TASK-401.4
title: Persist sealed citation aggregates transactionally
status: To Do
assignee: []
created_date: '2026-07-24 00:43'
labels:
  - rag
  - citations
  - database
  - migration
dependencies:
  - TASK-401.2
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
Add a versioned SQLite repository and dormant write seam that can atomically store a final message and its complete sealed provenance aggregate before live producers cut over from compatibility storage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The schema stores identity context, trace summaries, runs, governed snapshots, answer-attempt payloads, evidence references, message owners, observations, purge tombstones, stable artifact leases and operation identities, and bounded legacy migration journals.
- [ ] #2 One transaction writes or reuses the final message and every sealed aggregate row; incomplete builders write no canonical rows.
- [ ] #3 Prompt and attempt metadata are bounded, while governed source and content fields remain outside immutable trace JSON.
- [ ] #4 Citation tables are excluded from FTS, Library indexing, and RAG ingestion, and schema migration and rollback-safety tests pass.
- [ ] #5 A stable local profile and authority context plus injected fingerprint key are available for writes, and a recovery switch can disable repository writes while preserving authorized aggregate reads.
- [ ] #6 ChatPersistenceService receives citation persistence explicitly and fails before any write when a sealed citation is supplied without an available repository, policy, identity, or key.
<!-- AC:END -->
