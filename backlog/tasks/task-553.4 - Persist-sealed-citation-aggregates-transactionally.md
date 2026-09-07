---
id: TASK-553.4
title: Persist sealed citation aggregates transactionally
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 00:43'
updated_date: '2026-07-24 08:37'
labels:
  - rag
  - citations
  - database
  - migration
dependencies:
  - TASK-553.2
  - TASK-553.3
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-553
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a versioned SQLite repository and dormant write seam that can atomically store a final message and its complete sealed provenance aggregate before live producers cut over from compatibility storage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The schema stores identity context, trace summaries, runs, governed snapshots, answer-attempt payloads, evidence references, message owners, observations, purge tombstones, stable artifact leases and operation identities, and bounded legacy migration journals.
- [x] #2 One transaction writes or reuses the final message and every sealed aggregate row; incomplete builders write no canonical rows.
- [x] #3 Prompt and attempt metadata are bounded, while governed source and content fields remain outside immutable trace JSON.
- [x] #4 Citation tables are excluded from FTS, Library indexing, and RAG ingestion, and schema migration and rollback-safety tests pass.
- [x] #5 A stable local profile and authority context plus injected fingerprint key are available for writes, and a recovery switch can disable repository writes while preserving authorized aggregate reads.
- [x] #6 ChatPersistenceService receives citation persistence explicitly and fails before any write when a sealed citation is supplied without an available repository, policy, identity, or key.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reserve the next ChaChaNotes schema version and add failing fresh-create, upgrade, exact-schema, origin-coherence, FK/index, inventory-exclusion, and rollback tests.
2. Add one standalone citation-provenance SQL migration and execute complete statements through the active transaction cursor, creating stable identity context and updating version last.
3. Add the default-off CitationProvenanceRuntimePolicy and injected identity/key seam with read-without-key and fail-before-transaction write behavior.
4. Add failing repository tests for complete atomic writes, row-family rollback injection, aggregate governance separation, summary reads, and authorized hydration denials.
5. Implement CitationTraceRepository over caller-owned ChaChaNotes transactions and the explicit optional ChatPersistenceService sealed-write seam with no silent grounded fallback.
6. Add config defaults/typed access and prove existing no-citation callers retain current behavior.
7. Extend qualification benchmarks with sealed repository storage proxies, run focused database/chat/config/performance suites, lint, diff, and independent review gates without rewriting the committed baseline.
8. Complete acceptance criteria and implementation notes only after both reviews approve.

ADR required: yes
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: This task implements ADR-024’s accepted schema, stable identity, governance, recovery-switch, and atomic persistence boundaries; no new architecture decision is introduced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the exact 12-table citation-provenance schema, stable local identity context, default-off runtime policy, atomic CitationTraceRepository, authorized summary/hydration reads, and explicit ChatPersistenceService sealed-write seam. `TASK-557` later integrated the migration into current `dev` as v26→v27. The standalone SQL is the sole DDL source and is executed statement-by-statement through the active transaction cursor; identity creation and version update are rollback-safe and the version changes last. Exact schema tests cover columns/types/nullability/PKs, every unique/composite/partial index and predicate, grouped FKs/delete behavior, bounds/CHECKs, fresh/upgrade/partial-schema paths, and FTS/sync/Library/RAG exclusion.

Writes revalidate and prepare immutable canonical row tuples before any transaction, bind the exact issued object and digest to the repository, enforce same-DB affinity, stable identity/key policy, local run authority, aggregate/governed bounds, and full message/provenance rollback. No silent grounded fallback is possible. Keyless summaries remain readable; governed hydration requires both sealed and request snapshot/source-identity capabilities and uses one parameterized guarded SQLite statement to reassert exact row/reference sets, scope, authority, redaction/purge state, JSON shape, and tombstone absence before returning any governed field.

Verification: 425 tests passed and 2 skipped in the broad regression run; focused repository review tests passed 53/53; Ruff check/format and git diff checks passed. Qualification at 30 samples/5 warmups remained eligible with overall_pass=true: sealed write p95 5.225 ms (<75 ms), summary read p95 0.243 ms (<25 ms), and 35/35 trace-owner rows. The committed v1 baseline and message-only control were unchanged. Independent specification and quality/security reviews approved the final implementation with no remaining Critical or Important findings.

ADR required: yes. Applied existing backlog/decisions/024-rag-citation-provenance-and-source-resolution.md; no new ADR was needed.
<!-- SECTION:NOTES:END -->
