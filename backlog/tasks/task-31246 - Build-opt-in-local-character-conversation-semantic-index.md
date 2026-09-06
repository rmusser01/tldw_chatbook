---
id: TASK-31246
title: Build opt-in local character conversation semantic index
status: To Do
assignee: []
created_date: '2026-09-04 02:09'
labels:
  - rag
  - embeddings
  - search
  - privacy
dependencies:
  - TASK-31245
references:
  - >-
    Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md
  - >-
    Docs/superpowers/plans/2026-09-03-character-conversation-navigation-implementation.md
priority: high
---

## Renumbering provenance

Renumbered from TASK-31238 on 2026-09-04. The final pre-commit worktree sweep
found the older `Review set replacement notice and distinguishable picker rows`
task created at 01:50; it keeps TASK-31238 under the older-arrival rule. This
unshipped task moves with all plan and dependency references.

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a default-off, UI-unreachable local semantic indexing subsystem for eligible character conversations with truthful lifecycle, atomic generations, bounded direct ANN retrieval, and no transcript persistence in the vector store.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Local model validation performs no remote fallback and the feature remains disabled and unreachable from production UI.
- [ ] #2 CharacterConversationVectorStore writes embeddings and safe metadata only, pins cosine HNSW semantics, and never stores documents or transcript plaintext.
- [ ] #3 Generation manifests include model, dimension, normalization, chunk, eligibility, projection, metric, distance, and aggregation compatibility fields.
- [ ] #4 Initial builds and rebuilds cut over atomically; partial, paused, cancelled, failed, damaged, or incompatible generations never rank.
- [ ] #5 Durable outbox revisions and per-conversation ready fences prevent stale or mixed chunks and converge through idempotent replay and reconciliation.
- [ ] #6 Typed query outcomes distinguish results, unavailable, damaged, and query errors; only successful empty results mean no matches.
- [ ] #7 Direct ANN queries scan at most 200 chunks plus one refill to 400, aggregate by lowest cosine distance, cap at 50 conversations, and use deterministic ties.
- [ ] #8 Tests prove excluded-content privacy, no-plaintext storage, semantic-only retrieval, crash recovery, mutation invalidation, bounded memory, latency, and network-free operation.
<!-- AC:END -->
