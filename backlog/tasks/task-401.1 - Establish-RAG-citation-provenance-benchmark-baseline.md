---
id: TASK-401.1
title: Establish RAG citation provenance benchmark baseline
status: To Do
assignee: []
created_date: '2026-07-24 00:43'
labels:
  - rag
  - citations
  - performance
  - foundation
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-401
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Record a reproducible pre-feature performance and storage baseline so citation provenance has numeric delivery budgets before persistence work begins.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A versioned deterministic corpus and fixture manifest exercises representative local RAG retrieval, generation, conversation persistence, and exact boundary or over-bound payload cases.
- [ ] #2 The benchmark records exact commands, sample and warmup rules, supported hardware and provider envelope, current results, and a committed machine-readable v1 baseline.
- [ ] #3 Numeric pass or fail budgets cover first-token regression, finalization, inspector data load, trace size, database growth, and migration throughput.
- [ ] #4 External source refresh and network latency are measured separately from local rendering and persistence budgets.
<!-- AC:END -->
