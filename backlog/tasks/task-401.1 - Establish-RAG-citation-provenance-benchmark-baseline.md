---
id: TASK-401.1
title: Establish RAG citation provenance benchmark baseline
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 00:43'
updated_date: '2026-07-24 04:47'
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
- [x] #1 A versioned deterministic corpus and fixture manifest exercises representative local RAG retrieval, generation, conversation persistence, and exact boundary or over-bound payload cases.
- [x] #2 The benchmark records exact commands, sample and warmup rules, supported hardware and provider envelope, current results, and a committed machine-readable v1 baseline.
- [x] #3 Numeric pass or fail budgets cover first-token regression, finalization, inspector data load, trace size, database growth, and migration throughput.
- [x] #4 External source refresh and network latency are measured separately from local rendering and persistence budgets.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add deterministic synthetic fixtures and a versioned manifest with exact-limit and over-bound cases.
2. Add failing benchmark contract tests covering network isolation, reproducibility, baseline compatibility, and all budget families.
3. Implement the Console/control-path benchmark runner and committed machine-readable baseline.
4. Run focused tests and baseline measurements, document the reference environment and budgets, and self-review.
5. Complete TASK-401.1 acceptance criteria and implementation notes after both review gates pass.

ADR required: yes
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: This benchmark is the prerequisite measurement contract for the ADR-024 storage and pipeline implementation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the ADR-024 prerequisite benchmark as a deterministic, network-isolated harness. Added compact domain-shaped fixtures and manifest validation; real Console control-path TTFB measurement; corpus-driven finalization, inspector, storage-growth, and restart migration measurements; exact-limit acceptance and +1 rejection; direct historical and in-process qualification gates; an isolated optional external mode; strict host/config/secret isolation; malformed-baseline and statistical-eligibility checks; and a committed 30-sample/5-warmup baseline plus reproducibility documentation. Kept large boundary payloads descriptor-generated to avoid repository bloat. Verification completed with 316 integrated tests, Ruff check/format, diff checks, and passing fresh baseline/qualification runs. Modified the six planned benchmark/fixture/test/documentation artifacts. ADR: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md.
<!-- SECTION:NOTES:END -->
