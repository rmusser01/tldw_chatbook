---
id: TASK-401.1
title: Establish RAG citation provenance benchmark baseline
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 00:43'
updated_date: '2026-07-24 03:21'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add deterministic synthetic fixtures and a versioned manifest with exact-limit and over-bound cases.\n2. Add failing benchmark contract tests covering network isolation, reproducibility, baseline compatibility, and all budget families.\n3. Implement the Console/control-path benchmark runner and committed machine-readable baseline.\n4. Run focused tests and baseline measurements, document the reference environment and budgets, and self-review.\n5. Complete TASK-401.1 acceptance criteria and implementation notes after both review gates pass.\n\nADR required: yes\nADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md\nReason: This benchmark is the prerequisite measurement contract for the ADR-024 storage and pipeline implementation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Added the deterministic v1 corpus/manifest, baseline and qualification runner,
contract tests, machine-readable reference result, and benchmark report.
Corpus records now drive each measured seam and publish coverage plus stable
input hashes. Qualification compares first-token p95 directly with the
compatible committed v1 result and current in-process control. A fixed 20 ms
mock first-token floor keeps sub-millisecond scheduler/SQLite noise from
dominating the direct percentage comparison while retaining the independent 25
ms ceiling. Exact and one-unit-over domain values run through executable
validators for every frozen v1 count/byte limit; the 4 MiB trace is split into
valid 64 KiB snapshots. External resolver latency has a separate, explicit
network mode whose result is informational and excluded from local pass/fail.

The updated 30-sample/5-warmup ARM64 reference run passed all six local budget
families: first-token p95 22.112 ms, standard/maximum finalization p95
0.036/2.217 ms, inspector cold/warm p95 0.382/0.077 ms, SQLite growth p95
4,235,264 bytes for 4 MiB governed data, and migration median 109,293
messages/second with zero duplicate restart rows. Qualification runs passed
against the regenerated baseline without modifying it.

ADR required: yes

ADR path:
`backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`

Reason: This implements ADR-024's prerequisite measurement contract; no new
architectural decision was introduced.

The task remains In Progress with acceptance criteria unchecked pending the
required specification and quality review gates.
