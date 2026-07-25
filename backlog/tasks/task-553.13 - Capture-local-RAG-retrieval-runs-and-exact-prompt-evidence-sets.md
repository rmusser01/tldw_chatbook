---
id: TASK-553.13
title: Capture local RAG retrieval runs and exact prompt evidence sets
status: In Progress
assignee: []
created_date: '2026-07-25 13:52'
updated_date: '2026-07-25 13:56'
labels:
  - rag
  - citations
  - provenance
  - local-pipeline
dependencies:
  - TASK-553.2
  - TASK-553.3
  - TASK-553.4
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md
  - Docs/superpowers/plans/2026-07-25-local-rag-retrieval-prompt-evidence-capture.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-553
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Carry one local RAG request’s ordered retrieval results and the exact transformed evidence submitted to the provider into a request-scoped canonical citation builder, so later answer-attempt and sealing work has trustworthy prompt-boundary provenance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Canonical local RAG capture records an ordered evidence run and governed candidate metadata for the retrieval execution without putting sensitive payload fields in immutable trace data or logs.
- [ ] #2 The prompt evidence set preserves the exact post-formatting and post-truncation text submitted for every included source, with stable chatbook_s_v1 marker ordinals and governed embedded snapshot payloads.
- [ ] #3 Retrieved candidates omitted by the prompt character budget retain bounded explanatory metadata but no snapshot text, and unauthorized or invalid evidence fails closed before prompt submission.
- [ ] #4 Canonical capture remains request-scoped and unsealed; this task does not persist partial builders or implement answer attempts, occurrence parsing, repair, source opening, export, or sync.
- [ ] #5 When canonical writes are disabled or capture prerequisites are unavailable, existing local RAG return values and provider prompt text remain backwards compatible and no partial provenance is retained.
- [ ] #6 Focused unit and integration tests cover enabled capture, truncation and exclusion, disabled compatibility, empty results, malformed result metadata, and the plain/semantic/hybrid/custom pipeline boundary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a pure request-scoped CitationTraceBuilder that can record bounded local EvidenceRun metadata and exact PromptEvidenceSet payloads without sealing or persistence.
2. Add a repository-owned factory that supplies the builder with the existing local identity and keyed fingerprint context only when canonical capture is ready.
3. Extend RAG context formatting with an opt-in structured capture result so the exact marked, transformed, and truncated evidence text is the same text prepended to the provider request.
4. Wire the local Chat RAG boundary to retain the request-scoped capture while preserving the current string API and byte-for-byte behavior when canonical writes are disabled or prerequisites are unavailable.
5. Add RED/GREEN unit and integration coverage for ordering, bounds, truncation/exclusion, malformed evidence, all pipeline modes, and disabled compatibility; update citation pipeline documentation and run focused plus repository verification.

ADR required: yes
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: Implements ADR-024’s local retrieval-run and exact prompt-boundary evidence capture contract without changing the accepted architecture.
<!-- SECTION:PLAN:END -->
