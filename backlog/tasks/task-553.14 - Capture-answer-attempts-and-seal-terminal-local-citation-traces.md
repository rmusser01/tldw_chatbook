---
id: TASK-553.14
title: Capture answer attempts and seal terminal local citation traces
status: To Do
assignee: []
created_date: '2026-07-26 18:18'
updated_date: '2026-07-26 18:26'
labels:
  - rag
  - citations
  - provenance
  - local-pipeline
dependencies:
  - TASK-553.13
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - >-
    Docs/superpowers/specs/2026-07-26-local-answer-attempt-terminal-sealing-design.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
  - TASK-553.13
parent_task_id: TASK-553
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete one eligible marker-free local RAG generation by binding its exact final assistant body to a governed answer attempt, sealing the request-scoped citation builder, and atomically persisting the message and canonical trace so retrieval provenance survives restart without overstating citation trust.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The local builder records a bounded governed initial answer attempt whose exact body and secret-scoped integrity fingerprint never enter immutable trace JSON or logs.
- [ ] #2 Sealing is one-shot and produces a validated local SealedCitationWrite with repository-owned policy metadata, selected-attempt linkage, and deterministic completeness.
- [ ] #3 Eligible marker-free initial Console direct-provider and agent generations seal from the exact materialized visible body and atomically persist the message plus trace under stable idempotent identities.
- [ ] #4 Disabled, marker-mapping-ineligible, or deterministically unavailable canonical persistence preserves the ordinary answer as ungrounded, while ambiguous transaction failure receives at most one same-identity retry and never leaves partial provenance.
- [ ] #5 Failed, stopped, canceled, empty, retry, and regenerate paths do not seal or inherit unfinished builders.
- [ ] #6 Focused tests cover builder atomicity, direct and agent completion, exact-body fidelity, transient-finalizer cleanup, atomic persistence, fallback, idempotent retry, and content-free diagnostics.
<!-- AC:END -->
