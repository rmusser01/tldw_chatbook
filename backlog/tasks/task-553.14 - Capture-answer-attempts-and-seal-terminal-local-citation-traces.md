---
id: TASK-553.14
title: Capture answer attempts and seal terminal local citation traces
status: In Progress
assignee: []
created_date: '2026-07-26 18:18'
updated_date: '2026-07-26 19:03'
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
- [ ] #2 Sealing requires closed, chronologically ordered local retrieval and produces a one-shot validated SealedCitationWrite with repository-owned policy metadata, selected-attempt linkage, and deterministic completeness.
- [ ] #3 Eligible marker-free initial Console direct-provider and agent generations use the same repository for capture and persistence, seal from the exact materialized visible body, and atomically persist the message plus trace under stable idempotent identities.
- [ ] #4 Disabled, marker-mapping-ineligible, or deterministically unavailable canonical persistence preserves the ordinary answer as ungrounded, while ambiguous transaction failure receives at most one same-identity retry and never leaves partial provenance.
- [ ] #5 Failed, stopped, canceled, empty, retry, and regenerate paths do not seal or inherit unfinished builders.
- [ ] #6 Focused tests cover builder atomicity, production repository wiring, persistence-capability gating, direct and agent completion, exact-body fidelity, transient-finalizer cleanup, atomic persistence, fallback, idempotent retry, and content-free diagnostics.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: Direct implementation of ADR-024’s accepted request-scoped builder, terminal seal, governed answer body, message ownership, and atomic persistence contracts; no new architectural decision.

Detailed plan: Docs/superpowers/plans/2026-07-26-local-answer-attempt-terminal-sealing.md

1. Add repository-owned local seal policy, bounded initial answer attempts, closed-run chronology, and one-shot builder sealing.
2. Return the exact prompt-evidence-set identity from every successful local capture path.
3. Expose fail-closed canonical-write readiness and wire the app’s exact citation repository into Console persistence.
4. Add transient terminal finalization, early-write deferral, stable identity, deterministic fallback, and one ambiguous same-write retry to ConsoleChatStore.
5. Install finalizers only for initial direct-provider and agent sends; clear them on every non-success, empty, retry, regenerate, replacement, and outer-exit path.
6. Prove exact-body atomic persistence and rollback with real SQLite integration tests, then run only the touched-code verification listed in the detailed plan.
<!-- SECTION:PLAN:END -->
