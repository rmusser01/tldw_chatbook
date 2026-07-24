---
id: TASK-401.2
title: Define canonical citation trace contracts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 00:43'
updated_date: '2026-07-24 04:48'
labels:
  - rag
  - citations
  - contracts
  - foundation
dependencies:
  - TASK-401.1
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-401
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Introduce pure versioned provenance and identity contracts that distinguish immutable trace structure from governed retrieval, snapshot, and answer-attempt payloads before persistence schema is fixed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Contracts cover CitationTrace, EvidenceRun, PromptEvidenceSet, AnswerAttempt, EvidenceSnapshot references, CitationOccurrence, completeness, and trust.
- [ ] #2 The selected attempt deterministically reduces mixed evidence storage modes into complete, partial, redacted, or unavailable.
- [ ] #3 Aggregate serialization excludes governed text, source identity, locators, lineage, and content hashes.
- [ ] #4 Pure round-trip, bounds, marker occurrence, and legacy EvidenceBundle or CitationRef adapter tests pass.
- [ ] #5 Local, server, imported, payload, and owner namespaces plus domain-separated secret fingerprint contracts are defined and tested before schema work begins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing invariant, round-trip, bounds, marker, completeness, and legacy adapter tests.
2. Implement the smallest frozen canonical trace and governed write-bundle model graph.
3. Add deterministic selected-attempt completeness reduction and property tests.
4. Add pure legacy EvidenceBundle/CitationRef adapters that never overstate completeness.
5. Add failing identity/fingerprint/key-provider tests, then implement pure namespace and secret-scoped identity contracts.
6. Run focused compatibility, property, lint, and benchmark-regression checks and self-review.
7. Complete acceptance criteria and implementation notes after both review gates pass.

ADR required: yes
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: This task establishes the immutable provenance and identity contracts governed by ADR-024 before persistence schema is fixed.
<!-- SECTION:PLAN:END -->
