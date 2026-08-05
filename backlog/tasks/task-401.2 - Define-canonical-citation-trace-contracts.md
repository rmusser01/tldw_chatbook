---
id: TASK-401.2
title: Define canonical citation trace contracts
status: To Do
assignee: []
created_date: '2026-07-24 00:43'
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
