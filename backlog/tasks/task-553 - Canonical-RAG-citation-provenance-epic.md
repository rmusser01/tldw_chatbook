---
id: TASK-553
title: Canonical RAG citation provenance epic
status: In Progress
assignee: []
created_date: '2026-07-24 00:42'
updated_date: '2026-07-26 18:18'
labels:
  - rag
  - citations
  - provenance
  - epic
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Carry exact prompt-boundary RAG evidence through local and server answer generation into durable governed traces, user inspection, source resolution, artifacts, export, import, and compatible synchronization without overstating structural or semantic trust.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Canonical traces preserve selected-attempt retrieval, submitted evidence, citation occurrences, and trust state across restart.
- [ ] #2 Users can distinguish cited evidence from additionally submitted context and inspect or open policy-permitted sources.
- [ ] #3 Persistence, revocation, migration, export, import, cache reuse, artifacts, and synchronization follow ADR-024 without leaking restricted metadata.
- [ ] #4 Local and compatible server RAG paths remain backwards compatible and pass source, security, performance, and quality gates.
<!-- AC:END -->
