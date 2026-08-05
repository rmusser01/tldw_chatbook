---
id: TASK-401.9
title: Migrate legacy citation sidecars safely
status: To Do
assignee: []
created_date: '2026-07-24 00:44'
labels:
  - rag
  - citations
  - migration
  - compatibility
dependencies:
  - TASK-401.4
  - TASK-401.5
  - TASK-401.6
  - TASK-401.3
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-401
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide bounded dual-read and canonical single-write migration from existing evidence bundles, validation metadata, chat RAG sidecars, and legacy Chatbook package data without implementing the future portable provenance import protocol.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Legacy EvidenceBundle, CitationRef, citation_validation, and chat_rag_context records read as partial legacy_inferred traces.
- [ ] #2 Conversation migration uses bounded batches, normalized journal progress, hidden staging rows, and an atomic visibility cutover; it is restartable and does not block opening or delete the legacy sidecar.
- [ ] #3 Free-form legacy paths, URLs, and content references remain inert unless a current allowlisted authority lookup maps them safely.
- [ ] #4 Post-cutover legacy modifications are reported as divergence and are never silently merged into canonical provenance.
- [ ] #5 Legacy Chatbook package citations adapt only to partial legacy_inferred traces; portable canonical import, authority rebinding, and imported-origin identity remain out of scope.
- [ ] #6 Disabled recovery mode preserves pre-cutover compatibility writes; enabled canonical mode permits no product sidecar citation writes or dual writes.
<!-- AC:END -->
