---
id: TASK-401.9
title: Migrate legacy citation sidecars safely
status: In Progress
assignee: []
created_date: '2026-07-24 00:44'
updated_date: '2026-07-24 14:37'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. ADR required: yes; ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md; Reason: implement the accepted storage/migration and canonical/legacy read boundary without a new decision.
2. Add failing pure legacy synthesis tests for bounded adapters, malformed/partial inputs, markers, inert locators, and Chatbook package citations.
3. Add failing journal, bounded-batch, interruption/restart, idempotency, raw-fingerprint, and divergence tests.
4. Implement staged per-conversation migration, atomic visibility cutover, canonical-first dual reads, explicit divergence, and single-write compatibility enforcement.
5. Route legacy Chatbook import and creator reads through the legacy adapter/canonical repository without portable provenance semantics.
6. Wire deferred startup migration with at most one 100-message batch per idle unit and disabled-mode suppression.
7. Replace the benchmark proxy with the real migration service candidate and run qualification plus adjacent regression gates.
8. Self-review, independent spec/quality review, then complete Backlog hygiene.
<!-- SECTION:PLAN:END -->
