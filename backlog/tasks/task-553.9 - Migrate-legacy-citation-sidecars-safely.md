---
id: TASK-553.9
title: Migrate legacy citation sidecars safely
status: Done
assignee: []
created_date: '2026-07-24 00:44'
updated_date: '2026-07-24 16:21'
labels:
  - rag
  - citations
  - migration
  - compatibility
dependencies:
  - TASK-553.4
  - TASK-553.5
  - TASK-553.6
  - TASK-553.3
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-553
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide bounded dual-read and canonical single-write migration from existing evidence bundles, validation metadata, chat RAG sidecars, and legacy Chatbook package data without implementing the future portable provenance import protocol.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Legacy EvidenceBundle, CitationRef, citation_validation, and chat_rag_context records read as partial legacy_inferred traces.
- [x] #2 Conversation migration uses bounded batches, normalized journal progress, hidden staging rows, and an atomic visibility cutover; it is restartable and does not block opening or delete the legacy sidecar.
- [x] #3 Free-form legacy paths, URLs, and content references remain inert unless a current allowlisted authority lookup maps them safely.
- [x] #4 Post-cutover legacy modifications are reported as divergence and are never silently merged into canonical provenance.
- [x] #5 Legacy Chatbook package citations adapt only to partial legacy_inferred traces; portable canonical import, authority rebinding, and imported-origin identity remain out of scope.
- [x] #6 Disabled recovery mode preserves pre-cutover compatibility writes; enabled canonical mode permits no product sidecar citation writes or dual writes.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented bounded pure legacy synthesis for EvidenceBundle, CitationRef, validation, sidecar, and Chatbook package inputs; locators remain inert and deterministic secret-scoped identities prevent raw legacy identifiers from becoming authority. Added the normalized per-conversation journal and at-most-100-message hidden staging batches with CAS cutover, restart, interruption, and concurrency safety. Raw sidecar reads use descriptor no-follow checks, strict size/shape bounds, HMAC fingerprints, explicit divergence, and atomic visibility cutover. Canonical-first reads are verified and never merge changed legacy data; recovery mode preserves legacy compatibility writes, while canonical mode enforces single-write policy. Importer and creator share the migration composition. Deferred startup migration remains single-flight, policy-rechecked, bounded to one batch per yielded iteration, and isolates terminal conversation failures without consuming the retry budget. The benchmark now measures the real migration service honestly.

Commits: b7602269, d412cb77, 4ceac933, 5126d80a. Verification evidence: prescribed migration gate 176 passed; benchmark contracts 69 passed; DB/repository gate 103 passed; scheduler gate 8 passed. The 30-sample/5-warmup qualification passed overall at 1064.76 median and 1088.97 p95 messages/second, with 100 traces/owners, 34 partial traces/snapshots/references, and zero duplicates. Independent specification and quality reviews approved.

ADR required: yes. ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md. Existing ADR-024 governs this implementation; no new ADR was required.

Unrelated baseline disclosures: one Textual WaitForScreenTimeout remains outside this task, and app.py retains four pre-existing Ruff findings. Changed scope is clean.
<!-- SECTION:NOTES:END -->
