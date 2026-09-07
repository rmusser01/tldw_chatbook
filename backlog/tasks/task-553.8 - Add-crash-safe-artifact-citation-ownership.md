---
id: TASK-553.8
title: Add crash safe artifact citation ownership
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 00:44'
updated_date: '2026-07-24 14:35'
labels:
  - rag
  - citations
  - artifacts
  - reliability
dependencies:
  - TASK-553.5
  - TASK-553.6
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-553
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep immutable traces alive for saved artifacts across same-database and cross-database save or delete failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The ownership contract requires a foreign key and shared transaction for any same-database backend; the current JSON registry is explicitly classified and tested through the cross-store path.
- [x] #2 Cross-database saves and deletes use durable pending operations, a stable owner lease, and separately idempotent link and unlink operation identities.
- [x] #3 Startup or background reconciliation completes interrupted link and unlink operations, including final release after a durable unlink acknowledgement.
- [x] #4 Garbage collection cannot remove a trace during a pending link, live artifact lease, or unresolved unlink.
- [x] #5 Disabled canonical writes prevent artifact lease mutation and reconciliation without blocking ordinary artifact save or canonical reads.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: Implements the accepted cross-store artifact ownership and recovery decision; no new ADR is required.

1. Review existing registry, repository, lifecycle, Console save, and startup wiring contracts.
2. RED: add bounded registry/outbox and real shared-database backend contract tests.
3. RED: add lease phase-interruption/restart/idempotency/concurrency crash-matrix tests and garbage-collection barriers.
4. Implement the minimal ownership coordinator plus repository/lifecycle lease and receipt transitions.
5. Carry verified opaque ownership requests through both Console artifact save seams using atomic registry mutation plus outbox.
6. Wire bounded deferred startup reconciliation behind the canonical-write recovery switch with sanitized failures.
7. Run the specified five-file gate, adjacent citation repository/lifecycle tests, startup performance, Ruff, formatting, and diff checks; self-review against the design/ADR.
8. Leave acceptance criteria unchecked and the task In Progress for independent spec and code-quality review; complete Backlog hygiene only after those gates approve.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented crash-safe citation ownership for saved artifacts across cross-store JSON and shared-database backends.

- Added a bounded, atomically persisted JSON cross-store outbox; shared-database backends now prove real composite foreign keys and apply artifact ownership in the trace database transaction.
- Bound opaque owner requests to exact artifact bodies with keyed MACs, stable signed leases, separately idempotent link/unlink identities, and immutable artifact revisions.
- Implemented four-phase recovery (apply, durable acknowledgement, trace finalization, prune), restart-safe handshakes, a durable fair cursor, and mandatory race-safe garbage-collection barriers.
- Preserved ordinary artifact mutation while canonical writes are disabled or the verification key is unavailable by recording signed deferred unlinks for later bounded recovery; canonical reads remain available.
- Wired both Console save seams and bounded sanitized startup/background reconciliation. Enforced strict schemas, size/batch limits, opaque identifiers, keyed fingerprints, sanitized errors, and no persisted raw citation bodies.
- ADR required: yes. Existing ADR-024 governs the design; no new ADR was required.
- Feature/correction commits: 32d7dbf, c2de890, c422c8b, 7a31d37.
- Verification: ownership 48 passed; specified task gate 85 passed; native 1 passed; startup 2 passed; adjacent suite 133 passed with the same 6 pre-existing fixed-clock lifecycle failures; Ruff, formatting, and diff checks clean. Independent specification and quality reviews approved.
<!-- SECTION:NOTES:END -->
