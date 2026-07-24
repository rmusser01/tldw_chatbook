---
id: TASK-401.8
title: Add crash safe artifact citation ownership
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 00:44'
updated_date: '2026-07-24 12:57'
labels:
  - rag
  - citations
  - artifacts
  - reliability
dependencies:
  - TASK-401.5
  - TASK-401.6
references:
  - Docs/superpowers/specs/2026-07-23-rag-citation-provenance-design.md
  - Docs/superpowers/plans/2026-07-23-rag-citation-provenance-foundation.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-401
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep immutable traces alive for saved artifacts across same-database and cross-database save or delete failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The ownership contract requires a foreign key and shared transaction for any same-database backend; the current JSON registry is explicitly classified and tested through the cross-store path.
- [ ] #2 Cross-database saves and deletes use durable pending operations, a stable owner lease, and separately idempotent link and unlink operation identities.
- [ ] #3 Startup or background reconciliation completes interrupted link and unlink operations, including final release after a durable unlink acknowledgement.
- [ ] #4 Garbage collection cannot remove a trace during a pending link, live artifact lease, or unresolved unlink.
- [ ] #5 Disabled canonical writes prevent artifact lease mutation and reconciliation without blocking ordinary artifact save or canonical reads.
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
