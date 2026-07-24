---
id: TASK-401.8
title: Add crash safe artifact citation ownership
status: To Do
assignee: []
created_date: '2026-07-24 00:44'
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
