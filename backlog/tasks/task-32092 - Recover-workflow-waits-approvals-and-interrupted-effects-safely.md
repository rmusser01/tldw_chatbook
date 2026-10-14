---
id: TASK-32092
title: Recover workflow waits approvals and interrupted effects safely
status: To Do
assignee: []
created_date: '2026-09-08 21:03'
labels:
  - workflows
  - recovery
  - safety
dependencies:
  - TASK-32091
references:
  - Docs/superpowers/plans/2026-09-08-workflows-local-file-to-note.md
documentation:
  - Docs/superpowers/specs/2026-09-08-workflows-local-first-parity-design.md
  - backlog/decisions/138-portable-workflow-definitions-and-local-execution.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve human decisions, permission review, attempt ownership and accumulated limits across pauses, shutdown and restart without replaying uncertain effects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Human decisions identify run, step, wait generation and actor; duplicates, late decisions, expired waits and changed effect payloads cannot authorize work.
- [ ] #2 Restart preserves absolute response deadlines, budgets, bindings and result provenance; unfinished or uncertain effects require review rather than automatic replay.
- [ ] #3 Orderly shutdown drains physical workers before closing stores, parked waits release only idle capacity, and explicit policy amendments never silently reset counters.
<!-- AC:END -->
