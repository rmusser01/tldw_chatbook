---
id: TASK-32094
title: Integrate workflow runs results settings and exact Console follow
status: To Do
assignee: []
created_date: '2026-09-08 21:11'
labels:
  - workflows
  - ui
  - runtime
dependencies:
  - TASK-32092
  - TASK-32093
references:
  - Docs/superpowers/plans/2026-09-08-workflows-local-file-to-note.md
documentation:
  - Docs/superpowers/specs/2026-09-08-workflows-local-first-parity-design.md
  - backlog/decisions/138-portable-workflow-definitions-and-local-execution.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Connect the redesigned Workflows destination to app-owned execution and canonical settings while keeping drafts and historical run identities distinct.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Real app navigation composes one runtime, preserves execution across screen changes, flushes drafts and drains owned work before dependent stores close.
- [ ] #2 Run, review, pause, cancel and retry controls report actual capability and lifecycle state; defaults and explicit per-run limit amendments remain distinct.
- [ ] #3 Results retain workflow, revision, run, target, step and attempt provenance; Console follows the exact selected run and stale targets never substitute the latest run.
<!-- AC:END -->
