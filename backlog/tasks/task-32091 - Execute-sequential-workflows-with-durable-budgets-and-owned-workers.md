---
id: TASK-32091
title: Execute sequential workflows with durable budgets and owned workers
status: To Do
assignee: []
created_date: '2026-09-08 21:02'
labels:
  - workflows
  - runtime
  - safety
dependencies:
  - TASK-32089
  - TASK-32090
references:
  - Docs/superpowers/plans/2026-09-08-workflows-local-file-to-note.md
documentation:
  - Docs/superpowers/specs/2026-09-08-workflows-local-first-parity-design.md
  - backlog/decisions/138-portable-workflow-definitions-and-local-execution.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run saved local workflow snapshots sequentially while retaining physical worker ownership and recording truthful durable attempt and budget state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Duplicate launch delivery returns the same run for the same operation payload, changed payloads conflict, and Run again creates a new operation.
- [ ] #2 A single run-state writer atomically records ordered attempts, outputs, events, and persistent limits; no UI or adapter writes run state directly.
- [ ] #3 Pause, cancel, timeout, and retry cannot release capacity or start an overlapping effect while prior owned work is alive; late completions never advance a cancelled run.
<!-- AC:END -->
