---
id: TASK-36006
title: Manage optional keyword indexes in Settings and qualify release
status: To Do
assignee:
  - '@codex'
created_date: '2026-10-21 14:32'
labels: []
dependencies:
  - TASK-36005
documentation:
  - Docs/superpowers/plans/2026-10-20-workspace-project-keyword-index.md
  - backlog/decisions/240-workspace-project-context-and-folder-bound-index.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users explicitly manage folder-bound keyword indexing and understand coverage from Settings and Console.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Settings exposes truthful Off/Building/Partial/Ready/Paused/Interrupted/Error controls and processing choices using existing UI tokens; Console shows status and a Settings route.
- [ ] #2 Compose/status projections perform no I/O, stale actions have no effects, and late binding/job/epoch callbacks cannot overwrite another view.
- [ ] #3 Isolated app journeys cover build/query/pause/resume/restart/refresh/detach/Clear with preserved Library choices and no semantic controls.
- [ ] #4 Both20k corpus shapes, physical retirement, permission races, missing embeddings and blocked sockets are qualified with measured limits and updated architecture/user documentation.
<!-- AC:END -->
