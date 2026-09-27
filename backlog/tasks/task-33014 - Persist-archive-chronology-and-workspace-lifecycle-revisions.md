---
id: TASK-33014
title: Persist archive chronology and workspace lifecycle revisions
status: To Do
assignee: []
created_date: '2026-09-27 15:19'
labels: []
dependencies:
  - TASK-32772
documentation:
  - Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md
  - backlog/decisions/166-agent-assisted-archive-recovery.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make archive dates accurate and stale workspace recovery detectable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both stores record real archive times and preserve unknown historical dates.
- [ ] #2 Workspace lifecycle changes use atomic archive-revision checks, including existing UI and Undo paths.
- [ ] #3 Migration and real SQLite tests preserve conversation version and sync invariants.
<!-- AC:END -->
