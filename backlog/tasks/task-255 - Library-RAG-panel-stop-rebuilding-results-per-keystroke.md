---
id: TASK-255
title: Library RAG panel: stop rebuilding results/history per keystroke
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, library]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Input.Changed on the RAG query box (library_screen 11821-11827) calls the full panel-state refresh, tearing down and remounting the Evidence results list + Recent-searches history (~100+ widgets via awaited remove/mount) on every keystroke of an unsubmitted query — neither depends on unsubmitted text (search runs on Submitted). The status-line refresh is already a separable cheap function. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P1 B5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Typing in the query box updates only the run-button/status line; results/history refresh only on submit or outcome application
- [ ] #2 Existing RAG panel tests green
<!-- AC:END -->
