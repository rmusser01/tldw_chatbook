---
id: TASK-10.6.1
title: 'Gate 1.5.1: Console native display-state contracts'
status: To Do
assignee: []
created_date: '2026-05-07 03:37'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - gate-1-5-console
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md
  - >-
    Docs/superpowers/plans/2026-05-07-gate-1-5-console-internals-decomposition.md
parent_task_id: TASK-10.6
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create pure Console display-state seams and mounted red regressions that define the native Console chrome before replacing legacy ChatWindowEnhanced internals.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console display-state helpers expose provider model persona source RAG tool approval artifact and recovery labels without reading widget internals.
- [ ] #2 Mounted regressions prove the current Console still exposes legacy embedded chrome that Gate 1.5 must remove or isolate.
- [ ] #3 Existing Gate 1 Console shell tests remain runnable as compatibility guardrails.
<!-- AC:END -->
