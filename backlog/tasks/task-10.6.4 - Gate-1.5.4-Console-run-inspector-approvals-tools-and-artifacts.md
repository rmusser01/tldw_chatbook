---
id: TASK-10.6.4
title: 'Gate 1.5.4: Console run inspector approvals tools and artifacts'
status: To Do
assignee: []
created_date: '2026-05-07 03:37'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - gate-1-5-console
dependencies:
  - TASK-10.6.3
documentation:
  - Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md
  - >-
    Docs/superpowers/plans/2026-05-07-gate-1-5-console-internals-decomposition.md
parent_task_id: TASK-10.6
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move run tool approval RAG and artifact state into the Console inspector/action model so operators can understand and recover live agentic work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Run inspector shows live-work provenance tool readiness approval status RAG/source state artifact availability and recovery actions from Console display-state seams.
- [ ] #2 Approval tool-call and Chatbook artifact actions remain reachable from Console with target-specific disabled reasons when unavailable.
- [ ] #3 Mounted tests cover blocked provider missing RAG/source and pending approval/artifact states.
<!-- AC:END -->
