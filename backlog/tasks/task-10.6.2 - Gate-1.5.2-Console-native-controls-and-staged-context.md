---
id: TASK-10.6.2
title: 'Gate 1.5.2: Console native controls and staged context'
status: To Do
assignee: []
created_date: '2026-05-07 03:37'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - gate-1-5-console
dependencies:
  - TASK-10.6.1
documentation:
  - Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md
  - >-
    Docs/superpowers/plans/2026-05-07-gate-1-5-console-internals-decomposition.md
parent_task_id: TASK-10.6
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move provider/model readiness and staged context into Console-owned widgets while preserving legacy chat behavior behind compatibility seams.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console owns visible provider model persona RAG/source readiness controls outside the transcript region.
- [ ] #2 Staged Library Search/RAG and live-work context appears in a native staged-context tray with provenance and recovery copy.
- [ ] #3 Provider/model selector changes still synchronize with existing chat send behavior and compact controls.
<!-- AC:END -->
