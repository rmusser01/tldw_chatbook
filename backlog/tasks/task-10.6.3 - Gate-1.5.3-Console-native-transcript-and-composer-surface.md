---
id: TASK-10.6.3
title: 'Gate 1.5.3: Console native transcript and composer surface'
status: To Do
assignee: []
created_date: '2026-05-07 03:37'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - gate-1-5-console
dependencies:
  - TASK-10.6.2
documentation:
  - Docs/superpowers/specs/2026-05-06-screen-design-adaptation-audit-design.md
  - >-
    Docs/superpowers/plans/2026-05-07-gate-1-5-console-internals-decomposition.md
parent_task_id: TASK-10.6
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the embedded ChatWindowEnhanced main surface with Console-owned transcript and composer widgets that reuse existing ChatSession and tab services where safe.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console transcript/event stream is a native region and no longer contains the full legacy ChatWindowEnhanced container.
- [ ] #2 Console composer is a native action row with send stop attach and save-Chatbook affordances wired through existing chat handlers or documented compatibility adapters.
- [ ] #3 Basic chat tabs session state handoff draft text and streaming fallback regressions continue to pass.
<!-- AC:END -->
