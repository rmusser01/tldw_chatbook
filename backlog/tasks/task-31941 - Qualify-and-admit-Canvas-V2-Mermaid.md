---
id: TASK-31941
title: Qualify and admit Canvas V2 Mermaid
status: To Do
assignee: []
created_date: '2026-09-06 22:15'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31940
documentation:
  - >-
    backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
  - Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md
  - Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the first immutable diagram profile only after end-to-end security and usability qualification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Mandatory real Chromium zero-egress and containment gates include diagram attacks, combined budgets, useful mixed documents and positive controls.
- [ ] #2 Native and same-origin served workflows, two-browser isolation, archive recovery, restart revocation, packaging and reproducibility have fresh recorded evidence.
- [ ] #3 Only a fully qualified immutable profile is admitted; failures leave V2 unavailable without changing V1 limits, and user and operator documentation report actual scope and coverage.
<!-- AC:END -->
