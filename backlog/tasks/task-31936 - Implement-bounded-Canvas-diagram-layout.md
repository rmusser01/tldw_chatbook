---
id: TASK-31936
title: Implement bounded Canvas diagram layout
status: To Do
assignee: []
created_date: '2026-09-06 22:11'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31935
documentation:
  - >-
    backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
  - Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md
  - Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Render useful flow and sequence models with deterministic geometry inside the existing resource boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Approved flow branches and rejoining, sequence messages and notes, and Unicode labels have deterministic bounded geometry.
- [ ] #2 Individual and aggregate work, text, SVG, geometry and area ceilings fail explicitly without truncating source or raising V1 limits.
- [ ] #3 Default typography and scrolling remain readable under qualified fonts, with documented CSS override behavior and no native measurements.
<!-- AC:END -->
