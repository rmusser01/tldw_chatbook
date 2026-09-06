---
id: TASK-31940
title: Add Mermaid Canvas actions guidance and recovery
status: To Do
assignee: []
created_date: '2026-09-06 22:14'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31939
documentation:
  - >-
    backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
  - Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md
  - Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose the approved diagram experience with accurate assistant guidance and honest preview status.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Mermaid fences open through existing replay and ownership checks with exact escaped text and no script interpolation.
- [ ] #2 Bounded profile-aware guidance and executable examples retain unchanged tool parameters and source-free logs and cards.
- [ ] #3 Saved, pending, ready, failed and unavailable states are distinguished per load; source, View previous and confirmed unsent repair remain usable in both modes.
<!-- AC:END -->
