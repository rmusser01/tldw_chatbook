---
id: TASK-31937
title: Compile and execute declarative Canvas diagrams
status: To Do
assignee: []
created_date: '2026-09-06 22:12'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31936
documentation:
  - >-
    backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
  - Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md
  - Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Connect inert Mermaid declarations to one transactional QuickJS startup before authored scripts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 V1 wire output remains unchanged and V2 plans preserve exact source identity with closed diagram records and verified profiles.
- [ ] #2 All declarations render once before authored scripts with shared byte, time, memory, DOM and patch limits and no partial startup commit.
- [ ] #3 Renderer and worker reject invalid output and stale failures while preserving zero-egress bootstrap acknowledgement and private controls.
<!-- AC:END -->
