---
id: TASK-31937
title: Compile and execute declarative Canvas diagrams
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-06 22:12'
updated_date: '2026-09-07 02:55'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31936
documentation:
  - >-
    backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
  - Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md
  - >-
    Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing V2 source-preservation and closed-schema compiler tests; implement bounded single-parse profile admission.
2. Integrate verified inert library data and private QuickJS diagram handles into the existing single startup budget and transaction.
3. Exercise real worker/renderer startup, malformed wire, atomic failure, private handles and zero-egress regressions; regenerate assets and self-review.
ADR required: yes
ADR path: backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md (existing, extends ADR-121)
Reason: Direct implementation of the approved profile wire and transactional runtime boundary; no new privileges.
<!-- SECTION:PLAN:END -->
