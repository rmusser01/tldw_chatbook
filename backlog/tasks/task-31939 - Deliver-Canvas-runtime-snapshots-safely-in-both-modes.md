---
id: TASK-31939
title: Deliver Canvas runtime snapshots safely in both modes
status: To Do
assignee: []
created_date: '2026-09-06 22:14'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31938
documentation:
  - >-
    backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
  - Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md
  - Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep native and served browser delivery consistent with verified restart-bound runtime policy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Native host and served parent-child delivery use matching immutable build, catalog and policy snapshots and verified cached bytes.
- [ ] #2 Restart into a revoked policy, mixed process identities and stale loads fail closed; explicit Canvas disable still stops live execution.
- [ ] #3 Two-browser capability, selection freshness, source-only recovery and confirmed bridge isolation pass without adding a public port or weakening authentication.
<!-- AC:END -->
