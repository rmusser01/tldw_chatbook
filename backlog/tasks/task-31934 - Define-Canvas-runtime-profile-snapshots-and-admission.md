---
id: TASK-31934
title: Define Canvas runtime profile snapshots and admission
status: To Do
assignee: []
created_date: '2026-09-06 22:09'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31933
documentation:
  - >-
    backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
  - Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md
  - Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give every Canvas operation one immutable compatibility and security-policy authority while keeping unqualified V2 profiles unavailable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Immutable profile identities cover engine, facade, plan, grammar, layout, Unicode, integrity and quotas, with bounded source-free failures.
- [ ] #2 New, parent-based, historical and unavailable-profile selection obey the approved transition table without implicit upgrades or downgrades.
- [ ] #3 Verified process-lifetime snapshots cannot be replaced by archive inputs or mutable asset rereads, and V1 admission remains compatible.
<!-- AC:END -->
