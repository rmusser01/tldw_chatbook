---
id: TASK-31934
title: Define Canvas runtime profile snapshots and admission
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-06 22:09'
updated_date: '2026-09-06 22:41'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31933
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
Give every Canvas operation one immutable compatibility and security-policy authority while keeping unqualified V2 profiles unavailable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Immutable profile identities cover engine, facade, plan, grammar, layout, Unicode, integrity and quotas, with bounded source-free failures.
- [x] #2 New, parent-based, historical and unavailable-profile selection obey the approved transition table without implicit upgrades or downgrades.
- [x] #3 Verified process-lifetime snapshots cannot be replaced by archive inputs or mutable asset rereads, and V1 admission remains compatible.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing profile selection and strict catalog tests using isolated pytest fixtures.
2. Implement immutable verified profile snapshots, bounded admission, and canonical cross-process identity; keep V2 unavailable.
3. Verify profile/runtime asset tests and scoped static checks, document compatibility, and obtain independent task review.
ADR required: yes
ADR path: backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md (existing, extends ADR-121)
Reason: Direct implementation of the accepted immutable runtime identity and security admission contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a strict packaged profile catalog, frozen source-free snapshot and pure exact-profile resolver. Extended the V1 runtime manifest and reproducible generator with engine, facade, plan, grammar, layout, Unicode and quota identities; runtime assets retain immutable verified manifest bytes. The production policy admits only V1 and leaves the diagram default unset, while tests cover every selection transition, unavailable identities, tampering, duplicate/unknown input, immutable retention and canonical cross-process identity. Added the V2 runtime compatibility schema and restart-required ownership documentation. ADR-124 remains the governing decision; no new ADR was required. Package-data registration for the new catalog remains intentionally assigned to the following packaging slice.
<!-- SECTION:NOTES:END -->
