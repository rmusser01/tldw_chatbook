---
id: TASK-31936
title: Implement bounded Canvas diagram layout
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 22:11'
updated_date: '2026-09-07 02:55'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31935
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
Render useful flow and sequence models with deterministic geometry inside the existing resource boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Approved flow branches and rejoining, sequence messages and notes, and Unicode labels have deterministic bounded geometry.
- [x] #2 Individual and aggregate work, text, SVG, geometry and area ceilings fail explicitly without truncating source or raising V1 limits.
- [x] #3 Default typography and scrolling remain readable under qualified fonts, with documented CSS override behavior and no native measurements.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing exact layout/geometry/aggregate budget tests around the existing QuickJS-only probe.
2. Implement bounded deterministic flow/sequence scenes, shared layout accounting and pinned grapheme wrapping using only V1 rendering privileges.
3. Verify useful and adversarial fixtures, typography/scrolling evidence, reproducible assets and scoped static checks; document limits and obtain independent review.
ADR required: yes
ADR path: backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md (existing, extends ADR-121)
Reason: Direct implementation of accepted bounded layout, typography and shared runtime limits; no new privileges or schema.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the ADR-124 candidate layout with Kahn ranks, bounded ordering sweeps,
orthogonal lanes, sequence event/notes spacing, grapheme wrapping, inert scenes,
exact output accounting and shared refusal ceilings. Conservative 16px logical
cells replace the unqualified 8px rule; candidate identities and real layout
hashes were regenerated. Public worker/compiler integration is deferred to Task 4.

Evidence: 172 semantic/layout tests; 66 profile/assets cases plus the separately
permitted loopback redirect case; offline rebuild twice matched packaged bytes.
Two scene-only Chromium tests verify actual glyph extents/non-overlap, inherited
typography, scrolling, explicit CSS override, source identity, zero generated
egress, and four mixed scenes through the unchanged V1 worker patch budget.
Ruff/formatter, JS syntax and diff checks pass. Existing RequestsDependencyWarning
persists; optional V1 engine archive rebuild was not requested. Candidate remains
non-executable/default null. In Progress pending controller independent review.
Full evidence and qualification limitations are in the Task 3 implementation report.

Independent task review and scoped fix re-review complete. Fixed the reported TD/LR branch route-label collisions with reserved outer strips and charged segment/label checks; no quotas raised. Final evidence: 36 layout tests, 2 scene browser tests, 67 profile/assets tests with twice-reproducible Mermaid build; baseline warning and optional V1 archive-cache skip remain. Actual V2 startup and release qualification remain owned by Tasks 4 and 8. ADR-124 retained.
<!-- SECTION:NOTES:END -->
