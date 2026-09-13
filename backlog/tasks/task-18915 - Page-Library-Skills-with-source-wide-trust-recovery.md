---
id: TASK-18915
title: Page Library Skills with source-wide trust recovery
status: Done
assignee: []
created_date: '2026-08-15 02:48'
updated_date: '2026-08-29 03:52'
labels:
  - library
  - pagination
  - skills
  - trust
dependencies:
  - TASK-18912
references:
  - >-
    Docs/superpowers/specs/2026-08-14-library-top-level-source-pagination-design.md
  - backlog/decisions/067-library-top-level-pagination-contracts.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make every installed Skill reachable through 20-item Library pages while keeping filtering literal and preserving truthful source-wide trust status and Review targeting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Skill name and description filtering, trust/status sorting, and deterministic identity ordering occur before 20-row slicing with an exact filtered total.
- [x] #2 Filtering never matches Skill bodies, supporting files, argument hints, metadata keys, or trust diagnostics.
- [x] #3 Each page response carries the source-wide blocked total and first blocked Skill name independently of the current page or active filter.
- [x] #4 Trust Review opens the stable blocked target directly even when it is off-page, and the trust header never reports a false page-local zero.
- [x] #5 Malformed page rows or metadata fail closed; stale mutation recovery suppresses inexact totals and disables unsafe row or pager actions.
- [x] #6 Skill request generations, unmount fencing, broad-snapshot isolation, focus, restoration, and recoverable errors match the approved design.
- [x] #7 Automated local-service/state, mounted Textual, geometry, trust, mutation, and isolated live verification pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend LocalSkillsService.list_skills with validated local query/status-or-name sort, exact pre-slice totals, and source-wide blocked_total/first_blocked_skill_name; test more than 40 records and excluded match surfaces.\n2. Add immutable Skills browse scope/result validation and shared pager display integration with fail-closed malformed response handling.\n3. Add a source-owned Skills browse controller for generations, requested/applied scopes, clamp, retry, stale retention, and unmount fencing.\n4. Wire the retained Skills canvas, exact pager, focus/restoration, broad-snapshot isolation, and off-page trust Review target.\n5. Route Skills mutations through fresh-page invalidation and committed-but-stale recovery with unsafe actions disabled.\n6. Run targeted service/state/controller/mounted/geometry/trust/mutation and isolated live verification, mutation-check the critical guards, then document evidence.\n\nADR required: no new ADR\nADR path: backlog/decisions/067-library-top-level-pagination-contracts.md\nReason: Implements the accepted Skills tranche without changing authority or architecture.\nDetailed plan: Docs/superpowers/plans/2026-08-29-task-18915-library-skills-pagination.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented exact 20-row local Skills paging with literal name/description filtering, deterministic name/status ordering, exact totals, and source-wide blocked recovery metadata. Added immutable browse contracts and a source-owned controller for generations, clamping, retries, stale retention, focus, restoration, and broad-snapshot isolation; wired the retained Skills canvas pager and all successful mutations through authoritative refresh. Fixed the compact-reader follow-up defect found live so an explicitly expanded 80-column Items pane survives ordinary layout syncs. ADR: existing backlog/decisions/067-library-top-level-pagination-contracts.md; no new ADR required. Verification: 72 service/state/controller tests, 167 Skills canvas/reader tests, 17 persistence/continue-receipt cases, and 29 CSS build/staleness tests passed; Ruff and git diff whitespace checks passed. Isolated 160x50, 120x35, 100x30, and 80x24 live walkthroughs covered pages 1-3, exact filtering and focus, pager containment, off-page Review, scratch-owned mutable handles, and byte-identical real config/index files. Both planned adversarial mutations failed their owning tests and were restored green. Updated lessons-live-verification.md with the compact pane persistence incident.
<!-- SECTION:NOTES:END -->
