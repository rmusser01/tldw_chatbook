---
id: TASK-16314
title: Page Library Skills with source-wide trust recovery
status: To Do
assignee: []
created_date: '2026-08-15 02:48'
labels:
  - library
  - pagination
  - skills
  - trust
dependencies:
  - TASK-16311
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
- [ ] #1 Skill name and description filtering, trust/status sorting, and deterministic identity ordering occur before 20-row slicing with an exact filtered total.
- [ ] #2 Filtering never matches Skill bodies, supporting files, argument hints, metadata keys, or trust diagnostics.
- [ ] #3 Each page response carries the source-wide blocked total and first blocked Skill name independently of the current page or active filter.
- [ ] #4 Trust Review opens the stable blocked target directly even when it is off-page, and the trust header never reports a false page-local zero.
- [ ] #5 Malformed page rows or metadata fail closed; stale mutation recovery suppresses inexact totals and disables unsafe row or pager actions.
- [ ] #6 Skill request generations, unmount fencing, broad-snapshot isolation, focus, restoration, and recoverable errors match the approved design.
- [ ] #7 Automated local-service/state, mounted Textual, geometry, trust, mutation, and isolated live verification pass.
<!-- AC:END -->
