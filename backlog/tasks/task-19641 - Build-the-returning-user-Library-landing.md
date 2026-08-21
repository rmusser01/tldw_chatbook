---
id: TASK-19641
title: Build the returning-user Library landing
status: In Progress
assignee: []
created_date: '2026-08-21 21:49'
updated_date: '2026-08-21 21:59'
labels:
  - library
  - ux
  - landing
dependencies:
  - TASK-19022
references:
  - >-
    Docs/superpowers/specs/2026-08-20-library-lifecycle-progressive-disclosure-design.md
  - backlog/decisions/076-library-lifecycle-progressive-disclosure.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give populated Library profiles a truthful action-oriented landing that helps regular users resume work, recover current problems, and reach recent content without synchronous scans, fabricated state, or power-user regression.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At the wide breakpoint, a populated returning-user landing presents Continue from the last successfully applied Library route and full scope when that target remains valid.
- [ ] #2 A deleted or invalid Continue target falls back to the source valid scope, explains the adjustment, and never opens a fabricated selection.
- [ ] #3 Needs attention appears only for a current screen-owned recoverable failed import or stale state, provides its existing Review or Retry path, and is not implied to survive restart.
- [ ] #4 Recent sections use only existing trustworthy summaries, omit unavailable or unresolved data, and perform no synchronous scan, ranking, or new source read during composition.
- [ ] #5 Quick actions reuse the existing Import, New note, and Search routes with guarded semantic focus restoration and no duplicate router or action owner.
- [ ] #6 Compact presentation remains rail-first; wide and compact production-CSS tests prove truthful containment, keyboard order, focus preservation, and recovery at 170x48 and 100x30.
- [ ] #7 Only modified or touched landing components and direct owners are tested; no repository-wide pytest claim is made.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend the retained Library landing state/canvas with optional Continue and Needs attention sections while preserving existing Recent and quick-action owners.
2. Save a validated primitive Continue receipt separately from the active route through the runtime-scoped ScreenStateStore snapshot, restore existing full source scopes, and retain legacy/deep-link compatibility.
3. Dispatch Continue through existing guarded source routes, fail closed to valid source-list scope with adjustment copy, and preserve semantic focus.
4. Derive one current-screen recoverable failed-import or stale attention item without persistence, and route Review/Retry through existing owners.
5. Prove cached Recent/no-I/O composition and existing Import/New note/Search ownership.
6. Verify production-CSS behavior at 170x48 and 100x30, run only touched/direct-owner tests and exact Ruff/diff gates, update docs, review, and close through Backlog CLI.
<!-- SECTION:PLAN:END -->
