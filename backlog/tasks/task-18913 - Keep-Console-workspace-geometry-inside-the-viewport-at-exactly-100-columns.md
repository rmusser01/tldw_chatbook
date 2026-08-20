---
id: TASK-18913
title: Keep Console workspace geometry inside the viewport at exactly 100 columns
status: In Progress
assignee: []
created_date: '2026-08-20 07:07'
labels:
  - console
  - ux
dependencies:
  - TASK-18912
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent the reproduced exact-100-column Console rail state from expanding the grid far beyond the viewport so every supported persona retains a visible transcript and usable rail controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 At exactly 100x30, all four stored Context/Inspector open-state combinations keep every displayed Console workspace child within the viewport; the default Context-only state gives `#console-main-column` at least 55 allocated columns and every state gives `#console-native-transcript` at least 40 readable content columns excluding borders and padding.
- [ ] #2 Effective rail priority, compact overrides, responsive focus handoff, selected-message and transcript-anchor state, stored preferences, and the 70/74-column usable-transcript floors remain consistent with ADR-043; cold start and responsive resize make no preference-save call.
- [ ] #3 Existing geometry and access behavior at 80x24, 120x30, 160x45, and 235x52 do not regress.
- [ ] #4 Production-CSS Textual compositor regressions prove the exact-100 cold-start failure and the 101→100 live-resize failure, cover both directions across 99/100/101, and are mutation-checked independently against the inclusive override and exact-100 resize-boundary corrections.
- [ ] #5 The change is terminal-specific and does not alter TASK-18911's phone, pointer, hover, or served-browser ownership.
<!-- AC:END -->

## Design

<!-- SECTION:DESIGN:BEGIN -->
Approved design: `Docs/superpowers/specs/2026-08-20-task-18913-console-exact-100-column-containment-design.md`.
<!-- SECTION:DESIGN:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing pure-state tests for the 99/100/101 minimum-waiver and resize-band boundaries.
2. Implement the inclusive exact-100 waiver and distinct semantic resize band in `console_rail_state.py`.
3. Add production-CSS four-state cold geometry coverage with separate 55-column allocation and 40-column readable-content assertions.
4. Add all four adjacent live-resize directions with focus, selection, anchor, and zero-preference-save assertions.
5. Correct stale width arithmetic and compact-override comments without changing other rail behavior.
6. Mutation-check both corrections independently, run focused/full verification, and record evidence before closing the task.

Detailed plan: `Docs/superpowers/plans/2026-08-20-task-18913-console-exact-100-column-containment.md`.

ADR required: no

ADR path: `backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md`

Reason: direct defect correction within ADR-043's existing minimum-width-waiver and responsive priority mechanism; no persistence or application-structure decision changes.
<!-- SECTION:PLAN:END -->
