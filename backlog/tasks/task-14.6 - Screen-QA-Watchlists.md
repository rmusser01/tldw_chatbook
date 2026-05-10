---
id: TASK-14.6
title: 'Screen QA: Watchlists'
status: In Progress
assignee: []
created_date: '2026-05-09 03:46'
labels:
  - ux
  - screen-qa
  - product-maturity
  - screen-watchlists
dependencies: []
references:
  - Docs/superpowers/qa/product-maturity/screen-qa/watchlists/notes.md
documentation:
  - Docs/superpowers/plans/2026-05-08-12-screen-screenshot-qa-campaign.md
  - Docs/superpowers/specs/2026-05-08-12-screen-screenshot-qa-campaign-design.md
  - >-
    Docs/superpowers/specs/2026-05-08-destination-visual-parity-correction-design.md
parent_task_id: TASK-14
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate and correct the Watchlists top-level destination screen through actual rendered screenshot QA. This task owns one focused PR for Watchlists; do not claim approval from geometry dumps, SVG exports, mockups, or code layouts. Capture baseline and final screenshots from the running app, exercise one realistic interaction or blocked recovery path, and record explicit user approval before opening the PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Baseline actual screenshot captured
- [x] #2 Interaction smoke path exercised
- [x] #3 Final actual screenshot captured
- [x] #4 User approval recorded before PR
- [x] #5 Focused tests pass
- [ ] #6 PR merged before next screen starts unless user explicitly overrides
<!-- AC:END -->

## Implementation Plan

1. Capture the current Watchlists screen from the running app as the baseline screenshot.
2. Add failing mounted regressions for the approved Watchlists IA: filter strip, list/detail/status inspector columns, no Collections management copy, and Console stage/follow actions.
3. Update the Watchlists screen minimally to match the approved Textual-native control-plane layout while preserving existing service data and Console handoff behavior.
4. Exercise one realistic interaction or blocked recovery path, then run focused verification.
5. Capture a final actual screenshot from the running app and obtain explicit user approval before PR work.

## Implementation Notes

Converted Watchlists into the approved compact control-plane layout with a Watchlists-only header, filter strip, list/detail/status-inspector columns, and visible pane dividers for future resizing work. Added focused regressions for the approved column contract, removed user-facing Collections ownership copy from the Watchlists screen, and made unavailable local Watchlists services fail closed into a visible recovery state rather than an indefinite loading state. Captured actual textual-web baseline and final screenshots and recorded user approval before PR creation.
