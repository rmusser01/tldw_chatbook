---
id: TASK-14.10
title: 'Screen QA: ACP'
status: In Progress
assignee: []
created_date: '2026-05-09 03:46'
labels:
  - ux
  - screen-qa
  - product-maturity
  - screen-acp
dependencies: []
references:
  - Docs/superpowers/qa/product-maturity/screen-qa/acp/notes.md
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
Validate and correct the ACP top-level destination screen through actual rendered screenshot QA. This task owns one focused PR for ACP; do not claim approval from geometry dumps, SVG exports, mockups, or code layouts. Capture baseline and final screenshots from the running app, exercise one realistic interaction or blocked recovery path, and record explicit user approval before opening the PR.
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

1. Capture the actual rendered ACP baseline from textual-web with an isolated profile.
2. Add a failing mounted regression for the approved ACP blocked-runtime three-column contract.
3. Rework ACP into explicit Agents/Sessions, Runtime Setup, and Compatibility/Actions columns while preserving disabled launch/follow recovery.
4. Capture the actual rendered final ACP screenshot and record user approval before PR creation.
5. Run focused ACP and destination shell verification.

## Implementation Notes

- Captured ACP baseline and final PNG evidence through textual-web/Playwright and recorded approval for `final-2026-05-10-acp-columns.png`.
- Updated ACP from a weak runtime-unconfigured row into a compact three-column shell with explicit pane titles and dividers.
- Kept ACP runtime launch/follow honestly blocked while moving runtime setup ownership copy into ACP rather than Settings.
- Added a mounted regression that verifies the visible ACP runtime setup and compatibility column contract.
