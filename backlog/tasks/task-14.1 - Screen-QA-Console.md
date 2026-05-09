---
id: TASK-14.1
title: 'Screen QA: Console'
status: In Progress
assignee: []
created_date: '2026-05-09 03:46'
updated_date: '2026-05-09 04:16'
labels:
  - ux
  - screen-qa
  - product-maturity
  - screen-console
dependencies: []
references:
  - Docs/superpowers/qa/product-maturity/screen-qa/console/notes.md
documentation:
  - Docs/superpowers/plans/2026-05-08-12-screen-screenshot-qa-campaign.md
  - Docs/superpowers/specs/2026-05-08-12-screen-screenshot-qa-campaign-design.md
  - >-
    Docs/superpowers/specs/2026-05-08-destination-visual-parity-correction-design.md
parent_task_id: TASK-14
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate and correct the Console top-level destination screen through actual rendered screenshot QA. This task owns one focused PR for Console; do not claim approval from geometry dumps, SVG exports, mockups, or code layouts. Capture baseline and final screenshots from the running app, exercise one realistic interaction or blocked recovery path, and record explicit user approval before opening the PR.
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

<!-- SECTION:PLAN:BEGIN -->
1. Confirm latest dev includes the screen QA scaffold and read the Console task/evidence template.
2. Run a baseline mounted verification slice for Console and destination layout regressions.
3. Launch the app through textual-serve/textual-web when available and capture an actual rendered Console screenshot.
4. Exercise the Console composer/primary controls enough to verify typed text visibility and blocked recovery behavior.
5. Record baseline defects in the Console notes file, implement only evidence-backed fixes, then recapture a final actual screenshot for user approval before PR creation.
6. Run focused Console/UI verification and update TASK-14.1 only after explicit screenshot approval.
<!-- SECTION:PLAN:END -->
