---
id: TASK-14.8
title: 'Screen QA: Workflows'
status: In Progress
assignee: []
created_date: '2026-05-09 03:46'
labels:
  - ux
  - screen-qa
  - product-maturity
  - screen-workflows
dependencies: []
references:
  - Docs/superpowers/qa/product-maturity/screen-qa/workflows/notes.md
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
Validate and correct the Workflows top-level destination screen through actual rendered screenshot QA. This task owns one focused PR for Workflows; do not claim approval from geometry dumps, SVG exports, mockups, or code layouts. Capture baseline and final screenshots from the running app, exercise one realistic interaction or blocked recovery path, and record explicit user approval before opening the PR.
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
1. Start from current origin/dev in a clean Workflows screen-QA branch.
2. Capture the baseline actual Workflows screenshot from the running app and record visible defects in Workflows QA notes.
3. Add focused mounted regressions for the approved Workflows column shell and blocked/console recovery behavior before changing UI code.
4. Apply the smallest Textual-native layout corrections needed to make Workflows usable and consistent with the approved destination patterns.
5. Rebuild generated CSS when TCSS changes, run focused Workflows verification, capture the final actual screenshot, and request explicit user approval before opening a PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
- Captured actual textual-web baseline and final Workflows screenshots with an isolated runtime profile.
- Converted Workflows to the approved destination workbench layout with compact header, mode strip, full-height Procedure Library / Run Detail / Run Inspector columns, divider rails, blocked Console state, next action, and approval/status summaries.
- Added regression coverage for the Workflows visual contract and for active workflow status propagation into the inspector.
- Updated generated modular CSS from the edited TCSS module.
<!-- SECTION:NOTES:END -->
