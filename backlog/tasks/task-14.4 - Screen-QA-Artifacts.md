---
id: TASK-14.4
title: 'Screen QA: Artifacts'
status: In Progress
assignee: []
created_date: '2026-05-09 03:46'
labels:
  - ux
  - screen-qa
  - product-maturity
  - screen-artifacts
dependencies: []
references:
  - Docs/superpowers/qa/product-maturity/screen-qa/artifacts/notes.md
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
Validate and correct the Artifacts top-level destination screen through actual rendered screenshot QA. This task owns one focused PR for Artifacts; do not claim approval from geometry dumps, SVG exports, mockups, or code layouts. Capture baseline and final screenshots from the running app, exercise one realistic interaction or blocked recovery path, and record explicit user approval before opening the PR.
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

1. Capture the current Artifacts screen from the running app as the baseline screenshot.
2. Add failing mounted regressions for the approved Artifacts IA: type/filter strip, list/preview/provenance panes, and empty-state recovery actions.
3. Implement the smallest safe screen changes that preserve existing routes and Chatbook launch behavior while making all artifact types visible.
4. Exercise one realistic interaction or blocked recovery path, then run focused verification.
5. Capture a final actual screenshot and request explicit user approval before opening the PR.

## Implementation Notes

- Captured the baseline Artifacts screen from the running app with textual-web browser automation.
- Reworked the empty Artifacts shell into three explicit columns: artifact list, preview/detail, and provenance.
- Added mounted regression coverage for the Artifacts IA and explicit column labels.
- Captured the approved final screenshot at `Docs/superpowers/qa/product-maturity/screen-qa/artifacts/final-2026-05-09-artifacts-columns.png`.
- Focused verification passed with `Tests/UI/test_destination_visual_parity_correction.py -k "artifacts"`.
