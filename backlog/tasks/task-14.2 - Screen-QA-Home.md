---
id: TASK-14.2
title: 'Screen QA: Home'
status: In Progress
assignee: []
created_date: '2026-05-09 03:46'
updated_date: '2026-05-09 17:03'
labels:
  - ux
  - screen-qa
  - product-maturity
  - screen-home
dependencies: []
references:
  - Docs/superpowers/qa/product-maturity/screen-qa/home/notes.md
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
Validate and correct the Home top-level destination screen through actual rendered screenshot QA. This task owns one focused PR for Home; do not claim approval from geometry dumps, SVG exports, mockups, or code layouts. Capture baseline and final screenshots from the running app, exercise one realistic interaction or blocked recovery path, and record explicit user approval before opening the PR.
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
1. Confirm latest dev includes the merged Console QA branch and close Console task bookkeeping.
2. Run focused baseline Home/destination verification before visual edits.
3. Launch the actual app through textual-web/textual-serve and capture the Home baseline screenshot after the splash clears.
4. Exercise one Home primary path from the attention queue or next-best action into selection/control/status behavior.
5. Record screenshot-backed defects in Home QA notes and implement only minimal Home-focused fixes.
6. Recapture an actual final Home screenshot and request explicit user approval before opening the PR.
7. Run focused Home verification, update task evidence, then commit and open the Home-only PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR #288 opened against dev for the approved Home screenshot QA pass. AC #6 remains open until PR merge is confirmed.
<!-- SECTION:NOTES:END -->
