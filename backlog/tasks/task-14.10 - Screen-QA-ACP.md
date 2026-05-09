---
id: TASK-14.10
title: 'Screen QA: ACP'
status: To Do
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
- [ ] #1 Baseline actual screenshot captured
- [ ] #2 Interaction smoke path exercised
- [ ] #3 Final actual screenshot captured
- [ ] #4 User approval recorded before PR
- [ ] #5 Focused tests pass
- [ ] #6 PR merged before next screen starts unless user explicitly overrides
<!-- AC:END -->
