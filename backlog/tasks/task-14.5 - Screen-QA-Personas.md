---
id: TASK-14.5
title: 'Screen QA: Personas'
status: Done
assignee: []
created_date: '2026-05-09 03:46'
labels:
  - ux
  - screen-qa
  - product-maturity
  - screen-personas
dependencies: []
references:
  - Docs/superpowers/qa/product-maturity/screen-qa/personas/notes.md
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
Validate and correct the Personas top-level destination screen through actual rendered screenshot QA. This task owns one focused PR for Personas; do not claim approval from geometry dumps, SVG exports, mockups, or code layouts. Capture baseline and final screenshots from the running app, exercise one realistic interaction or blocked recovery path, and record explicit user approval before opening the PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Baseline actual screenshot captured
- [x] #2 Interaction smoke path exercised
- [x] #3 Final actual screenshot captured
- [x] #4 User approval recorded before PR
- [x] #5 Focused tests pass
- [x] #6 PR merged before next screen starts unless user explicitly overrides
<!-- AC:END -->

## Implementation Plan

1. Capture and keep the baseline actual Personas screenshot from the running app.
2. Add failing mounted regressions for the approved Personas IA: explicit mode strip, three named columns, local/server readiness, and inspector-owned actions.
3. Update the Personas screen minimally to match the approved Textual-native three-pane design while preserving existing service data and Console handoff behavior.
4. Run focused UI regressions plus diff hygiene.
5. Capture a final actual screenshot from the running app and obtain explicit user approval before PR work.

## Implementation Notes

Updated the Personas destination into an approved three-column workbench with explicit mode structure, full-height pane boundaries, and divider rails that can become future resize handles. Preserved existing local character/persona service data and Console handoff behavior while adding bounded-executor timeout coverage for stalled snapshot loads. Captured baseline and final actual textual-web screenshots under the Personas QA evidence folder; the final screenshot was approved in-session on 2026-05-09. Focused Personas UI regressions and `git diff --check` passed. PR #294 merged into `dev` as merge commit `842d21fb5438a66578b027f4b57e49f92ae077bc`.
