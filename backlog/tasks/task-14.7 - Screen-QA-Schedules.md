---
id: TASK-14.7
title: 'Screen QA: Schedules'
status: Done
assignee: []
created_date: '2026-05-09 03:46'
labels:
  - ux
  - screen-qa
  - product-maturity
  - screen-schedules
dependencies: []
references:
  - Docs/superpowers/qa/product-maturity/screen-qa/schedules/notes.md
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
Validate and correct the Schedules top-level destination screen through actual rendered screenshot QA. This task owns one focused PR for Schedules; do not claim approval from geometry dumps, SVG exports, mockups, or code layouts. Capture baseline and final screenshots from the running app, exercise one realistic interaction or blocked recovery path, and record explicit user approval before opening the PR.
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

1. Capture the current Schedules screen from the running app as the baseline screenshot.
2. Add a failing mounted regression for the approved Schedules IA: compact status header, list/detail/status inspector columns, state/retry/next-action copy, and explicit future-resizable dividers.
3. Update the Schedules screen minimally to match the approved Textual-native control-plane layout while preserving existing Console follow and reading-digest launch behavior.
4. Rebuild generated modular CSS from source TCSS and run focused verification.
5. Capture a final actual screenshot from the running app and obtain explicit user approval before PR work.

## Implementation Notes

Converted Schedules into the approved compact control-plane layout with a schedule queue, run detail/output pane, status inspector, and visible pane dividers for future resizing work. Added a focused visual-parity regression for the approved column contract, preserved the existing Console follow and reading-digest launch paths, and documented the empty/blocked recovery path as the approved screenshot state. Captured actual textual-web baseline and final screenshots and recorded user approval before PR creation.

PR #297 merged on 2026-05-10 as merge commit `e7ad1f623382b3f2a687a549ea216a363c4dde1d`. Follow-up PR #298 merged on 2026-05-10 as merge commit `45371878e8ef8b7eff69d2b1cee7fa5e29fdc0da`.
