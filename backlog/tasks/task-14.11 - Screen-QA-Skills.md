---
id: TASK-14.11
title: 'Screen QA: Skills'
status: Done
assignee: []
created_date: '2026-05-09 03:46'
labels:
  - ux
  - screen-qa
  - product-maturity
  - screen-skills
dependencies: []
references:
  - Docs/superpowers/qa/product-maturity/screen-qa/skills/notes.md
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
Validate and correct the Skills top-level destination screen through actual rendered screenshot QA. This task owns one focused PR for Skills; do not claim approval from geometry dumps, SVG exports, mockups, or code layouts. Capture baseline and final screenshots from the running app, exercise one realistic interaction or blocked recovery path, and record explicit user approval before opening the PR.
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

1. Capture the current Skills screen from the running app as the baseline screenshot.
2. Exercise a realistic Skills workflow: inspect installed/empty/local state and verify Console attach/import disabled recovery.
3. Add or update a focused mounted regression for the approved Skills IA if current coverage does not catch the visual defect.
4. Update the Skills screen minimally to match the approved Textual-native column pattern while preserving existing local skills listing and Console handoff behavior.
5. Rebuild generated modular CSS if TCSS changes, run focused verification, then capture the final actual screenshot for user approval before opening a PR.

## Implementation Notes

Adapted the Skills destination to the approved compact three-column workbench pattern and recorded baseline/final actual textual-web screenshots. Added focused mounted coverage for the Skills column contract and empty-state behavior, and moved local skills discovery to a thread worker so the live textual-web screen resolves out of the loading state. Import remains intentionally disabled because the shell does not yet wire external Agent Skills import; the inspector now presents that state explicitly.

PR #305 was merged into `dev` at merge commit `3bb8f36ab6f9440a85f519f854182bbd9e099bac`.
