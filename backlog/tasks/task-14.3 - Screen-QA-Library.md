---
id: TASK-14.3
title: 'Screen QA: Library'
status: Done
assignee: []
created_date: '2026-05-09 03:46'
labels:
  - ux
  - screen-qa
  - product-maturity
  - screen-library
dependencies: []
references:
  - Docs/superpowers/qa/product-maturity/screen-qa/library/notes.md
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
Validate and correct the Library top-level destination screen through actual rendered screenshot QA. This task owns one focused PR for Library; do not claim approval from geometry dumps, SVG exports, mockups, or code layouts. Capture baseline and final screenshots from the running app, exercise one realistic interaction or blocked recovery path, and record explicit user approval before opening the PR.
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

1. Capture the current rendered Library screen through textual-web and browser automation.
2. Add failing mounted regressions for observed Library UX failures before code changes.
3. Implement the smallest safe Library screen fixes while preserving route IDs and existing services.
4. Rebuild generated TCSS from source modules and rerun focused visual/state tests.
5. Capture the final actual rendered screenshot and request explicit user approval before opening a PR.

## Implementation Notes

- Current polish candidate fixes Library snapshot loading so slow or blocking source services cannot leave the detail pane stuck on Loading.
- Empty local Library state now reports `Empty` with next-action copy instead of claiming `Ready`.
- Source Browser now exposes Collections directly and keeps Search/RAG inside the Library-native mode.
- The right pane now shows a generic `Inspector` empty state until a source/evidence item is selected; Search/RAG mode still swaps in the `Retrieval Inspector`.
- Verification passed for 26 focused Library visual/state, Library contract layout, and Gate 1.6 Search/RAG tests plus `git diff --check`.
- Final screenshot is captured at `Docs/superpowers/qa/product-maturity/screen-qa/library/review-2026-05-09-playwright-library-contextual-inspector.png`; user approval was recorded on 2026-05-09 before PR creation.
- PR #290 merged into `dev` on 2026-05-09 with merge commit `c9c4eeaafc1d094e7feafff51754f79a39c287a3`, closing the Library screen QA task.
