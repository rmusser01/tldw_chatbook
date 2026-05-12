---
id: TASK-14.12
title: 'Screen QA: Settings'
status: In Progress
assignee: []
created_date: '2026-05-09 03:46'
labels:
  - ux
  - screen-qa
  - product-maturity
  - screen-settings
dependencies: []
references:
  - Docs/superpowers/qa/product-maturity/screen-qa/settings/notes.md
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
Validate and correct the Settings top-level destination screen through actual rendered screenshot QA. This task owns one focused PR for Settings; do not claim approval from geometry dumps, SVG exports, mockups, or code layouts. Capture baseline and final screenshots from the running app, exercise one realistic interaction or blocked recovery path, and record explicit user approval before opening the PR.
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

1. Capture the current Settings screen from the running app as the baseline screenshot.
2. Exercise a realistic Settings workflow: inspect global preferences, verify Appearance routing, and verify the Console large-paste preference can persist.
3. Add focused mounted regressions for the approved Settings shell contract and the large-paste preference control.
4. Update the Settings screen minimally to match the approved Textual-native three-column pattern while keeping runtime-specific MCP and ACP controls out of global Settings.
5. Rebuild generated modular CSS from the source TCSS, run focused verification, then capture the final actual screenshot for user approval before opening a PR.

## Implementation Notes

Adapted Settings to a compact three-column workbench with a narrower settings-section column and wider preference-detail and scope-inspector columns. Replaced the unreadable checkbox glyph with a readable button toggle for Console large-paste display, preserved the persisted `console.collapse_large_pastes` setting, and kept MCP/ACP runtime control ownership explicit. Added mounted regressions for the Settings column contract, narrow-left column geometry, Appearance routing, and large-paste toggle persistence. Final actual textual-web screenshot was approved before PR creation.
