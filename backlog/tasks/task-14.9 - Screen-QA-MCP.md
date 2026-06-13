---
id: TASK-14.9
title: 'Screen QA: MCP'
status: Done
assignee: []
created_date: '2026-05-09 03:46'
updated_date: '2026-05-10 15:55'
labels:
  - ux
  - screen-qa
  - product-maturity
  - screen-mcp
dependencies: []
references:
  - Docs/superpowers/qa/product-maturity/screen-qa/mcp/notes.md
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
Validate and correct the MCP top-level destination screen through actual rendered screenshot QA. This task owns one focused PR for MCP; do not claim approval from geometry dumps, SVG exports, mockups, or code layouts. Capture baseline and final screenshots from the running app, exercise one realistic interaction or blocked recovery path, and record explicit user approval before opening the PR.
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

<!-- SECTION:PLAN:BEGIN -->
1. Start from current origin/dev in a clean MCP screen-QA branch.
2. Capture the baseline actual MCP screenshot from textual-web with an isolated runtime profile.
3. Add focused regression coverage for compact MCP title/purpose/mode strips and full-height workbench geometry.
4. Extend the existing destination screen TCSS selector groups to include MCP without changing runtime behavior.
5. Regenerate modular CSS, run focused MCP verification, capture the final actual screenshot, and request explicit user approval before opening a PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured actual textual-web baseline and final MCP screenshots with an isolated runtime profile.
- Corrected MCP destination shell styling so title, purpose, and mode strips render compactly and the Servers And Scope / Server Detail / Readiness And Actions workbench uses the available viewport height.
- Added a focused visual parity regression for MCP compact shell rows and taller workbench panes.
- Regenerated the modular TCSS bundle from the source TCSS module.
- PR #301 merged into dev; follow-up consolidated redundant MCP pane selectors raised during review.
<!-- SECTION:NOTES:END -->
