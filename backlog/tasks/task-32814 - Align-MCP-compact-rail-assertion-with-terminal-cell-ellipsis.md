---
id: TASK-32814
title: Align MCP compact rail assertion with terminal-cell ellipsis
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-18 18:36'
updated_date: '2026-09-18 19:01'
labels:
  - ui
  - test-health
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the existing compact MCP reachability contract after TASK-32812 switched visible truncation to the terminal-cell ellipsis. The saved PR Fast Lane currently expects three ASCII dots.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The compact reachability case verifies the current visible ellipsis and passes with all existing content and reachability assertions intact.
- [x] #2 The failing CI evidence, targeted verification and unchanged product scope are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the saved PR Fast Lane failure showing the stale ASCII ellipsis assertion. 2. Match the terminal-cell ellipsis introduced by TASK-32812 while retaining the existing compact-content assertions. 3. Run the exact compact reachability case, review the small diff and save it in the draft PR. ADR required: no. ADR path: backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md. Reason: Test-only alignment with already implemented behavior; no product or architectural change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed only the compact rail test expectation from ASCII three-dot truncation to Rich's terminal-cell ellipsis introduced by TASK-32812. All existing content and reachability assertions remain intact. Saved-head Fast Lane evidence (1151 passed / one stale assertion failed) and the exact passing case are retained in Docs/superpowers/qa/2026-09-18-css-consolidation/. Independent review confirms the actual glyph contract; no production behavior changed. ADR required: no; ADR-150/161 already govern the implementation. Draft PR2707 remains unmerged; remote verification follows push.
<!-- SECTION:NOTES:END -->
