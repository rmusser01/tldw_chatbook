---
id: TASK-65
title: Redesign master shell navigation tabs as terminal tabs
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the top-level navigation match the terminal-native shell by replacing dot-separated navigation pills with bordered ASCII-style tabs while preserving destination labels, IDs, tooltips, active state, and route behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Top-level navigation renders existing destination buttons as terminal-style tabs with a stable ASCII tab class.
- [x] #2 Dot separator widgets are no longer mounted in the top navigation rail.
- [x] #3 Existing destination order, labels, IDs, tooltips, active state, and click routing remain intact.
- [x] #4 Focused navigation tests and diff checks pass.
- [x] #5 An actual rendered screenshot is captured for visual approval.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a mounted regression that verifies top navigation buttons expose a stable ASCII tab class and no dot separator widgets are mounted.
2. Update MainNavigationBar styling and composition so the existing NavigationButton widgets render as bordered terminal tabs while preserving destination labels, IDs, tooltips, active state, and routing.
3. Run the focused navigation regression suite and diff checks.
4. Capture an actual Textual-web screenshot of the rendered Console navigation for approval before PR packaging.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a mounted regression for the top navigation terminal-tab rail.
- Updated `MainNavigationBar` to render the existing `NavigationButton` destinations as bordered ASCII-style tabs while preserving labels, route IDs, tooltips, active state, and click routing.
- Removed the mounted dot separator widgets from the top navigation rail.
- Captured and approved the Textual-web screenshot at `Docs/superpowers/qa/console-ui/2026-05-21-console-ascii-tabs.png`.
- Addressed PR review feedback by removing the now-dead `.nav-separator` CSS rule and using the standard `$primary-darken-1` active-tab background token.
- Captured post-review Textual-web evidence at `Docs/superpowers/qa/console-ui/2026-05-21-console-ascii-tabs-review-fixes.png`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The master shell top navigation now uses terminal-style bordered tabs instead of dot-separated navigation pills, with focused tests covering the structural contract and existing route behavior preserved.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
