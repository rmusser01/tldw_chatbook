---
id: TASK-149
title: Polish Console rail layout and sticky header
status: Done
assignee: []
created_date: '2026-07-01 01:18'
updated_date: '2026-07-01 01:44'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improve the Console left-rail UX shown in the setup-blocked screen: reduce the loud focused boundary, let rail content consume the available vertical space, and keep the top blocker/control region visually solid while scrolling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused conversation browser does not render a large high-saturation blue boundary box.
- [x] #2 Console left rail uses available vertical space instead of leaving avoidable empty panel space.
- [x] #3 Top setup-blocked and control rows remain solid/opaque and do not visually collide with scrolled content.
- [x] #4 Targeted UI tests cover the layout/focus regressions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: scoped UI layout and focus styling polish within existing Console/Textual contracts.

1. Add failing CSS and mounted geometry tests for single-scroll rail behavior and quiet focus framing.
2. Convert the nested conversation-browser list to a normal vertical region so the left rail body owns overflow.
3. Add explicit TCSS focus/background rules for the left rail, workspace context, conversation browser, recovery/control rows, and transcript empty panel.
4. Rebuild modular CSS and run focused Console rail tests.
5. Capture rendered screenshot evidence and update task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Reworked the Console workspace context rail so the left rail body is the scroll owner. The workspace tray now fits to measured content while keeping short states at least as tall as the remaining rail viewport.
- Replaced the nested scrollable conversation list with deterministic full-content list heights, preventing the conversation browser from trapping scroll or drawing the loud blue focus boundary.
- Added quiet focus rules for the left rail, workspace tray, and conversation list; kept setup-blocked/control/transcript/composer surfaces fully opaque; expanded the transcript empty panel to fill its available region.
- Updated generated modular CSS and targeted UI tests for rail geometry, focus styling, and persistent rail stylesheet regressions.
- ADR required: no. This is scoped Textual layout and styling polish within existing Console UI contracts.
- Verification: `pytest` affected Console UI suite passed, `62 passed, 1 warning`; `git diff --check` passed.
- Visual evidence: `/tmp/console-rail-layout-polish-initial.png` and `/tmp/console-rail-layout-polish-scrolled.png`.
<!-- SECTION:NOTES:END -->
