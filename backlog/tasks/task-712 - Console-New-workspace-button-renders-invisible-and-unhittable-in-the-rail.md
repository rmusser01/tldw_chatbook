---
id: TASK-712
title: Console New-workspace button renders invisible and unhittable in the rail
status: Done
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - console
  - workspaces
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Session action row's 12-column left margin plus two 16-column-minimum buttons overflow the ~37-column Console rail, so the New button's label renders entirely outside the clip while a blank strip stays clickable (live-verified: sweep-clicking blank space created a workspace). The only Console affordance for creating a workspace is invisible, while the adjacent copy tells users to add another workspace. The comment at console_workspace_context.py:751-766 documents this exact overflow failure mode for a third button; the margin re-broke the original pair. Finding C1.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both Switch and New are fully visible and clickable at the rail's real width
- [x] #2 A regression test asserts both action buttons' regions fit within the rail clip
- [x] #3 No invisible clickable region remains in the Session action row
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red geometry test mounting real ChatScreen with the shipped bundle CSS (StyledConsoleHarness) asserting each action button fits its row's content region.
2. Fix the app-tier CSS rule; rebuild the bundle; green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: Textual's default Button min-width (16) + the action row's 12-col indent left a 20-col row where Switch consumed everything and New's label rendered past the rail clip (region probe: New at x=35 w=16 in a row ending at 38). Fix: `.console-workspace-action { min-width: 5; width: auto; }` in `css/components/_agentic_terminal.tcss` (bundle rebuilt via build_css.py) so compact actions size to their labels; Switch+New+RAG Scope all fit with room to spare. Regression guard: `Tests/UI/test_console_workspace_action_row_geometry.py` — two tests, one asserting no button overflows its row/tray clip (this is the invisible-but-clickable failure mode), one asserting Switch and New coexist side by side. 93 adjacent rail tests pass.
<!-- SECTION:NOTES:END -->
