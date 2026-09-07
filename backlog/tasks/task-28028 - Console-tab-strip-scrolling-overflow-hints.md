---
id: TASK-28028
title: Console tab strip scrolling + overflow hints; complete rail-height seam
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-02 05:09'
updated_date: '2026-09-02 14:40'
labels:
  - console
  - bug
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tabs pushed off-screen in the Console session tab strip are unreachable and invisible: a plain mouse wheel does nothing (only undiscoverable shift+wheel scrolls), and nothing signals hidden tabs. Separately, uncommitted adaptive-row-limit WIP calls screen._console_rail_body_height() which does not exist, crashing the Console on every screen resume.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plain vertical mouse wheel over the tab strip scrolls it horizontally (up=left, down=right) and stops the event only when it moved
- [x] #2 Edge indicators appear on a side only when tabs are hidden beyond that side, and do not shift layout
- [x] #3 Auto-scroll-to-active and Alt+1..9 behavior unchanged
- [x] #4 ChatScreen._console_rail_body_height returns the measured #console-left-rail-body height or None when unmeasured, so the Console boots and screen-resume no longer crashes
- [x] #5 Live tmux pass: app boots on a scratch profile, Ctrl+K switcher opens and switches sessions, wheel scrolls the strip
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (UI affordance + completing an existing wiring seam; no storage/provider/security decisions). 1. Add ChatScreen._console_rail_body_height next to _console_conversation_browser_collapse_preferences. 2. Subclass HorizontalScroll as ConsoleSessionTabStrip overriding wheel handlers. 3. Flank strip with 1-cell Static indicators toggled on scroll. 4. RED/GREEN tests in Tests/UI/test_console_session_tab_strip.py. 5. Live tmux verification with scratch profile.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Shipped against `dev`, where the adaptive rail-height seam the original tree was missing had already landed as `be1ee8418` (feat: scale Console conversation rail cap to available height) -- so this PR's scope is the tab-strip affordances only: `ConsoleSessionTabStrip` (plain vertical wheel -> horizontal scroll, 8 cells/step, event stopped only when it moved), the 1-row strip row with flanking `#console-tab-overflow-left/right` Statics (display-toggled via `watch_scroll_x`/resize/sync hooks; always mounted so no layout shift), the `_agentic_terminal.tcss` hint rule (deterministic bundle regen), and 5 new tests in `Tests/UI/test_console_session_tab_strip.py` (17 green). The rail-height seam work documented in the plan/ACs happened on the `docs/lesson-adr-number-collisions` working tree and is superseded by dev's own landing; live tmux verification (boot, 10 tabs, both hints tracking, wheel both directions, Ctrl+K switcher opens/lists/switches) ran against that tree with identical widget code (the surface file is byte-identical between the two bases).

## Renumbering provenance

Originally created as TASK-28025 on 2026-09-02; renumbered to TASK-28028 before merge
because parallel Library-media PRs landed on `dev` first carrying
task-28025/task-28027 for different
work. Per the TASK-19601 owner rule (older arrival keeps the id), the
landed tasks keep those numbers.
