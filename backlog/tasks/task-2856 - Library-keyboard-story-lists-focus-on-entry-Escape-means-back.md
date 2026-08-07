---
id: TASK-2856
title: 'Library keyboard story: lists focus on entry, Escape means back'
status: To Do
assignee: []
created_date: '2026-08-07 01:10'
labels:
  - library
  - keyboard
  - accessibility
  - uat-2026-08-06
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 (LIB-08, prior-critique P1 now measured worse; A + B evidence at dev
`6ffa56516`).

Measured: from a fresh Library landing, the rail search box is 14 Tab stops away and the first
canvas control is 36 (Tabs 1–12 walk the top nav; 13–35 walk the entire rail). Up/Down never move
the media-list selection (7/7 checks, including directly after ‹ Back — the list is not focused).
Escape never functions as back in any detail view. "‹ Back to list" is mouse-only. Focus is
visible at most stops (bg + bold + underline), but two stops are provably invisible (Tab#35
released focus with nothing gaining it; Tab#40 produced a byte-identical capture) and the media
viewer's Author input never shows focus styling.

Keyboard-first is the product's first principle; the destination most users land on is its
slowest keyboard surface. Related open task-2520 covers the landing FOOTER advertisement; this
task covers the mechanics themselves.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Entering a list canvas (Media, Notes, Prompts, Skills) focuses its primary list; Up/Down move the selection and Enter opens it
- [ ] #2 Escape returns from detail/viewer surfaces to their list, and from a list canvas focus back toward the rail (no-op only where there is genuinely nothing to leave)
- [ ] #3 A direct rail-focus accelerator exists and is advertised (footer or F1), cutting the 14/36-Tab traversal
- [ ] #4 Every Tab stop in the Library screen produces a visible focus change (the two invisible stops and the Author input are fixed), proven by ANSI-attribute assertions, not "something changed"
- [ ] #5 Live keyboard-only walkthrough: landing → Media list → item → back → search, without touching the mouse
<!-- AC:END -->
