---
id: TASK-2762
title: 'Home rail and canvas cannot scroll; overflow work is unreachable and invisible'
status: To Do
assignee: []
created_date: '2026-08-06'
labels: [home, bug, ui]
dependencies: []
---
## Description (the why)

`HomeRail` and `HomeCanvas` subclass `Vertical` (Textual default
`overflow: hidden hidden`) and no CSS overrides it — runtime-verified:
`overflow_y=hidden`, `allow_vertical_scroll=False`,
`show_vertical_scrollbar=False` on `#home-rail` and `#home-canvas`
(dev @ 84e4b33f0, 2026-08-06). Contrast `#library-rail`, which sets
`overflow-y: auto` explicitly.

Capacity is real: Needs Attention/Running can hold up to 20 watchlist runs
plus every queued/parsing/writing/failed Library ingest job, and Recent holds
8 — each row 2 cells tall plus a 1-cell margin. Rows past the fold get no
scrollbar, no overflow cue, and — because Home declares no key bindings at
all — no keyboard path either. Work simply becomes invisible.

Documented as a Quirk in `Docs/User_Guide/home.md`.

## Acceptance Criteria (the what)

- [ ] The rail (and the canvas, if it can overflow) scrolls when content
      exceeds the viewport, with a visible scrollbar or overflow cue.
- [ ] Every rail row is reachable at 24-row terminal height with a full
      attention list.
- [ ] A test pins the scrollability (or an explicit overflow indicator).
