---
id: TASK-21114
title: >-
  Transcript drag-selection re-wraps and re-renders the message body on every mouse-move
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - console
  - transcript
priority: high
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21114).

During an active drag (MouseMove at 50-100 Hz), `console_transcript.py` per event: rebuilds the
full message render text for plain rows (uncached `get_display_text()`, :1727-1729), runs
`Content(text).wrap(width)` over the entire body (:2278-2311), calls `set_selection_range` ->
full `body.update()` re-render even when the range did not move (:1740-1787), and sweeps every
mounted row calling `clear_selection()` (:5055-5063; watermarks default 20k/12k lines so
hundreds of rows can be mounted). Tens of ms per event on multi-KB rows - seconds of cumulative
lag per drag on slow hardware. The whole feature is post-baseline.

## Acceptance Criteria

- [ ] Plain-row display text is cached and invalidated in sync_message; the wrap table is memoized per (text, width) for the drag's lifetime
- [ ] set_selection_range early-returns on an unchanged range; the drag remembers the single row holding a selection instead of sweeping all mounted rows
- [ ] A counter/timing probe over a drag across a ~20 KB row shows the per-event cost drop; numbers in the task
- [ ] Selection behavior (visuals, copy, menus) unchanged - existing selection tests green
