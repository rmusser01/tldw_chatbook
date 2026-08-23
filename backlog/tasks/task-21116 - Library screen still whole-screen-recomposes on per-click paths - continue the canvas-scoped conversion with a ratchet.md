---
id: TASK-21116
title: >-
  Library screen still whole-screen-recomposes on per-click paths - continue the canvas-scoped conversion with a ratchet
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - library
  - recompose
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21116).

library_screen.py holds ~105 statement-level `refresh(recompose=True)` sites (99 on self =
whole-screen) on a screen that grew 26k -> 34.8k lines since the audit pin. Confirmed per-click
sites: `_open_library_item_by_id` (rail row / RAG result / media open), `_apply_library_row_toggle`,
media-viewer back (Escape), skills/prompts import open/cancel, export open. The 15457
canvas-scoped seam (`library_canvas_sync`, 82 call sites) is the sanctioned conversion target;
9 whole-screen sites were re-added post-fix on low-frequency admin flows.

## Acceptance Criteria

- [ ] The confirmed per-click sites above no longer rebuild the whole screen (converted to the canvas-scoped seam or narrower)
- [ ] A ratchet test pins the statement-level whole-screen recompose count at (or below) the post-conversion number
- [ ] Click-to-settle timing on a rail-row open, before/after, recorded in the task
