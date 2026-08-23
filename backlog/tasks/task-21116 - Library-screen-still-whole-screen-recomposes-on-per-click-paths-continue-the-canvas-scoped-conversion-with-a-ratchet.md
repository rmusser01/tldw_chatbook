---
id: TASK-21116
title: >-
  Library screen still whole-screen-recomposes on per-click paths - continue the
  canvas-scoped conversion with a ratchet
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 11:51'
labels:
  - performance
  - library
  - recompose
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21116).

library_screen.py holds ~105 statement-level `refresh(recompose=True)` sites (99 on self =
whole-screen) on a screen that grew 26k -> 34.8k lines since the audit pin. Confirmed per-click
sites: `_open_library_item_by_id` (rail row / RAG result / media open), `_apply_library_row_toggle`,
media-viewer back (Escape), skills/prompts import open/cancel, export open. The 15457
canvas-scoped seam (`library_canvas_sync`, 82 call sites) is the sanctioned conversion target;
9 whole-screen sites were re-added post-fix on low-frequency admin flows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The confirmed per-click sites above no longer rebuild the whole screen (converted to the canvas-scoped seam or narrower)
- [ ] #2 A ratchet test pins the statement-level whole-screen recompose count at (or below) the post-conversion number
- [ ] #3 Click-to-settle timing on a rail-row open, before/after, recorded in the task
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline: teed runs of the Library suites covering the touched flows (canvas sync, selection updates, multiselect media/notes, media trash/side-by-side, prompts/skills canvases, export, RAG handoffs, shell) at base 30c7e1fe9.\n2. Probe each confirmed per-click site on the mounted harness (screen-recompose spy) to confirm which actually whole-screen-recompose per click, incl. whether _apply_library_row_toggle's in-place patch holds or falls back.\n3. Convert site by site: skills/prompts import open/cancel -> _sync_library_canvas(kind, then=focus); media-viewer sub-state Escape branches -> viewer-scoped _sync_library_media_viewer_state; media viewer exit + media open + _open_library_item_by_id non-entry paths + export open -> canvas-host child replacement via the strict _replace_library_canvas_child seam (+ explicit _register_footer_shortcuts), whole-screen recompose kept only as failure fallback; _refresh_library_media_detail completion -> targeted child swap.\n4. Red-first AST ratchet test pinning the statement-level whole-screen recompose count (fails on the pre-conversion count; message names the sanctioned seams).\n5. Evidence: rail-row open (media row click-to-settle) timing probe before/after, mounted identity tests for converted sites.\n6. Re-run Library suites + full --collect-only sweep; record per-site table, counts, timings in Implementation Notes.
<!-- SECTION:PLAN:END -->
