---
id: TASK-1540
title: Watchlists briefings (spec #2) programme tracking
status: To Do
assignee: []
created_date: '2026-07-30 15:53'
labels:
  - watchlists
  - briefings
  - tracking
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tracks delivery of `Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md` (spec #2:
Watchlists Briefings & Podcasts) across its four phases, so the board reflects the programme's
real state at a glance rather than requiring a re-read of the spec's status line each time.

Phase 1 (text briefings on demand, preset-less) shipped 2026-07-30: watchlist/briefing tables and
id-watermark selection, the `chat_api_call` generation pipeline, the reader's Artifacts section
(list + reader + Generate), and the queue-for-briefing affordance on watchlist items. Phases 2-4
(N-speaker presets/scripts/audio, exports + podcast feed directory, and scheduled generation via
the TASK-1383 scheduler seam) are designed but not started.

This task itself carries no acceptance work beyond keeping its checkboxes current as later phases
land; do not fold phase implementation into it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Phase 1: text briefings on demand -- tables, id-watermark selection, `chat_api_call` pipeline, Artifacts section (list + reader + Generate), queue-for-briefing affordance, preset-less
- [ ] #2 Phase 2: N-speaker presets, script casting, and audio synthesis
- [ ] #3 Phase 3: markdown/audio exports and the podcast feed directory
- [ ] #4 Phase 4: scheduled briefing generation via the TASK-1383 scheduler seam
<!-- AC:END -->
