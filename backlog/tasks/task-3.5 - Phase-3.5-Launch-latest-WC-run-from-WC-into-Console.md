---
id: TASK-3.5
title: 'Phase 3.5: Launch latest W+C run from W+C into Console'
status: Done
assignee: []
created_date: '2026-05-03 22:52'
updated_date: '2026-05-03 22:56'
labels:
  - unified-shell
  - phase-3
  - console
  - watchlists
dependencies: []
parent_task_id: TASK-3
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the W+C destination expose a real Console follow action when the existing active-work adapter can identify an actionable local watchlist run, while preserving an honest disabled state when no run context exists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 W+C destination keeps Console follow disabled with recovery copy when no actionable W+C run exists.
- [x] #2 W+C destination enables Console follow when a visible W+C active-work item has Console launch context.
- [x] #3 Clicking the enabled W+C Console follow action routes through the existing Home active-work adapter Console launch path.
- [x] #4 Focused automated tests and Phase 3 QA evidence verify the `watchlists-follow-in-console` producer and roadmap/task updates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regressions for W+C destination disabled fallback, enabled latest-run Console follow, adapter-routed click behavior, and Phase 3.5 tracking evidence.
2. Reuse the existing Home active-work adapter from W+C to discover a latest Console-capable W+C run without duplicating watchlist snapshot logic.
3. Render the W+C Console follow button as enabled only when that adapter context exists, and route clicks through the existing app-level Home Console launch method.
4. Add Phase 3.5 QA evidence plus roadmap and task updates.
5. Run focused destination, Console handoff, Home adapter, navigation, and diff hygiene checks before commit/PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the W+C destination to reuse the existing Home active-work adapter for latest Console-capable W+C run discovery. The `watchlists-follow-in-console` action now stays disabled with explicit recovery copy when no active run exists, becomes enabled when adapter context exists, and routes clicks through `open_active_home_item_in_console` so Home and W+C share the same Console launch path. PR review hardening added contextual adapter-failure logging, pinned click routing to the item shown in the rendered button label, and escaped markup-sensitive run title/status labels. Added focused destination/Console regressions plus Phase 3.5 QA evidence and roadmap tracking.
<!-- SECTION:NOTES:END -->
