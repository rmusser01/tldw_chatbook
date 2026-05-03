---
id: TASK-3.7
title: 'Phase 3.7: Launch active Schedules run from Schedules into Console'
status: Done
assignee: []
created_date: '2026-05-03 23:36'
updated_date: '2026-05-03 23:36'
labels:
  - unified-shell
  - phase-3
  - console
  - schedules
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md
  - Docs/superpowers/trackers/unified-shell-maturity-roadmap.md
parent_task_id: TASK-3
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Schedules destination expose a real Console follow action when the existing active-work adapter can identify an actionable schedule run, while preserving an honest disabled state when no run context exists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Schedules destination keeps Console follow disabled with recovery copy when no actionable schedule run exists.
- [x] #2 Schedules destination enables Console follow when a visible Schedules active-work item has Console launch context.
- [x] #3 Clicking the enabled Schedules Console follow action routes through the existing Home active-work adapter Console launch path.
- [x] #4 Focused automated tests and Phase 3 QA evidence verify the `schedules-follow-in-console` producer and roadmap/task updates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regressions for Schedules destination disabled fallback, enabled active-run Console follow, adapter-routed click behavior, source readiness, and Phase 3.7 tracking evidence.
2. Reuse the existing Home active-work adapter from Schedules to discover an active Console-capable schedule run without inventing a separate schedule service.
3. Render the Schedules Console follow button as enabled only when that adapter context exists, and route clicks through the existing app-level Home Console launch method.
4. Add Phase 3.7 QA evidence plus roadmap and task updates.
5. Run focused destination, Console handoff, navigation, and diff hygiene checks before commit/PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the Schedules destination to reuse the existing Home active-work adapter for active schedule-run Console discovery. The `schedules-follow-in-console` action now stays disabled with explicit recovery copy when no active schedule run exists, becomes enabled when adapter context exists, and routes clicks through `open_active_home_item_in_console` so Home and Schedules share the same Console launch path. Updated Console source readiness to show Schedules as connected and added focused destination/Console regressions plus Phase 3.7 QA evidence and roadmap tracking.
<!-- SECTION:NOTES:END -->
