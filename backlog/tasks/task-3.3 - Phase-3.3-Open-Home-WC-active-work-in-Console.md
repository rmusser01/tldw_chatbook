---
id: TASK-3.3
title: 'Phase 3.3: Open Home W+C active work in Console'
status: Done
assignee: []
created_date: '2026-05-03 21:41'
updated_date: '2026-05-03 21:48'
labels:
  - unified-shell
  - phase-3
  - console
  - home
  - watchlists
dependencies: []
parent_task_id: TASK-3
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Home W+C active-work items produce a real Console live-work launch context so users can inspect local watchlist run status from the primary agentic Console surface instead of seeing an unavailable Console action.

This slice uses the existing `ConsoleLiveWorkLaunch` status-card contract instead of introducing a parallel Console display path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Home W+C active-work rows advertise Console availability only when the adapter can provide launch context.
- [x] #2 Open in Console for a visible local watchlist run stages a Console launch with source, title, status, recovery, action, and run metadata.
- [x] #3 Unknown or unavailable watchlist run targets remain recoverable unavailable states.
- [x] #4 Focused automated tests and Phase 3 QA evidence verify the source-specific Console launch wiring and roadmap/task updates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regressions for Home local W+C active-work Console availability, adapter `OPEN_IN_CONSOLE` launch payloads, app Console staging metadata, and Phase 3 tracking evidence.
2. Implement the smallest Home adapter changes needed to reuse visible local watchlist run identity and return a Console launch payload for known runs while preserving unavailable fallback for unknown targets.
3. Pass status, recovery, and action label through app-level Console launch staging so the existing status card shows useful W+C run context.
4. Add Phase 3 QA evidence and roadmap/task updates.
5. Run focused Home, Console, roadmap, and diff hygiene checks before commit/PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Home W+C active-work Console launches by marking visible local watchlist run rows as Console-capable, returning a `HomeConsoleLaunch` with W+C source, status, recovery, action label, and run metadata, and preserving that metadata through app-level Console staging. Unknown W+C targets still return recoverable unavailable states. Added focused regression coverage plus Phase 3.3 QA evidence and roadmap tracking.
<!-- SECTION:NOTES:END -->
