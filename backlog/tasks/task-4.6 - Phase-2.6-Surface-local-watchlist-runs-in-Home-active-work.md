---
id: TASK-4.6
title: 'Phase 2.6: Surface local watchlist runs in Home active work'
status: Done
assignee: []
created_date: '2026-05-03 19:03'
updated_date: '2026-05-03 19:08'
labels:
  - unified-shell
  - phase-2
  - home
  - watchlists
  - active-work
dependencies: []
parent_task_id: TASK-4
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose queued, running, and failed local watchlist runs on Home so the dashboard reflects real local W+C work instead of only static counts or placeholder controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local watchlist runs expose a synchronous Home-safe snapshot path that does not require awaiting inside Textual compose.
- [x] #2 Home maps queued, running, and failed local watchlist runs into visible active-work rows with stable item identity and W+C detail routing.
- [x] #3 Failed local watchlist runs become the next-best recovery action before generic active-work resume.
- [x] #4 Home primary action for failed local watchlist work opens the subscriptions watchlist-runs context.
- [x] #5 App wiring passes both local notification and local watchlist services into the Home adapter.
- [x] #6 Focused automated tests and Phase 2 QA evidence verify the service snapshot, Home state, Home screen behavior, app wiring, and tracking updates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regressions for a synchronous local watchlist run snapshot, Home adapter mapping, failed-run prioritization, Home primary-action routing, app wiring, and Phase 2 evidence tracking.
2. Implement the smallest service snapshot and Home adapter changes needed to expose queued, running, and failed local watchlist runs without adding false controls.
3. Route failed local watchlist work to the W+C runs context and update roadmap, QA evidence, and backlog task hygiene.
4. Run focused Home, W+C service, UI, and tracking verification plus diff hygiene before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a synchronous `LocalWatchlistsService.list_home_run_snapshot` path and extended the Home adapter to map local W+C watchlist runs into `HomeActiveWorkItem` rows while filtering completed/cancelled runs out of active work. Failed local watchlist runs now produce a `review_failed_work` next-best action that opens the subscriptions `watchlist-runs` context, and app startup wires Home to both notification and watchlist services. Added focused service, Home state, mounted Home screen, app wiring, and Phase 2 tracking tests plus QA evidence for `TASK-4.6`.
<!-- SECTION:NOTES:END -->
