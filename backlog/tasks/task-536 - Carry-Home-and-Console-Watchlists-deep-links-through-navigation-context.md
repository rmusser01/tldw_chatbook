---
id: TASK-536
title: Carry Home and Console Watchlists deep links through navigation context
status: Done
assignee: []
created_date: '2026-07-24 20:35'
updated_date: '2026-07-24 21:14'
labels:
  - ui
  - navigation
  - watchlists
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore route-specific Watchlists follow-through after the legacy SubscriptionWindow retirement so Home notification/run actions and Console live-work actions land on the intended Watchlists section and run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Home notification and failed-run primary actions emit current Watchlists Notifications and Runs section context instead of legacy pending subscription attributes.
- [x] #2 Home detail and Console live-work run actions carry both the Runs section and target run id through `NavigateToScreen.screen_context`.
- [x] #3 `WatchlistsCollectionsScreen` safely applies supported navigation context before mount and selects a requested run after runs load.
- [x] #4 Legacy `subscriptions` route aliases remain supported while no corrected flow depends on write-only `pending_watchlists` fields.
- [x] #5 The retained local notifications inbox is reachable in Watchlists and supports reviewing, marking read, and dismissing rows.
- [x] #6 Missing or backend-mismatched run targets are consumed without stale later selection, while canonical and raw run identifiers select the same visible record.
- [x] #7 Focused Home, Console, Watchlists, notification, and navigation tests plus Ruff, formatting, and diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red tests for Home primary-action contexts, app/Console run deep links, and Watchlists pre-mount section/run consumption.
2. Define small shared Watchlists navigation-context keys and make Home derive section context directly from its selected action.
3. Retain the local notifications inbox as a Watchlists section using the existing policy-aware `NotificationsInboxController`.
4. Replace app-level write-only pending Watchlists fields with NavigateToScreen screen_context payloads, including canonical routes and backend ownership for run targets.
5. Apply validated section/backend/run context in WatchlistsCollectionsScreen, consume one-shot run targets after loading, and keep visible pane selection consistent for canonical and raw identifiers.
6. Run focused Home, Console, Watchlists, notification, navigation-alias, Ruff/format, and diff checks; request independent review.

ADR required: yes
ADR path: backlog/decisions/018-watchlists-tui-screen.md
Reason: ADR-018 owns the legacy-subscriptions-to-Watchlists boundary and is amended here to make explicit that retiring `SubscriptionWindow` retains its local notifications inbox rather than silently dropping that user workflow.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced write-only app pending fields with validated `NavigateToScreen.screen_context` payloads for Home, Console, canonical Watchlists routes, and the legacy `subscriptions` alias. Run contexts carry section, backend ownership, and target id.
- Added the retained, client-owned Notifications section to Watchlists using `NotificationsInboxController`, with local-only ownership messaging, safe rendering, inspect/mark-read/dismiss actions, and screen-owned selection that survives Textual recomposition.
- Made run deep links one-shot and backend-aware across raw and canonical identifiers. Pre-mount and mounted navigation share one loader, stale/missing targets are consumed, and run/notification selections are committed outside transient pane lookups.
- Amended [ADR-018](../decisions/018-watchlists-tui-screen.md) and the Watchlists design spec to document the local notification-inbox ownership boundary and the Rules/Notifications module split.
- Verification: the combined Home/Console/Watchlists/notification suite passed 129 tests before the final lifecycle hardening; the final Watchlists/inspector/controller sweep passed 23 tests. Ruff passed for all changed Python files (with the repository's existing `app.py` ignores), scoped formatter checks passed, `compileall` passed, and `git diff --check` passed. Independent review approved the final state with 16 Watchlists destination tests and 21 combined checks passing.
- A broader visual sweep reached 384 passed and 1 skipped before failing on compact Schedules shell readiness. The same failure was reproduced from a clean HEAD archive, so it is recorded as a pre-existing follow-up rather than a TASK-536 regression.
<!-- SECTION:NOTES:END -->
