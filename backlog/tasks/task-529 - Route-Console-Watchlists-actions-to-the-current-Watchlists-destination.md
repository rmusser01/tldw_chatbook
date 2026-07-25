---
id: TASK-529
title: Route Console Watchlists actions to the current Watchlists destination
status: Done
assignee: []
created_date: '2026-07-24 19:21'
updated_date: '2026-07-24 19:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore direct and button-driven Console follow-through for Watchlists runs after SubscriptionWindow retirement by resolving live-work actions to the active Watchlists route and staged run-context fields.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Watchlists live-work primary actions resolve to the active watchlists_collections route
- [x] #2 The app stages pending_watchlists_section and pending_watchlists_run_id for a selected run
- [x] #3 Direct and mounted Console action tests navigate to Watchlists without warning
- [x] #4 The full Console live-work handoff module passes after schedule test alignment
- [x] #5 The regression source, existing ADR, and verification are documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the direct and mounted action failures and trace the route mismatch through the resolver and app handler.
2. Update the resolver to the active Watchlists destination and update tests to assert the current one-shot staging fields.
3. Run focused resolver/action tests, the full live-work module, related Console live-work tests, Ruff, format, and diff checks.
4. Independently review the cross-module route repair and document the existing ADR before completion.

ADR required: no (existing ADR applies)
ADR path: backlog/decisions/018-watchlists-tui-screen.md
Reason: ADR-018 already requires the subscriptions route to alias/fold into watchlists_collections; this repairs a missed resolver consumer of that decision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed the Console live-work primary-action resolver to return `watchlists_collections` for Watchlists runs, matching the app handler and ADR-018. Direct and mounted tests now verify the current `pending_watchlists_section`/`pending_watchlists_run_id` staging contract, current destination, and absence of warning notifications. The mounted test uses Textual's deterministic `Button.press()` path because coordinate clicking the off-screen card landed on the composer send button.

The full UI sweep exposed the route regression as two failures: the resolver returned retired `subscriptions`, which the app primary-action handler intentionally no longer accepts. Focused schedule/Watchlists coverage passes 8/8; the complete live-work handoff module passes 47/47 on this branch; Ruff and formatting checks pass.

ADR required: no new ADR. Existing `backlog/decisions/018-watchlists-tui-screen.md` owns the SubscriptionWindow retirement and Watchlists destination alias.
<!-- SECTION:NOTES:END -->
