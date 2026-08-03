---
id: TASK-2050
title: Watchlists resume affordance for auto-paused sources
status: To Do
assignee: []
created_date: '2026-08-02'
labels:
  - watchlists
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix wave for the task-1410 review (Finding #1) gave an auto-paused source its first real recourse:
`SubscriptionsDB.record_check_result`'s success branch now writes `is_paused = 0` alongside its
existing counter reset, so a check that succeeds resumes a paused source. Combined with
`launch_run`/`execute_run` never having a paused guard, a **manual re-check** of a paused source
runs, and if it succeeds the source resumes.

That recourse exists entirely at the data layer. There is still no explicit "Resume" / un-pause
action anywhere in the watchlists UI, and nothing in the UI distinguishes an auto-paused source from
one that is merely inactive or healthy. `SubscriptionsDB.update_subscription(is_paused=0)` and
`reset_subscription_errors` both exist as un-pause writes, but grep confirms neither has a caller
outside the DB layer itself — there is no UI (or service-layer) path that invokes either one.

Net effect: a source that auto-pauses after repeated failures is visible only as a silently
stalled feed. A user who does not already know to trigger a manual re-check (and does not know
that succeeding is what un-pauses it) has no way to tell the source is paused, let alone resume it,
without editing config directly or reverse-engineering the recourse above.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A paused source (`is_paused = 1`) is visibly marked as paused somewhere in the watchlists UI, distinguishable from an inactive or healthy source
- [ ] #2 The user can resume a paused source with a single UI action, without editing config or manually crafting a re-check
- [ ] #3 That resume action clears `is_paused` and resets the failure counters (`error_count`, `consecutive_failures`, `last_error`) via an existing or new service-layer call, not a direct DB write from the UI layer
<!-- AC:END -->
