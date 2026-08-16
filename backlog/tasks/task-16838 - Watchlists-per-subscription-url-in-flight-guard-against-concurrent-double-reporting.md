---
id: TASK-16838
title: 'Watchlists: per-(subscription,url) in-flight guard against concurrent double-reporting'
status: To Do
assignee: []
created_date: '2026-08-16'
labels:
  - bug
  - concurrency
  - watchlists
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the TASK-15764 review (PR #1679, finding 1), re-verified at dev `ee741cf10`:
there is **no serialization mechanism for concurrent checks of the same source** —
`grep -rn "asyncio.Lock\|Semaphore\|in_flight" tldw_chatbook/Subscriptions/
tldw_chatbook/Scheduling/` still returns nothing relevant. Serialization is structural
only (the scheduler loop awaits one due task at a time, `Scheduling/scheduler/loop.py:141`;
url_list/sitemap loops are sequential, `Subscriptions/local_watchlists_service.py:1616-1643`).

But the scheduler runs as an async worker on the app's own event loop (`app.py`,
`run_worker(self.scheduler_loop.run(), ...)`), and a UI "Check Now" runs
`launch_run` → `execute_run` on the same loop
(`watchlists_collections_screen.py:4896-4903` → `watchlist_scope_service.py:606-624`) —
so a scheduled check of source X and a manual check of source X **can interleave**.
`check_url`'s read-baseline → await (network fetch at `monitoring_engine.py:1248`,
plus the off-loop hops) → write-snapshot shape means both runs can read the same baseline
before either writes: the review forced the interleave and got
`dispositions=['changed','changed']`, i.e. one page change **double-reported with two
snapshots written**. This pre-dates 15764 (identical on base) — the off-loop work only
widened an already network-wide window by ~35 ms.

Fix direction: a per-(subscription_id, url) in-flight guard (skip-or-coalesce the second
entrant), at the `check_url` orchestration seam rather than inside the engine.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A scheduled run overlapping a manual Check Now of the same source cannot double-report one page change or write two baseline snapshots for it (test forcing the interleave as evidence)
- [ ] #2 Distinct sources still check concurrently exactly as before (no global serialization)
- [ ] #3 The guard cannot strand a source as permanently "in flight" after a failed or cancelled check
<!-- AC:END -->
