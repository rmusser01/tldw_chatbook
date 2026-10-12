---
id: TASK-2763
title: 'Home never refreshes after mount — statuses freeze until a click or remount'
status: Done
assignee: ['@t2-home']
created_date: '2026-08-06'
updated_date: '2026-10-11'
labels: [home, bug, ui]
dependencies: []
---

## Description (the why)

`_sync_home_triage` has exactly three callers: the two on-mount workers
(`UI/Screens/home_screen.py:325, 364`) and the rail-row click handler
(`:719`). The screen has no `set_interval`, no timer, no reactive watcher,
and no event subscription — `@on(Button.Pressed)` is its only handler.

Live-verified (dev @ 84e4b33f0, 2026-08-06): with no clicks, the rail is
byte-identical after 6 s idle — twice the active-work adapter's own 3.0 s
cache TTL. A Library ingest job progressing `queued → parsing → writing →
done` while the user watches Home does not move on screen; the Running
count in the section header is equally frozen. The only refresh paths are
clicking a row or leaving and re-entering the screen.

## Acceptance Criteria (the what)

- [x] Home reflects active-work state changes without user interaction
      (poll, subscription, or message-driven re-sync — implementer's choice).
- [x] The refresh path respects the adapter's cache TTL (no busy-loop).
- [x] A test drives an ingest-job state change and asserts the rail updates
      without a click.

## Implementation Plan

1. Premise-verify at origin/dev 9f8533686c: no set_interval/timer/watcher
   in home_screen.py; `_sync_home_triage` reachable only from the two
   mount workers and the row-click handler.
2. Red-first: drive the REAL adapter (no test-input override) with a live
   ingest-jobs provider; assert a queued -> parsing transition moves the
   canvas/rail with no clicks (RED pre-fix: never updates); pin the poll
   interval == adapter TTL; pin no-update while another route covers Home
   and update-after-pop.
3. Implement poll: `set_interval(_HOME_AUTO_REFRESH_INTERVAL_SECONDS)` —
   a module constant defaulting to the adapter's now-public
   `ACTIVE_WORK_CACHE_TTL_SECONDS`. The tick gates on Home being the
   current screen, compares the adapter's new
   `active_work_change_signature()` (live-read ingest feed) against the
   last-seen value, and hands off to the existing cache worker, which
   re-syncs when the adapter recomputed OR the live feed moved. Pause on
   ScreenSuspend, resume on ScreenResume (Home is a reusable route).

## Implementation Notes

- Mechanism (implementer's choice: poll): one timer per HomeScreen,
  created on first `on_screen_resume`, paused in a new
  `on_screen_suspend` (no super call — same MRO contract as the existing
  resume hook; base defines no suspend handler). Cadence ==
  `ACTIVE_WORK_CACHE_TTL_SECONDS` (3.0 s), now public on the adapter: a
  tick inside a fresh cache window can only no-op, so polling faster is a
  busy-loop and slower needlessly extends staleness. The tick's
  screen-current gate plus the suspend pause keep an invisible Home from
  churning seams.
- Ingest jobs are read live at every `build_dashboard_input` (not part of
  the TTL cache), so a job can progress inside a fresh cache window. The
  adapter gained `active_work_change_signature()` ((job_id, state) pairs)
  and the tick compares it against the compose-time-seeded baseline; a
  change sets a pending flag that makes the existing
  `_refresh_home_active_work_cache` worker re-sync even when
  `refresh_active_work_cache_async` reports the cache fresh. Steady-state
  cost: one cheap signature read per tick, one seam recompute per TTL
  window, and the B3 equality guards make an unchanged re-sync a no-op.
- Test override semantics preserved: with `_home_dashboard_test_input`
  set, the worker still returns early (hermetic inputs stay verbatim);
  the pending flag is cleared on that path too.
- Tests (red-first): `test_home_auto_refresh_updates_running_ingest_
  without_clicks` (real adapter, queued -> parsing observed on the canvas
  status line with zero clicks), `test_home_auto_refresh_pauses_while_
  another_screen_is_on_top` (frozen while covered by another screen — 5
  ticks elapse with no update — then updates after pop),
  `test_home_auto_refresh_interval_follows_adapter_cache_ttl`.
- Files: `tldw_chatbook/UI/Screens/home_screen.py`,
  `tldw_chatbook/Home/active_work_adapter.py`,
  `Tests/UI/test_home_screen.py`.
- Verified: all six Home suites (254 passed, including the new tests) at
  the fix commit.

ADR required: no
ADR path: N/A
Reason: screen-local lifecycle/polling change reusing the adapter's
existing cache contract; no storage, sync policy, provider boundary, or
cross-module interface change beyond one additive, defensive-read method.
