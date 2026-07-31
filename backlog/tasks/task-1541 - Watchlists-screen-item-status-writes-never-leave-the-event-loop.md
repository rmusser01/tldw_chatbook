---
id: TASK-1541
title: 'Watchlists screen: item-status writes never leave the event loop'
status: To Do
assignee: []
created_date: '2026-07-30 15:53'
labels:
  - watchlists
  - performance
  - spec-divergence
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`WatchlistsBackendController._maybe_await` (`tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py:29`)
only awaits its argument if it is already awaitable; it never wraps a synchronous call in
`asyncio.to_thread`. `WatchlistsCollectionsScreen._update_item_status` (`watchlists_collections_screen.py:3923`)
calls `self._controller.update_item_status(...)` through exactly this path, and is itself dispatched
via `self.run_worker(self._update_item_status(...), exclusive=True)` from the Ingest, Ignore, and
unread-toggle handlers, plus the silent mark-read-on-open path -- `run_worker` on a coroutine only
*schedules* it back onto the same event loop, it does not move it to a thread. The result: every one
of those transactional `SubscriptionsDB` writes runs on the UI thread from inside a worker, and
`Subscriptions_DB.py` configures no `busy_timeout` pragma, so a second app instance (or a background
check) contending for the same row blocks the UI for the duration of the lock wait, not just the
write.

Found during Task 5's review (2026-07-30), which hit the identical shape in the new
queue-for-briefing toggle and moved that one write to `asyncio.to_thread` (`c43f0f840`,
`_toggle_briefing_queue`) rather than fix the whole screen. That commit's own docstring names
`_maybe_await` as the reason a worker alone is not enough. This task is the pre-existing,
screen-wide version of the same bug across every other item-status write path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `_update_item_status`'s DB write runs off the event loop thread, verified by a thread-identity test following the shape of `test_the_queue_write_runs_off_the_event_loop_thread` (`Tests/UI/test_watchlists_inspector.py`)
- [ ] #2 The fix does not add `exclusive=True` cancellation that would abort one in-flight item-status write because another item's write started
- [ ] #3 Existing Ingest/Ignore/unread-toggle/mark-read-on-open item-action tests still pass unchanged
<!-- AC:END -->
