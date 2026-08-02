---
id: TASK-1541
title: 'Watchlists screen: item-status writes never leave the event loop'
status: In Progress
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
- [x] #1 `_update_item_status`'s DB write runs off the event loop thread, verified by a thread-identity test following the shape of `test_the_queue_write_runs_off_the_event_loop_thread` (`Tests/UI/test_watchlists_inspector.py`)
- [x] #2 The fix does not add `exclusive=True` cancellation that would abort one in-flight item-status write because another item's write started
- [ ] #3 Existing Ingest/Ignore/unread-toggle/mark-read-on-open item-action tests still pass unchanged -- see Implementation Notes: one pre-existing test is now measurably flakier, root-caused to an adjacent, out-of-scope hazard, left unchecked pending a decision on the follow-up
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Mirror `_toggle_briefing_queue`'s (`c43f0f840`) `asyncio.to_thread` shape in `_update_item_status`: the controller call wrapped so the transactional write leaves the event loop; keep `run_worker` dispatch but audit its `exclusive=True` per AC #2 (per-item writes must not cancel each other).
2. Thread-identity test following `test_the_queue_write_runs_off_the_event_loop_thread`; run the existing Ingest/Ignore/unread/mark-read tests unchanged (AC #3).
3. Mutation: revert the to_thread wrap → thread-identity test REDs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Added `WatchlistsCollectionsScreen._update_item_status_off_loop` (`watchlists_collections_screen.py:6663`):
`_update_item_status` now awaits this instead of `self._controller.update_item_status(...)` directly.
The whole chain underneath (`controller.update_item_status` -> `WatchlistScopeService.update_item` ->
`LocalWatchlistsService.update_item` -> `SubscriptionsDB.mark_item_status`) is `async def` all the way
down with no genuine `await` anywhere in it, so a plain `await` -- even from inside a `run_worker`
coroutine -- ran the transactional `UPDATE` synchronously on the event-loop thread. The new helper
captures `runtime_backend` on the calling (loop) thread, then drives the controller coroutine to
completion inside `asyncio.to_thread`, using a throwaway `asyncio.run()` in the worker thread (the
thread has no event loop of its own). This mirrors the ALREADY-established codebase idiom for this
exact shape -- `library_screen.py`'s `_run_library_service_call(..., isolate_in_worker=True)` -- rather
than inventing a new one, and keeps going through the controller/scope-service/local-service layers
(so status validation, id normalization and policy enforcement all still run) instead of reaching past
them to call `SubscriptionsDB.mark_item_status` directly the way `_toggle_briefing_queue` does for its
simpler, controller-free write.

**`exclusive=` decision (AC #2).** Audited all four `run_worker` dispatch sites for `_update_item_status`:
- **Ingest/Ignore** (`handle_ingest_requested`/`handle_ignore_requested`) previously had `exclusive=True`
  with no `group=`, landing in the plain `"default"` group shared by ~25 other call sites (`_check_now_
  source`, `_load_items`, etc.). Before this fix that was inert for cancellation purposes -- the write
  had no real suspension point, so by the time any other worker could start, this one had already run to
  completion (asyncio only delivers a cancellation at a genuine `await` boundary). Once the write gets a
  real `asyncio.to_thread` boundary, that same `exclusive=True` would newly be able to cancel an
  in-flight write for a DIFFERENT item (or any unrelated default-group action) mid-flight -- exactly what
  AC #2 forbids, and exactly the "zombie work" `_toggle_briefing_queue`'s own docstring warns about
  (`asyncio.to_thread`'s underlying thread keeps running after the awaiting Task is cancelled). Fixed by
  giving Ingest/Ignore a **per-item** group (`_ITEM_STATUS_ACTION_WORKER_GROUP_PREFIX + item_id`),
  keeping `exclusive=True`: a second Ingest/Ignore on the SAME item still supersedes its own earlier
  write (deterministic "last press wins", same trade-off `_toggle_briefing_queue` already accepts for the
  briefing-queue flag), but two DIFFERENT items' writes -- or an unrelated default-group action -- can
  never cancel each other. This matters concretely because `SubscriptionsDB` sets no `busy_timeout`
  pragma, so a contended write is exactly the scenario (from this task's own description) where a write
  for item A could sit in flight for seconds while the user moves on and acts on item B.
- **Unread toggle / mark-read-on-open** (`_mark_item_unread` / `_mark_item_read_on_open`) already share a
  DEDICATED group (`_ITEM_STATUS_WORKER_GROUP`) with `exclusive=True`, added by Task 5 specifically so a
  fast `j`/`k` run supersedes its own earlier (possibly different-item) write rather than piling one up
  per keystroke, while never touching unrelated default-group work. This is pre-existing, documented,
  intentional cross-item-supersede design, not something this task adds -- left completely unchanged.

**AC #3 -- one existing test is now measurably flakier; documented rather than silently patched around.**
Full targeted suite (`test_watchlists_inspector.py`, `test_watchlists_item_actions.py`,
`test_watchlists_read_status.py`, `test_watchlists_content_pane.py`, `test_watchlists_check_now_
failure.py`, `Tests/Watchlists/` -- 470 tests) passes green. Under repeated isolated runs, though,
`test_mark_unread_refuses_to_overwrite_an_item_ingested_by_the_real_gesture`
(`Tests/UI/test_watchlists_content_pane.py`) went from ~8% flaky on unmodified `dev` to ~35-40% flaky
with this fix applied (interleaved A/B sampling, n=24 each). Root-caused with temporary instrumentation
(not committed): the failure is a `NoMatches` on a widget the test queries immediately after its own
DB-polling loop observes the new status -- i.e. `Ingest`'s `refresh=True` path calls
`_refresh_overview_data()`, which sets `overview_data` (`reactive(..., recompose=True)`), forcing a full
screen recompose. Before this fix, the write and the recompose dispatch were atomically coupled (no real
suspension point), so the test's DB-polling loop could never observe the new status before the recompose
had *also* been dispatched. After this fix, the write's visibility (a commit inside the worker thread)
and the screen's own continuation resuming on the loop are genuinely decoupled by the `asyncio.to_thread`
handoff, so the test can race ahead and query a widget while the recompose is transiently mid-flight.
Tried a lighter-weight variant (driving the coroutine manually via `.send(None)` instead of
`asyncio.run()`, to test whether call overhead was the driver) -- it did not help (measured *worse*,
~60% in n=15), which rules out "make the thread hop cheaper" as a fix and confirms this is inherent to
having a genuine off-thread yield at all, not to this implementation's specific shape. `_toggle_briefing_
queue` and mark-read-on-open both avoid this because their post-write UI update is a targeted patch
(`_patch_item_queued_flag` / `patch_item=`), never a full recompose; Ingest/Ignore/unread-toggle still
use `_update_item_status`'s `refresh=True` default, which does recompose. Fixing that (giving Ingest/
Ignore/unread-toggle the same non-recomposing post-write update already proven for the other two paths)
would close this properly, but is a materially larger, user-visible behavior change outside this task's
ACs, so it is left unimplemented and unrecommended-into-code here -- AC #3's checkbox is left unchecked
and status stays In Progress pending a decision on that follow-up, rather than silently declared done.

**Files changed:** `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (new
`_ITEM_STATUS_ACTION_WORKER_GROUP_PREFIX` constant, `_update_item_status_off_loop`, Ingest/Ignore
dispatch groups); `Tests/UI/test_watchlists_inspector.py` (new
`test_the_item_status_write_runs_off_the_event_loop_thread`, mirroring the queue-write precedent).
