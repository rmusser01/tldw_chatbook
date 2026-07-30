---
id: TASK-1410
title: Watchlist auto-pause never fires; its only implementation is unreachable
status: To Do
assignee: []
created_date: '2026-07-30 08:20'
labels:
  - watchlists
  - correctness
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while implementing TASK-1383, which required establishing whether routing scheduled checks
through `LocalWatchlistsService` preserved the auto-pause behaviour of the path it replaced. It
does — exactly — because **neither path has ever auto-paused anything.**

`subscriptions.auto_pause_threshold` (default 10, `DB/Subscriptions_DB.py:195`) is compared against
`consecutive_failures` in exactly one place: the `if error:` branch of
`SubscriptionsDB.record_check_result` (`DB/Subscriptions_DB.py:1318-1341`), which sets
`is_paused = 1` and logs "Auto-paused subscription N after M failures".

That branch has no caller. After TASK-1383, `record_check_result` has a single production caller,
`Subscriptions/local_watchlists_service.py:448`, and it passes `items=None, stats=stats` with no
`error` argument at all — so the success branch is the only one ever taken. (Before TASK-1383 the
scheduled handler was the second caller, and it likewise only ever called it on success.)

Failures instead go to `SubscriptionsDB.record_check_error` (`DB/Subscriptions_DB.py:1391-1411`)
via `LocalWatchlistsService.record_run_failure` (`local_watchlists_service.py:492`). That method
bumps `consecutive_failures`, but it does **not** consult `auto_pause_threshold`; it writes
`is_paused = 1 if should_pause else 0`, and `should_pause` defaults to `False` and is never passed
by any caller. So every recorded failure writes `is_paused = 0` — a permanently failing source is
not merely left running, it is actively **un-paused** on each failure, which would also silently
clear a pause the user set by hand.

Net effect: `consecutive_failures` climbs forever, `auto_pause_threshold` is a setting the UI can
show and store but nothing reads, and a dead feed is retried on its cadence indefinitely.

Same failure class as TASK-1210/1212/1383: the machinery is present and looks live to a grep — a
threshold column, a comparison, a warning log — but no execution path reaches it.

Deciding which way to close it is part of the work: either make the failure path honour
`auto_pause_threshold` (the behaviour the schema and Settings already advertise), or remove the
threshold and the dead branch so the app stops promising a feature it does not have. The
`is_paused = 0` write on failure is a bug under either choice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A source that fails `auto_pause_threshold` times in a row reaches the documented outcome, and a test drives it through the real failure path rather than calling the DB method directly
- [ ] #2 Recording a check failure never clears an existing `is_paused`, whether the pause was automatic or set by the user
- [ ] #3 `auto_pause_threshold` is either read by a live code path or removed from the schema, the Settings UI and the docs together, so no setting is shown that nothing consumes
<!-- AC:END -->
