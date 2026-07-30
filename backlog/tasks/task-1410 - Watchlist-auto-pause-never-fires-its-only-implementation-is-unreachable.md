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

`subscriptions.auto_pause_threshold` (schema default 10, `DB/Subscriptions_DB.py:195`) is compared
against `consecutive_failures` in exactly one place: the `if error:` branch of
`SubscriptionsDB.record_check_result` (`DB/Subscriptions_DB.py:1318-1341`), which sets
`is_paused = 1` and logs "Auto-paused subscription N after M failures".

**That branch has no caller.** After TASK-1383, `record_check_result` has a single production
caller — `Subscriptions/local_watchlists_service.py:448` — and it passes `items=None, stats=stats`
with no `error` argument, so only the success branch is ever taken. (Before TASK-1383 the scheduled
handler was the second caller and likewise only called it on success.) `DB/Subscriptions_DB.py:1333`
is consequently the only `is_paused = 1` write in the codebase, and it is dead.

Failures instead go to `SubscriptionsDB.record_check_error` (`DB/Subscriptions_DB.py:1391-1411`)
via `LocalWatchlistsService.record_run_failure` (`local_watchlists_service.py:509`). That method
bumps `consecutive_failures` but never consults `auto_pause_threshold`; it writes
`is_paused = 1 if should_pause else 0`, and `should_pause` defaults to `False` and is passed by no
caller. So every recorded failure writes `is_paused = 0`.

Net effect: `consecutive_failures` climbs forever, nothing reads `auto_pause_threshold`, and a dead
source is retried on its cadence indefinitely.

### Scope of the un-pause write, precisely

The `is_paused = 0` write is **not reachable from the scheduler**, and is **currently vacuous** —
both facts matter for how this is fixed:

- The scheduled path skips paused sources before any check runs, at
  `Scheduling/scheduler/handlers/watchlist_check_handler.py:132` and in the projection's status
  mapping, `Scheduling/services/watchlist_projection.py:60`. A paused source is never checked on a
  schedule and so never reaches `record_check_error`.
- `launch_run`/`execute_run` have **no** paused guard, so the write is reachable via a **manual
  re-check** of a paused source.
- But nothing in production ever writes `is_paused = 1` in the first place: the auto-pause branch
  above is dead, and there is no pause UI or CLI. (`is_paused` sits in `update_subscription`'s
  field allowlist at `DB/Subscriptions_DB.py:1072`, so the write surface exists, but no caller
  passes it.) There is therefore no pause for this to clear today — the bug is latent, and becomes
  live the moment anything starts setting `is_paused = 1`.

That ordering is the point: **AC#2 is a hard prerequisite of AC#1.** Landing auto-pause on its own
produces a pause that the next manual re-check of that source silently erases, which is worse than
today's honest do-nothing.

### The config surface is separately named and equally unread

The user-facing setting is **`auto_pause_after_failures`** (`config.py:3553`, documented at
`Docs/Features/SUBSCRIPTION_IMPLEMENTATION_PLAN.md:1052`), not `auto_pause_threshold`. It is read
by nothing. `auto_pause_threshold` is a per-subscription column with **no Settings UI at all** — it
appears only in the schema (`:195`) and in field allowlists (`DB/Subscriptions_DB.py:969,1085`,
`Subscriptions/local_watchlists_service.py:840`). Whichever direction this task takes, it must name
both and decide — or explicitly defer — whether the global config key and the per-source column
unify, and which wins when they disagree.

Same failure class as TASK-1210/1212/1383: the machinery is present and looks live to a grep — a
threshold column, a config key, a comparison, a warning log — but no execution path reaches it.

Deciding which way to close it is part of the work: either make the failure path honour the
threshold (the behaviour the schema and config already advertise), or remove the dead branch and
both settings so the app stops promising a feature it does not have.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Recording a check failure never clears an existing `is_paused`; landed BEFORE #2, because auto-pause without it produces a pause the next manual re-check erases
- [ ] #2 A source that fails the configured number of times in a row reaches the documented outcome, driven in a test through the real failure path rather than by calling the DB method directly
- [ ] #3 `auto_pause_threshold` (column) and `auto_pause_after_failures` (`config.py:3553`) are reconciled: either both are read by a live path with a stated precedence, or both are removed together with the dead branch and the docs that advertise them
<!-- AC:END -->
