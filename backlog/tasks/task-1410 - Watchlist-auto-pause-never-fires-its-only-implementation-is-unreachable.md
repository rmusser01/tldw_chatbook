---
id: TASK-1410
title: Watchlist auto-pause never fires; its only implementation is unreachable
status: Done
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
- [x] #1 Recording a check failure never clears an existing `is_paused`; landed BEFORE #2, because auto-pause without it produces a pause the next manual re-check erases
- [x] #2 A source that fails the configured number of times in a row reaches the documented outcome, driven in a test through the real failure path rather than by calling the DB method directly
- [x] #3 `auto_pause_threshold` (column) and `auto_pause_after_failures` (`config.py:3553`) are reconciled: either both are read by a live path with a stated precedence, or both are removed together with the dead branch and the docs that advertise them
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Direction (AC#3 fork): COMPLETE auto-pause rather than remove it — the schema column, config knob,
and dead branch all intend it, and task-1394 already made `record_check_result`'s pause branch
reachable for all-error runs. Wire it fully and consistently.
1. **AC#1 (prerequisite):** `record_check_error` must NEVER clear `is_paused`. Change its UPDATE so
   a failure only ever SETS is_paused (write `is_paused = CASE WHEN <should_pause> THEN 1 ELSE
   is_paused END`), never 0. A recorded failure can pause but never un-pause.
2. **AC#2:** `record_check_error` consults `auto_pause_threshold` and pauses when
   `consecutive_failures` (post-increment) >= threshold — the SAME logic `record_check_result`'s
   error branch uses (1394 made it live). Factor a shared helper so the two failure paths cannot
   diverge. Test through the real path (`record_run_failure` → `record_check_error`): a source that
   fails N times in a row ends `is_paused=1`.
3. **AC#3:** `auto_pause_after_failures` (config, currently read by nothing) seeds the
   `auto_pause_threshold` column default for NEW subscriptions; per-subscription column overrides
   (stated precedence). Document both in the config comment + a module docstring.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Shared helper.** Both failure paths now go through one new private method,
`SubscriptionsDB._advance_failure_and_maybe_pause(cursor, subscription_id, error, now, *,
force_pause=False)` (`DB/Subscriptions_DB.py`): it does the `last_checked`/`last_error`/
`error_count`/`consecutive_failures` UPDATE, reads back the post-increment count, and pauses (with
the "Auto-paused subscription N after M failures" WARNING) iff `force_pause` or
`consecutive_failures >= auto_pause_threshold`. `record_check_result`'s `if error:` branch and
`record_check_error` both call it — neither has its own copy of the threshold comparison, so they
cannot diverge. The pause `UPDATE` inside the helper only ever sets `is_paused = 1`; there is no
`is_paused = 0` write left anywhere in either failure path (AC#1), because `record_check_error` no
longer runs its own combined UPDATE that used to include `is_paused = ?`.

**`should_pause` fate.** Folded into the shared decision rather than removed: it is now the
`force_pause` argument to the helper, so passing `should_pause=True` still forces a pause on that
one failure regardless of the threshold, but — like the threshold path — it can only ever set
`is_paused = 1`, never clear it. No production caller passes `should_pause=True` today (grep
confirmed); it is kept as an escape hatch for a caller that already knows a failure is terminal,
documented as such in `record_check_error`'s docstring.

**AC#3 precedence.** `add_subscription` seeds `fields["auto_pause_threshold"]` from a new
module-level `_default_auto_pause_threshold()` (reads `[subscriptions].auto_pause_after_failures`
via the three-argument `get_cli_setting(section, key, default)` form, never the dotted form — the
TASK-1771 default-drop trap) **only when the caller did not already supply the field**, i.e. an
explicit `auto_pause_threshold` kwarg (including one forwarded from
`LocalWatchlistsService.create_source`/`update_source`) always wins. A missing or non-numeric
config value falls back to the same `10` the schema's own `DEFAULT 10` already uses, so a broken
config cannot block subscription creation. Existing rows are untouched — this only changes what a
brand-new INSERT defaults to. Documented in `_default_auto_pause_threshold`'s docstring and an
inline comment on `config.py`'s `[subscriptions]` template next to `auto_pause_after_failures`.

**Tests.** `Tests/DB/test_subscriptions_db.py` gained 4 tests (AC#1 direct-DB never-unpause test,
plus 3 for AC#3's seed/override/fallback precedence via `monkeypatch.setattr` on the module-level
`get_cli_setting` name). `Tests/Subscriptions/test_local_watchlists_service.py` gained 2 tests: AC#2
drives the real path (`execute_run` raising → `record_run_failure` → `record_check_error`) for 3
failures at `auto_pause_threshold=3` and asserts exactly one auto-pause WARNING (via a
`_loguru_to_caplog` bridge fixture, same pattern as
`Tests/Model_Artifacts/test_credentials_and_boundaries.py`); a consistency test drives an all-error
`url_list` run (the task-1394 `record_check_result` path) and a plain always-failing source (the
`record_check_error` path) at the same threshold and asserts both end `is_paused=1` at the same
failure count. Every new/changed behavior was mutation-tested (Edit → run → Edit-revert, `git
status --short` clean between): AC#1's mutation (restoring the old `is_paused = 1 if should_pause
else 0` write) reds all 3 pause-related tests including the consistency test; AC#2's mutation
(dropping the threshold comparison to `force_pause` only) reds the AC#2 test, the consistency test,
and the pre-existing task-1394 all-error test; AC#3's mutations (disabling the seed, forcing the
seed to always override, and removing the `int()` fallback) each red exactly the test they target.
Full suite: 901 passed (895 pre-existing + 6 new) across `Tests/Subscriptions`, `Tests/Scheduling`,
`Tests/DB/test_subscriptions_db.py`.

**Files touched:** `tldw_chatbook/DB/Subscriptions_DB.py` (shared helper, `record_check_result`/
`record_check_error` rewired, `_default_auto_pause_threshold` + `add_subscription` seeding),
`tldw_chatbook/config.py` (precedence comment on `auto_pause_after_failures`),
`Tests/DB/test_subscriptions_db.py`, `Tests/Subscriptions/test_local_watchlists_service.py`.

**Left open / not done here:** `Docs/Features/SUBSCRIPTION_IMPLEMENTATION_PLAN.md:1052` (mentioned
in the task description as documenting `auto_pause_after_failures`) was not re-checked against the
new seeding behavior — out of scope for the stated ACs, called out here rather than silently
skipped. Status intentionally left **In Progress** per dispatch instructions rather than moved to
Done.

**Fix wave (whole-branch review follow-up).** The review found the branch clean against the three
ACs but flagged that auto-pause, as landed, was a permanent trap: `reset_subscription_errors` (the
only `is_paused = 0` writer) has zero callers, the scheduler skips paused sources, and
`record_check_result`'s success branch did not clear `is_paused` — so nothing could ever un-pause an
auto-paused source. Fixed by giving a **successful check** the natural recourse: the success branch
in `record_check_result` now also sets `is_paused = 0` alongside its existing counter reset. AC#1 is
unchanged (a failure never un-pauses); a success now does. This is coherent with the fact that
`launch_run`/`execute_run` have no paused guard — a manual re-check of a paused source still runs,
and a successful one now resumes it. No paused guard was added to `launch_run`/`execute_run`; doing
so would have removed this recourse entirely. Also guarded
`_advance_failure_and_maybe_pause`'s threshold comparison (`consecutive_failures >=
auto_pause_threshold`) against a NULL or non-positive `auto_pause_threshold` — previously a `TypeError`
on NULL (`int >= None`) or an instant pause-on-first-failure on 0/negative; no production path seeds
either today (the config seed falls back to 10, the service strips `None`), but a direct
`update_subscription(auto_pause_threshold=...)` reached the unguarded comparison. Both are now
treated as "auto-pause disabled for this source." Also corrected the helper's docstring, which
claimed the success branch already un-paused before this fix wave made that true, and rewired the
`record_check_result` metric's `"auto_paused"` label — previously always `"false"` because it
text-sniffed `error` for a substring the helper never wrote there — to the helper's own return value
instead. New tests: `Tests/DB/test_subscriptions_db.py` gained
`test_record_check_result_success_resumes_an_auto_paused_subscription` and a
`bad_threshold`-parametrized (`None`/`0`/`-1`) guard test;
`Tests/Subscriptions/test_local_watchlists_service.py` gained
`test_local_watchlists_service_successful_manual_recheck_resumes_a_paused_source`, driving the resume
through the real `launch_run`/`execute_run` path. All mutation-tested (Edit → run → revert, `git
status --short` clean between): dropping the new `is_paused = 0` write reds both resume tests;
dropping the threshold guard reds all three parametrized cases (`None` raises `TypeError`; `0`/`-1`
pause on the first failure). Filed TASK-2050 (low priority) for the still-missing UI resume
affordance and paused indicator — this fix wave closes the data-layer trap, not the UX gap the
review's Finding #1 also named.
<!-- SECTION:NOTES:END -->
