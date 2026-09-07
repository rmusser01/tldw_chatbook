---
id: TASK-1812
title: Briefing schedule gate/UX residuals
status: Done
assignee: []
created_date: '2026-08-01 19:08'
updated_date: '2026-08-02 09:30'
labels:
  - watchlists
  - briefings
  - scheduling
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed during the whole-branch review fix wave for the Watchlists briefings phase 4 branch
(spec #2), bundling three minors the reviewer parked rather than blocking the wave on.

1. **The cadence picker ignores the kill switch.** `[scheduling] briefing_schedules_enabled`
   (`config.py:2370`, read only at `app.py:4738-4743`) gates whether `app.py` ever constructs the
   `BriefingProjection`/`BriefingJobHandler` pair that makes a schedule actually fire. Nothing gates
   the UI side: `cadence_scope_phrase` (`UI/Watchlists_Modules/artifacts_pane.py:331-361`) turns any
   non-NULL `briefing_cadence_seconds` into "scheduled &lt;cadence&gt; while the app is open"
   unconditionally, and the cadence `Select` itself is never disabled when the flag is off. There is
   no UI control for this flag today (hand-edit-only), so the gap is currently latent, but a
   watchlist can be fully configured to look scheduled while the process that would ever dispatch it
   was never wired up.
2. **A cadence pick has an undocumented activation delay.** `set_watchlist_briefing_settings`
   writes `briefing_cadence_seconds` synchronously the moment the picker changes, but the running
   `SchedulerLoop`'s `PriorityQueue` only re-reads `list_briefing_schedules` (via
   `BriefingProjection`) every `queue_reload_interval_ticks` ticks (`Scheduling/scheduler/loop.py:31`,
   default 60 -- roughly the ~30-minute reload cadence this same review's FIX 1 reasons about). A
   freshly-picked schedule can therefore sit inert for up to one reload cycle before the scheduler
   ever sees it. Neither the picker's own copy nor `Docs/User_Guide/watchlists.md`'s "Scheduled
   briefings" section states this.
3. **The zombie sweep's `exclude` is watchlist-granular, not row-granular.** `fail_interrupted_
   briefings`'s `exclude` (`Subscriptions/briefing_service.py:785-815`) skips every `generating` row
   for a watchlist id present in the collection, on the reasoning that such a row is "a LIVE,
   in-process generation" (the docstring's own words). That reasoning only holds if a watchlist can
   have at most one `generating` row at a time -- but a genuine crash-zombie row from a PRIOR process
   can coexist with a freshly-claimed live generation for the SAME watchlist (the crash predates the
   claim). When that happens, the live claim's presence in `exclude` incidentally shields the old
   zombie row too, so it survives until a sweep runs while that watchlist is NOT claimed -- the
   docstring over-claims what `exclude` actually protects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When `[scheduling] briefing_schedules_enabled` is `false`, the Artifacts cadence picker and the scope label no longer imply an active schedule (disabled control, and/or copy stating scheduling is off at the app level), for a watchlist that already has a stored cadence
- [x] #2 The cadence picker's UI copy or the user guide's "Scheduled briefings" section states that a newly picked cadence can take up to one queue-reload cycle to reach the running scheduler
- [x] #3 `fail_interrupted_briefings`'s `exclude` (or its docstring) is corrected so a crash-zombie row is swept even when its watchlist has an unrelated live claim -- either by scoping the exclusion to the actual claimed briefing row rather than the whole watchlist, or by an accurate docstring plus a regression test pinning the coexistence case (a zombie row and a live claim on the same watchlist in the same sweep)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Gate the cadence picker on `[scheduling] briefing_schedules_enabled`: when off, disable the Select and have `cadence_scope_phrase` say scheduling is off at the app level (stored cadence shown but inert).
2. Document the up-to-one-reload-cycle activation delay in the picker copy and/or `Docs/User_Guide/watchlists.md`.
3. Fix `fail_interrupted_briefings` exclusion: scope to the actual claimed briefing row if the claim seam can carry the row id cleanly, else correct the docstring and pin the zombie+live-claim coexistence case with a regression test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: all three sub-fixes shipped, ACs 1-3 all closed via the first (preferred) arm named in each.

AC #1 (kill-switch honesty): Added `ArtifactsPane.briefing_schedules_enabled` (reactive,
default True), screen-injected via a new `WatchlistsCollectionsScreen._briefing_schedules_
enabled()` reading `get_cli_setting("scheduling", "briefing_schedules_enabled", True)` --
the identical read `app.py`'s wiring block uses, following the "queries config helper"
convention (mirrors Tools_Settings_Window/STTS_Window) since no live app-instance handle
exists for this flag today. Injected at both pane-construction sites (initial build and
`_load_briefings` reload), same seam `chachanotes_available` uses. `cadence_scope_phrase`
gained a `schedules_enabled: bool = True` kwarg: when off, a stored cadence reads as
"stored to run <cadence>, but scheduled briefings are turned off for this app -- this
schedule will not fire" instead of implying an active schedule; `None` cadence is
unaffected either way. The cadence `Select` is now `disabled=` the inverse of the flag,
with a tooltip naming the config key when disabled.

AC #2 (activation delay): folded into the SAME cadence-Select tooltip (enabled branch) --
"A freshly picked cadence can take up to ~30 minutes (one scheduler reload cycle) to
start" -- plus a new bullet in Docs/User_Guide/watchlists.md's "Scheduled briefings"
section. Smallest-honest-surface choice per the AC's own wording.

AC #3 (row-scoped sweep exclusion, first arm taken): `fail_interrupted_briefings`'s
`exclude` now matches `briefings.id` (`AND id NOT IN (...)`) instead of `watchlist_id`.
Added `_ACTIVE_BRIEFING_CLAIM_ROW_IDS: dict[watchlist_id, briefing_id]`, populated by
`generate_briefing` as the very next statement after `_start_generation`'s `to_thread`
hop returns (no intervening `await`), and popped in `_claim_briefing`'s own `finally`
alongside the existing watchlist-level claim. New `active_briefing_claim_row_ids()`
snapshot accessor is what callers now pass as `exclude` (both screen call sites updated;
`active_briefing_claims()` itself is untouched and still used for its original
watchlist-granular purposes -- `GenerationInFlightError` and the scheduler's own
duplicate-dispatch guard in `briefing_handler.py`, neither of which this task touches).
`_claim_briefing` gained an optional `briefing_id` kwarg so tests that simulate a claim
directly (without a real `generate_briefing` call) can associate it with an existing row;
the two existing tests that did this (`test_a_claimed_watchlist_survives_an_artifacts_
open`, `test_generate_during_a_claimed_watchlist_refuses_without_falsifying_the_row`)
were updated to pass it, since row-scoped exclusion made their bare watchlist-only claim
insufficient to protect the row they assert on.

Residual (documented at the time, since fixed): a window existed between `_start_
generation`'s `INSERT` (its FIRST statement, inside its `to_thread` hop) and `generate_
briefing`'s coroutine resuming to record the row id -- not "right after the INSERT" as
this note originally (inaccurately) claimed, but after the WHOLE hop, three more DB reads
later. A whole-branch review of `chore/briefings-residuals-1810-1812` (verdict:
`.superpowers/sdd/briefings-residuals/whole-branch-verdict.md`, Important 1) proved the
window reachable -- a probe blocking inside `select_briefing_items` measured `swept == 1`
against the live row at that exact instant, confirming this was not theoretical. Closed
in the same branch's fix wave without splitting the `to_thread` hop: `fail_interrupted_
briefings` gained a second `exclude_watchlists` parameter, fed by a new
`pending_briefing_claim_watchlist_ids()` accessor (`active_briefing_claims() -
_ACTIVE_BRIEFING_CLAIM_ROW_IDS.keys()`) -- watchlists claimed but not yet row-recorded.
The set empties the instant the row id lands, so this task's own AC #3 coexistence fix
is untouched. See `tldw_chatbook/Subscriptions/briefing_service.py` and the fix-wave
section of `.superpowers/sdd/briefings-residuals/task-1812-report.md` for the mutation-
tested detail.

Testing: new tests in Tests/Subscriptions/test_briefing_service.py (row-id snapshot
default, and the zombie+live-claim coexistence case: zombie swept, live row untouched)
and Tests/Watchlists/test_watchlists_artifacts_pane.py (kill-switch off disables Select +
honest phrase; kill-switch on unchanged; tooltip states the activation delay; a plain
unit test of `cadence_scope_phrase`'s new kwarg). One pre-existing test
(`test_fail_interrupted_briefings_spares_a_claimed_watchlist_both_directions`) was
renamed and fixed: it had been passing `exclude={watchlist_id}` against a fresh
single-watchlist, single-row fixture where the watchlist id and the briefing row id
coincidentally both equal 1 -- silently masking the semantic change. Rewritten with a
padding row so the ids provably diverge, and to pass the row id as the new contract
requires.

Mutation checks performed and reverted (git status clean between): (1) reverted `id NOT
IN` back to `watchlist_id NOT IN` in `fail_interrupted_briefings` -- both new/updated
row-scoped tests went RED; (2) forced `schedules_disabled = False` in the pane's
`compose()` -- the kill-switch-off test went RED, the other three (on-direction,
tooltip, plain function test) stayed green as expected.

Verification: Tests/Subscriptions/test_briefing_service.py (30 passed), Tests/Scheduling/
(264 passed), Tests/Watchlists/ (364 passed, full directory) -- all via
/private/tmp/tldw-briefings/.venv/bin/python -m pytest directly.

Files: tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py,
tldw_chatbook/UI/Screens/watchlists_collections_screen.py,
tldw_chatbook/Subscriptions/briefing_service.py, Docs/User_Guide/watchlists.md,
Tests/Subscriptions/test_briefing_service.py, Tests/Watchlists/test_watchlists_artifacts_pane.py.
<!-- SECTION:NOTES:END -->
