---
id: TASK-16838
title: 'Watchlists: per-(subscription,url) in-flight guard against concurrent double-reporting'
status: Done
assignee:
  - '@claude'
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

## Implementation Plan

1. Verify the single-loop claim: every entrant that can reach `check_url`
   (scheduler worker, Check Now / Rerun coroutine workers) runs on the app's
   event loop; shadow mode writes nothing (`persist_snapshots=False`) and
   preview runs against a throwaway in-memory DB.
2. Guard shape: a module-level in-flight set in `local_watchlists_service.py`
   keyed `(id(db), subscription_id, url)` — module-level because the
   scheduler's handler and the UI hold two different `LocalWatchlistsService`
   instances over the one shared `SubscriptionsDB` (app.py task-15463
   wiring); `id(db)` scopes the key so preview's throwaway DB can never
   collide with the live one. No lock: claim/release are synchronous
   between awaits on the one loop.
3. Placement: a `_check_url_guarded` helper at the `check_url` orchestration
   seam in `_default_run_executor`, wrapping all three url-family arms
   (`url`, `url_list`, `sitemap`), released in `finally`.
4. Second entrant: SKIP with an INFO log and an honest, caller-synthesized
   disposition (`DISPOSITION_SKIPPED_IN_FLIGHT`, same pattern as
   `DISPOSITION_ERROR`), surfaced as a `skipped` run-stats counter, a
   conditional Runs-pane detail segment, and an honest Check Now toast for
   an entirely-skipped manual run.
5. Evidence: born-red deterministic interleave test (gated fetch, scheduled
   + manual entrants for the same source, assert one report/one snapshot and
   the skip), distinct-sources concurrency pin, no-strand pins (failure and
   cancellation), duplicate-URL-in-one-run no-self-skip pin; affected suites
   re-run against the green 1530-test baseline.

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A scheduled run overlapping a manual Check Now of the same source cannot double-report one page change or write two baseline snapshots for it (test forcing the interleave as evidence)
- [x] #2 Distinct sources still check concurrently exactly as before (no global serialization)
- [x] #3 The guard cannot strand a source as permanently "in flight" after a failed or cancelled check
- [x] #4 The skipped entrant is reported honestly, not silently: the run records a `skipped` disposition count, the Runs pane detail names it, and an entirely-skipped manual Check Now toasts that a check is already running instead of "0 found, 0 new"
<!-- AC:END -->

## Implementation Notes

**Guard shape and placement.** A module-level in-flight set
(`_IN_FLIGHT_URL_CHECKS` in `local_watchlists_service.py`) keyed
`(id(db), subscription_id, url)`, claimed in the new
`LocalWatchlistsService._check_url_guarded` and released in `finally`. That
helper is the choke point all three url-family arms of
`_default_run_executor` (`url`, `url_list`, `sitemap`) now route through, so
every entrant that can WRITE is covered: the scheduler's
`WatchlistCheckHandler`, the UI's Check Now / Rerun, and any direct
`launch_run`/`execute_run` caller. Module-level, not instance state, because
production wires TWO service instances over the one shared `SubscriptionsDB`
(app.py task-15463: the UI's `local_watchlists_service` plus the handler's
own default-constructed one); `id(db)` in the key keeps
`WatchlistPreviewService`'s throwaway in-memory DB (whose row ids can
collide with live ones) out of the live registry. No lock: the single-loop
claim was verified — the scheduler is a coroutine worker
(`run_worker(self.scheduler_loop.run(), ...)`), Check Now/Rerun are
coroutine workers, and claim/release are synchronous between awaits.
Shadow mode's direct `check_url` probe is deliberately outside the guard:
`persist_snapshots=False` means it cannot write, and the scheduler loop
already serializes shadow probes against each other.

**Second entrant = SKIP, honestly surfaced.** The loser returns a new
caller-synthesized `DISPOSITION_SKIPPED_IN_FLIGHT` (same pattern as
task-1394's `DISPOSITION_ERROR`), which lands in a new zero-filled `skipped`
run-stats counter, an INFO log, a conditional Runs-pane detail segment
("N skipped (check already running)"), and — for an ENTIRELY skipped manual
run — a "Check skipped: … already running" toast instead of the false-clean
"Check complete — 0 found, 0 new". `skipped` is excluded from
`_SUCCESS_DISPOSITION_COUNTERS` (a skip proves nothing about reachability,
so it cannot mask an all-error run's breaker advance). Same-run duplicate
URLs neither deadlock nor self-skip: the url_list/sitemap loops await each
check, so the claim is free before the loop reaches a duplicate (pinned).

**Evidence.** `Tests/Subscriptions/test_watchlist_check_in_flight_guard.py`
run against pre-fix HEAD `1af8c0414` in a throwaway worktree: 5 failed —
stated precisely, ONE of the five (the headline interleave test) reddened on
the behavior itself, "a manual Check Now overlapping a scheduled check of
the same source went to the network too"; the other four reddened on the
then-absent `skipped` key in their exact-dict assertions, i.e. vocabulary
pins, not behavioral reproductions. (A first attempt that reddened only on
ImportError was rewritten with a lazy registry lookup — lesson filed in
`lessons-testing-evidence.md`.) The pre-guard damage shape reproduced at
base is fetches=2, 2 snapshots, 1 item, with `changed`/`unchanged` split
nondeterministically between the two runs — a Check Now could answer
"0 found" for a change the other run got; the 15764 review's own
`['changed','changed']`/two-snapshot split is credited to that review, not
reproduced here. With the fix: the manual entrant completes `skipped`
without fetching while the scheduled fetch is still gated; one item, one
new snapshot. Affected suites: baseline 1530 passed/1 skipped at HEAD;
post-fix 1565 passed/1 skipped (Tests/Subscriptions, the two Scheduling
watchlist files, Tests/Watchlists, the three check-now UI tests — including
the 15764-ported thread-identity file). Ruff clean on every touched file.

**Scope note.** AC #4 (honest surfacing of the skip) was ADDED during
implementation — the filed task had three ACs; the parent brief's
second-entrant UX decision point called for reading Check Now's result
surface and keeping it honest, and the toast/counter/Runs-segment work is
that decision. Rerun and check-all keep their generic reporting (review F5,
disclosed follow-up scope); the Runs-pane segment covers those runs.

**Review fix wave (independent review, verdict FIX-FIRST; scratchpad
`review16838.md`).** The guard mechanism held every attack (no leak path,
choke point complete, single-loop verified, preview scoping correct); one
blocker downstream plus honesty items, all addressed:

- **B1 (blocker)**: an entirely-skipped run travelled `execute_run`'s
  ordinary completion accounting — zero `error` dispositions made
  `_all_error_check_message` return None, routing it into
  `record_check_result`'s SUCCESS branch: auto-pause breaker reset,
  `last_error` cleared, `last_successful_check` stamped, and a `no_items`
  alert fired, all by a run that contacted nothing (reviewer's probe: a
  2-failure streak wiped to clean by a skip). Fixed with option (a): new
  `_entirely_skipped_dispositions` helper + a short-circuit in
  `execute_run` — the run ROW still persists with its skipped stats (Runs
  pane honesty), but `record_check_result`, stats inflation, and alert
  evaluation are all skipped (`record_run_result` grew
  `evaluate_alerts: bool = True`, stronger than the existing
  `dispatch_notifications` flag which still evaluated). Born-red pin
  (reviewer's probe shape) written FIRST and observed red at the guard
  commit `72b67f25f` (`consecutive_failures 2 -> 0`) before the fix:
  `test_an_entirely_skipped_run_leaves_the_sources_health_row_untouched`.
  The `_SUCCESS_DISPOSITION_COUNTERS` comment's now-false "an
  entirely-skipped run is unaffected either way" sentence corrected.
- **F2**: new `Tests/Subscriptions/test_app_watchlists_db_wiring.py` pins
  the production invariant the id(db) keying rests on — the UI service and
  the scheduler handler's default service resolve to the ONE
  `app.subscriptions_db` object — so a wiring regression to per-instance
  DBs cannot leave every guard test green while the guard silently stops
  guarding.
- **F3**: the registry comment now states the true liveness mechanism —
  the set entry is `(int, int, str)` with no reference; it is
  `_check_url_guarded`'s own frame holding `db` across the await that
  prevents id reuse, and a key-only helper extraction would drop that.
- **F4**: single-loop caveat lines added at the three launch sites that
  could break the invariant (Check Now and Rerun `run_worker` sites in
  `watchlists_collections_screen.py`, the scheduler worker in `app.py`).
- **F6**: user-guide wording — the winner's result "appears in the Runs
  section the next time the screen refreshes" (nothing pushes it live).
- **F7**: born-red and damage-shape claims restated precisely here and in
  the test file's comments (see Evidence above).

**Files.** `tldw_chatbook/Subscriptions/monitoring_engine.py` (new
disposition constant), `tldw_chatbook/Subscriptions/local_watchlists_service.py`
(registry, guard, counter vocabulary), `tldw_chatbook/UI/Watchlists_Modules/runs_pane.py`
(conditional detail segment), `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
(`_check_was_entirely_skipped` + skip toast), `Docs/User_Guide/watchlists.md`
(new section), tests: new `Tests/Subscriptions/test_watchlist_check_in_flight_guard.py`
and `Tests/UI/test_watchlists_check_now_skipped.py`; counter-vocabulary pins
updated in `Tests/Subscriptions/test_local_watchlists_service.py`,
`test_watchlist_noise_not_volume.py` (whose `_counts` also serves
`test_watchlist_snapshot_pruning.py`), and `Tests/Watchlists/test_watchlists_runs_pane.py`.
