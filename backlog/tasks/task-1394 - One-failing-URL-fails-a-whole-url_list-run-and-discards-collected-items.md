---
id: TASK-1394
title: One failing URL fails a whole url_list run and discards collected items
status: Done
assignee: []
created_date: '2026-07-30 05:20'
labels:
  - watchlists
  - correctness
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`local_watchlists_service.py:894-921`: the `url_list`/`sitemap` arms loop URLs with no per-URL
`try/except`, so one failing URL (timeout, SSRF block, HTTP error) raises out of the loop, fails
the whole run via `record_run_failure`, and discards the items already collected from the URLs
that succeeded.

Pre-existing, found during TASK-1362's whole-branch review; newly load-bearing because per-URL
baselines and dispositions (TASK-1361/1362) make large multi-URL sources more useful, and a
50-URL source with one dead link currently yields nothing at all.

Design note: isolating failures per URL wants an error disposition (the current vocabulary is
changed/unchanged/withheld/baseline/rebaselined) so a partially-failed run says so in the Runs
pane rather than reporting clean counts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A url_list run with one failing URL persists the items and dispositions from the URLs that succeeded
- [x] #2 The failure is visible per run (an error count or disposition in the Runs detail), not silently absorbed
- [x] #3 A test with one poisoned URL among several fails under the old all-or-nothing behaviour
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. `_default_run_executor` (`local_watchlists_service.py`): wrap each `monitor.check_url(...)` in the
   `url_list` AND `sitemap` loops in try/except. On exception, append a NEW `error` disposition
   (type-only, no URL/content in the recorded value) and continue — never add to `items`, never
   raise out of the loop. Items/dispositions from the URLs that succeeded persist (AC#1).
2. Add a 6th `error` counter to `_ALL_DISPOSITION_COUNTERS` (:68) + `_disposition_counts` mapping so
   a partially/fully-failed run reports its error count in `run_stats["dispositions"]` (AC#2).
3. Render the error count in `runs_pane.py`'s Checks line (~:162) alongside changed/unchanged/etc.
4. Tests: one poisoned URL among several → succeeded items persist + error counter = 1 (AC#1/#2);
   the same test REDs without the try/except (poisoned URL raises, whole run fails — AC#3).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extracted the per-URL `monitor.check_url(...)` call (identical in both the `url_list` and
`sitemap` loops of `_default_run_executor`) into a new static helper,
`LocalWatchlistsService._check_url_isolated`, so both arms share one `try/except Exception`
instead of duplicating it. On success it returns `check_url`'s result unchanged; on any exception
it logs `type(exc).__name__` only (never the message or the URL — both can carry fetched page
content or a sensitive query string) and returns `(None, {"kind": DISPOSITION_ERROR, "reason":
None, "withheld_percentage": None})`, letting the caller's loop treat it exactly like any other
disposition and move on to the next URL. Neither loop's item list nor its dispositions list is
touched beyond that — everything the OTHER urls collected persists (AC#1).

Added `DISPOSITION_ERROR = "error"` to `monitoring_engine.py` alongside the other four
`DISPOSITION_*` constants (with a docstring noting it's synthesized by the caller around a
`check_url` call that never returned, not one of `check_url`'s own outcomes), appended `"error"`
to `_DISPOSITION_COUNTERS` (last, so the five pre-existing counters keep their position for
anything that might read the tuple positionally — nothing currently does, checked), and added
`(DISPOSITION_ERROR, None): "error"` to `_disposition_count_keys()`. `reason` is fixed at `None`
for this pair, deliberately: the counter answers "how many URLs errored", not "which exception",
so one stable pair covers every exception type rather than needing one entry per type (which
would defeat the existing "unlisted pair raises KeyError" safety net for the other five kinds).
`_max_withheld_percentage` needed no change — it already filters on `withheld_percentage is not
None`, and the error disposition's is `None`.

Rendered the error count in `runs_pane.py`'s `_stats_text` Checks line, appended after
`re-baselined`, unconditionally (matching how `changed`/`unchanged`/`baseline`/`rebaselined`
already render regardless of count — only the withheld-percentage suffix is conditional on the
existing convention, and that convention is untouched).

The sitemap arm's URL-producing fetch (`_urls_for_sitemap`, an `await` evaluated once before the
`for` loop starts) is deliberately NOT wrapped by `_check_url_isolated` or any new try/except: a
failure fetching the sitemap itself still fails the whole run exactly as before, because there is
no per-URL work to isolate at that point.

Whole-run posture: a run with >=1 URL error and >=1 success completes normally (`record_run_result`,
not `record_run_failure`) with the successful items/dispositions persisted and `dispositions.error`
counting the failures. A run where every URL errors also completes normally rather than hard-
failing -- `items` is empty and `dispositions.error` equals the URL count, which is an honest,
visible report per AC#2 rather than a fabricated "clean, nothing found" run. No special-cased
hard-fail path was added for the all-errors case; the task's own AC#2 treats "the failure is
visible" as sufficient, and the run failing outright would put it back at pre-fix behaviour for a
50-URL source whose FIRST url happens to be dead.

Tests added (`Tests/Subscriptions/test_local_watchlists_service.py`):
- `test_local_watchlists_service_url_list_isolates_one_failing_url` -- the AC#1/#3 discriminator: 3
  URLs, the middle one raises `TimeoutError`, the other two persist their items and the run
  completes with `dispositions == {..., "error": 1}`. Verified this REDs under the pre-fix
  all-or-nothing behaviour by temporarily stripping the `try/except` from `_check_url_isolated`
  and re-running (`status` came back `"failed"` instead of `"completed"`), then restored the fix.
- `test_local_watchlists_service_sitemap_isolates_one_failing_url` -- same shape for the sitemap
  arm, with a poisoned `ConnectionError` mid-list. Also confirmed RED under the same mutation.

Tests added (`Tests/Subscriptions/test_watchlist_noise_not_volume.py`):
- Extended `test_disposition_count_keys_are_bound_to_the_real_constants` with the sixth
  `(DISPOSITION_ERROR, None)` pair.
- `test_removing_the_error_mapping_keyerrors_instead_of_miscounting` -- mutation guard: monkeypatches
  `_disposition_count_keys` to drop the error pair and asserts `_disposition_counts` raises
  `KeyError` on an error disposition, proving the mapping is load-bearing rather than decorative.
- `_NO_DISPOSITIONS` (the shared zero-fill fixture reused by `Tests/Subscriptions/
  test_watchlist_snapshot_pruning.py` and `Tests/UI/test_watchlists_inspector.py` via `_counts`) now
  includes `"error": 0`.

Tests added (`Tests/Watchlists/test_watchlists_runs_pane.py`):
- `test_stats_text_shows_the_error_count_for_a_partially_failed_run` -- asserts the rendered Checks
  line includes `"1 error"` for a partially-failed run and `"0 error"` for a clean one.
- `_disposition_run`'s fixture dict now includes `"error": 0` by default.

Existing tests updated for the new 6-key shape (the only permitted edits to pre-existing tests,
per the task brief): the two hardcoded 5-key `dispositions` dict literals in
`test_local_watchlists_service.py`'s url_list/sitemap execution tests now include `"error": 0`.

Verified: the new/updated tests, full `Tests/Subscriptions/` + `Tests/Watchlists/` (991 passed),
`Tests/Scheduling/` (264 passed, unaffected -- it drives the real `check_url` through the scheduled
handler with no failing URLs in its fixtures), and `Tests/UI/test_watchlists_inspector.py` (35
passed, uses the same `_counts`/`_dispositions` fixtures).

Modified files: `tldw_chatbook/Subscriptions/local_watchlists_service.py`,
`tldw_chatbook/Subscriptions/monitoring_engine.py`, `tldw_chatbook/UI/Watchlists_Modules/
runs_pane.py`, `Tests/Subscriptions/test_local_watchlists_service.py`, `Tests/Subscriptions/
test_watchlist_noise_not_volume.py`, `Tests/Watchlists/test_watchlists_runs_pane.py`.

### Fix wave (whole-branch review, Finding #1 -- MAJOR)

The review caught a real regression the "whole-run posture" paragraph above missed: it reasoned
about run-level visibility only, not the subscription's own health tracking. `execute_run`'s
success path called `db.record_check_result(source_id, items=None, stats=stats)` with `error=None`
UNCONDITIONALLY -- including for a run where every single URL errored. That call's success branch
(`DB/Subscriptions_DB.py:1504-1517`) resets `consecutive_failures`/`error_count` to 0 and clears
`last_error` on every run, so a permanently dead `url_list`/`sitemap` source (every URL down) could
never accumulate toward `auto_pause_threshold` and auto-pause -- pre-task-1394 it would have raised
and reached `record_run_failure` -> `record_check_error`, advancing the breaker normally. The
isolation fix silently defeated the circuit breaker for exactly the "everything is dead" case it
should matter most for.

Fix: added `_all_error_check_message(dispositions_counts, item_count)` in
`local_watchlists_service.py`, called from `execute_run` right before `db.record_check_result`. It
reads the SAME `stats["dispositions"]` counters the run already produced -- returns a type-only
synthetic message (`"all {n} checked URL(s) failed"`, counts only, no URL/exception text, matching
`_check_url_isolated`'s own type-only logging) when `error` count > 0 AND every one of the five
success counters (`changed`/`unchanged`/`withheld`/`baseline`/`rebaselined`) is 0 AND zero items
were produced; `None` otherwise. Passing that message as `record_check_result`'s `error=` argument
routes the run through the DB's error branch instead of its success branch, so the breaker
ADVANCES and auto-pause fires at threshold exactly as pre-task-1394. The circuit-breaker semantics
this settles on: **an all-error run advances the breaker (and can auto-pause); a partial run
(>=1 successful check) resets it, same as a fully clean run** -- a source that made ANY real
progress is healthy, only a source that produced nothing but errors is not. Feed/API-arm runs
carry no `dispositions` key at all (`stats.get("dispositions")` is `None`), so they are structurally
unaffected and keep resetting the breaker on every non-raising run, unchanged.

Run status wiring: chose to compute this in `execute_run` (not inside `_default_run_executor`)
because `execute_run` is the one place that already has `raw_items`, `stats`, and `source_id`
together, and doing it there makes the behaviour apply to ANY executor (including a test's custom
`run_executor=`) that produces a `dispositions` stats shape matching the pattern, not only the
built-in `url_list`/`sitemap` arms. When `_all_error_check_message` returns non-`None` and the
executor's own status would otherwise default to `"completed"`, `execute_run` overrides it to
`"failed"` -- more honest than "completed" with zero items -- and folds the synthetic message into
`error_msg` too (only when the executor didn't already supply one). A partial run's status and
`error_msg` are untouched. The dispositions/stats payload is identical either way; only `status`,
`error_msg`, and the DB breaker call change.

Tests added (`Tests/Subscriptions/test_local_watchlists_service.py`):
- `test_local_watchlists_service_url_list_all_error_advances_breaker_and_pauses` -- pre-sets
  `consecutive_failures = auto_pause_threshold - 1` on a `url_list` source, both URLs raise;
  asserts `status == "failed"`, `dispositions == {..., "error": 2}`, `consecutive_failures`
  advanced to the threshold, `is_paused == 1`, and `last_error` set. Mutation-verified: reverted
  `_all_error_check_message` to always `return None` and re-ran -- RED
  (`assert 'completed' == 'failed'`, breaker would have reset instead of advancing); restored the
  fix and confirmed green again, `git status --short` clean throughout.
- `test_local_watchlists_service_url_list_partial_error_still_resets_breaker` -- one URL succeeds
  (item persists), one errors, `consecutive_failures` pre-set to 5; asserts the item persisted,
  `dispositions["error"] == 1`, `status == "completed"`, and `consecutive_failures`/`error_count`
  reset to 0 with `last_error is None` -- pins that the fix does not over-correct into failing
  partial runs. Stayed green under the same mutation (confirming it discriminates the two cases
  rather than accidentally passing regardless).

Verified: both new tests plus all 17 pre-existing `test_local_watchlists_service.py` tests (19
passed); `Tests/Watchlists/test_watchlists_runs_pane.py` (16 passed, unaffected);
`Tests/Scheduling/test_scheduled_watchlist_runs.py` (13 passed, unaffected -- its auto-pause tests
use `type="url"`, which this fix wave does not touch); `Tests/Subscriptions/ -k "disposition or
url_list or pause or breaker or failure"` (27 passed); `--collect-only Tests/Subscriptions
Tests/Watchlists Tests/Scheduling` (1257 collected, no errors).

Modified files (fix wave): `tldw_chatbook/Subscriptions/local_watchlists_service.py`,
`Tests/Subscriptions/test_local_watchlists_service.py`.
<!-- SECTION:NOTES:END -->
