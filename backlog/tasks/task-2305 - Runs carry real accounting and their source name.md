---
id: TASK-2305
title: Runs carry real accounting and their source name
status: In Progress
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - bug
  - uat-2026-08-04
dependencies: []
priority: high
---

## Description (the why)

UAT: after a check that demonstrably harvested ~30 items, the Runs table
showed "Untitled · completed · Found 0 · Processed 0 · Filtered 0 · Errors 0
· Duration -". Both identity and accounting are broken: a history of
"Untitled" rows with zeroed stats is unusable and reads as if checks do
nothing.

UAT findings F32, F33 (high).

## Acceptance Criteria (the what)

- [x] A run row names its source (and watchlist where applicable).
- [x] Found/Processed/Filtered/Errors and Duration reflect what the run
      actually did (the ~30-item check shows ~30 found).
- [x] A regression test asserts run accounting is populated from a real
      check against a stub feed.

## Implementation Plan

1. Trace a real check end to end (`execute_run` -> `record_run_result` ->
   `stats_json` -> `list_runs` -> `normalize_watchlist_run` -> `RunsPane`) and
   establish whether the numbers are never written, written elsewhere, or
   dropped at display.
2. Fix identity at the query: local `list_runs`/`get_run` resolve the run's
   source name (and its watchlists) the way `list_home_run_snapshot` already
   does, through one shared row normalizer so Home and the Runs pane cannot
   drift apart again.
3. Fix accounting at the normalizer: lift the counters the pane reads
   (`found_count`/`processed_count`/`filtered_count`/`error_count`) out of the
   nested `stats` blob, and derive `duration` from `started_at`/`finished_at`
   with a `response_time_ms` fallback.
4. Record what the run actually filtered in `execute_run`'s stats rather than
   leaving it derivable-only, so the filtered count is observed rather than
   inferred.
5. Render the run row and detail block inert (`Text`), since source and
   watchlist names are user-typed and item titles are remote-derived.
6. Regression tests: a real `execute_run` against a stub feed asserting the
   normalized run carries the source name and non-zero accounting, plus
   normalizer unit tests for the derivations.
7. Verification gates + live run in a real terminal.

## Implementation Notes

The check pipeline was never at fault: it had recorded everything the UAT said
was missing. Two seams between the pipeline and the pane dropped it, one per
half of the finding.

### F33 — accounting: a key-name mismatch at the normalizer

`execute_run` writes `items_found` / `items_ingested` into the run's `stats`
blob and `record_run_result` persists it to `stats_json`. `RunsPane` reads
`found_count` / `processed_count` / `filtered_count` / `error_count` off the
run's own top level. **No normalizer had ever written those keys**, so the
four columns were `run.get("found_count") or "0"` over a dict that had no such
key — four zeros over a run that had just harvested 30 items. `duration` and
`source_title` were the same: read by the pane, written by nobody.

`normalize_watchlist_run` now lifts the counters (with the aliases a server
`stats` blob may use), derives `duration` from the run's own timestamps with
`response_time_ms` as the fallback, and gives a failed run an honest error
count — `dispositions["error"]` for the url family, else 1 when the run failed
or carries an error message, instead of the flattering 0 a missing key
produced.

`execute_run` additionally records `items_filtered` where the exclusion
happens. This is not belt-and-braces: `found - processed` is **not** the
filtering count, because `stats.setdefault("items_found", len(raw_items))`
lets an executor report what the FEED held rather than what it handed over,
and the inferred number would then be the feed's backlog. The derived form is
kept as the fallback for rows written before this task.

`duration` is `None` (rendered `-`) for a run that has not finished: an
elapsed time on a `running` row would change on every repaint and never for a
row already drawn.

### F32 — identity: only Home ever resolved the source

`local_watchlist_runs` stores a `source_id` and nothing else about the source.
`list_home_run_snapshot` resolved the name with its own hand-written JOIN;
`list_runs` and `get_run` — the Runs pane's actual path — did not. So the Runs
tab said "Untitled" for every run while Home, reading the same table, showed
the real name.

All three local run reads now share one `_RUN_SELECT` projection and one row
normalizer (`_normalize_run_row`), which is what stops them drifting apart
again. The projection also carries the watchlists the run's source belongs to,
as one `group_concat` column rather than a query per run, joined with ASCII
UNIT SEPARATOR — watchlist names are user-typed and a comma in one would
otherwise split it into two watchlists that do not exist. `LEFT JOIN`, not
`JOIN`: a run whose source cannot be resolved must still be listed.

The row renders `Source · Watchlist`, first watchlist plus `+N` — `DataTable`
sizes a column to its widest cell, so an unbounded join would push the eight
accounting columns off the side of the pane. The detail block, which has no
width constraint, lists them all.

### Deliberately not done

Timestamp FORMAT. `Started` still shows the raw ISO string; **task-2308 owns
that sweep** and doing it here would collide with it.

### Verification

* `Tests/Subscriptions/test_local_watchlists_service.py` +4 (a real
  `execute_run` against a stub feed asserting the record `list_runs` returns —
  not the nested blob that was never the problem; a failed run; a queued run;
  and the executor-reports-the-feed's-total case that makes the recorded
  filtered count load-bearing).
* `Tests/Subscriptions/test_watchlist_normalizers.py` +13 (counter lifting,
  the derived-filtered fallback, the malformed-blob floor, error-count
  fallbacks, four duration scales, the response-time fallback, the separator,
  and a server run carrying no source name).
* `Tests/Watchlists/test_watchlists_runs_pane.py` +5 (row identity, the `+N`
  abbreviation, the unresolvable-source case, the detail block's full list,
  and inert rendering of a markup-bearing source name).
* Suites: `Tests/Subscriptions/` + `Tests/Watchlists/` + `Tests/Scheduling/` +
  `Tests/Home/test_active_work_adapter.py` + `Tests/DB/
  test_subscriptions_db_watchlists.py` **1338 passed**; the watchlists UI
  sweep **94 passed** (plus one PRE-EXISTING failure, see below); poisoned
  order **50 passed**; `--collect-only Tests/UI Tests/Watchlists` **8699
  collected**, no errors.
* **11 mutations**, each reverted individually → RED → restored byte-exact
  (md5-verified). Two survived their first tests and the tests were
  strengthened until each was load-bearing: the recorded `items_filtered`
  (the derived fallback gives the same answer unless the executor overrides
  `items_found`) and Home's `title` mirror (the rail falls back to
  `source_title`, so it needed pinning at the service).

`Tests/UI/test_watchlists_check_now_failure.py::…[_delete_item]` fails on this
branch and **also on `origin/dev`** — the test asserts a method
`_delete_item` that the screen does not define there either
(`git show origin/dev:…watchlists_collections_screen.py` has no such `def`).
Untouched by this task.

Two pinned assertions were updated on purpose:
`test_stats_text_without_dispositions_key_is_unchanged` (which pins the
absence of an empty `Checks:` line, not the absence of a `Source:` line) and
the Home-snapshot test, which now pins `title`.

### Live verification (fresh profile, real `https://hnrss.org/frontpage`)

Watchlist + source created through the UI, assigned, checked. The Runs table:

```
Source / Job                Status     Started                           Duration  Found  Processed  Filtered  Errors
Hacker News · Morning read  completed  2026-08-04T23:50:29.582049+00:00  490ms     20     20         0         0
Hacker News · Morning read  completed  2026-08-04T23:49:47.155058+00:00  453ms     20     20         0         0
Hacker News · Morning read  completed  2026-08-04T23:48:18.732138+00:00  596ms     20     20         0         0
```

against the UAT's `Untitled · completed · Started <raw ISO> · Duration - ·
Found 0 · Processed 0 · Filtered 0 · Errors 0`.

### Files

* `tldw_chatbook/Subscriptions/watchlist_normalizers.py` — the lift, the
  duration derivation, the separator.
* `tldw_chatbook/Subscriptions/local_watchlists_service.py` — `_RUN_SELECT`,
  `_normalize_run_row`, the recorded `items_filtered`.
* `tldw_chatbook/UI/Watchlists_Modules/runs_pane.py` — `_run_identity` and the
  detail block's Source/Watchlists lines.
