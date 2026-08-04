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

- [ ] A run row names its source (and watchlist where applicable).
- [ ] Found/Processed/Filtered/Errors and Duration reflect what the run
      actually did (the ~30-item check shows ~30 found).
- [ ] A regression test asserts run accounting is populated from a real
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
