---
id: TASK-1510
title: >-
  Evals rail running glyph is unreachable in the press-and-watch flow
status: Done
assignee: []
created_date: '2026-07-30 14:00'
labels:
  - evals
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Whole-branch review finding (F3) of the 2026-07-30 UAT fix batch. Nothing recomposes the Catalog rail when a bench run starts — the press handler does not select, and progress ticks touch only the primary-action button — so the spec-advertised in-flight run row (`● HH:MM · name`) appears only if the user happens to click something mid-run. The button's "Running… (n/m)" label carries the signal, so this is a visibility gap, not a break. A start-of-run refresh is non-trivial: the run group row does not exist until `WordBenchRunner.run` creates it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] Starting a bench run makes the in-flight run row (● glyph) visible in the rail without further user interaction
- [x] The rail returns to the settled state on completion without breaking selection
<!-- AC:END -->

## Implementation Notes

Implemented in the Evals UX follow-up batch (fix/t1-evals, 2026-10-11).

- **AC1**: the first progress tick of a run rebuilds the rail exactly
  once per run (`_refresh_rail_once_for_active_run`, gated by
  `_rail_shows_active_run`, reset at every worker start). Waiting for
  the first tick rather than firing at press time is load-bearing: the
  run-group row does not exist until the engine creates it and stamps
  its runs "running" -- the first completed cell is the earliest moment
  the row is actually renderable. Both progress hooks
  (`_on_bench_run_progress`, `_on_sample_bench_progress`) feed it, so
  all three engines (word, sample, character) are covered. The rebuild
  re-derives `rail.sample_bench_running` first so a fresh rail instance
  keeps the Create button disabled mid-run.
- **AC2**: completion paths that `select()` already rebuild the rail
  with the settled glyph and set a real selection (the finished run
  group); the paths that do NOT select -- the failure path and the
  "user navigated elsewhere" toast-only path -- now call
  `_refresh_rail_for_settled_run()` so the ● does not linger.
  `test_the_running_rail_row_appears_without_further_interaction` pins
  both halves end to end (no select(), no clicks: the row appears on
  its own, then settles to ✓ with the selection intact).

Modified files: `tldw_chatbook/UI/Screens/evals_screen.py`,
`Tests/UI/test_evals_screen.py`.
