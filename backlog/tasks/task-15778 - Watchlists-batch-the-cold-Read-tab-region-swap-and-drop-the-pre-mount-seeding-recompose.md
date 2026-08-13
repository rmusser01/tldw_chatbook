---
id: TASK-15778
title: 'Watchlists: batch the cold Read-tab region swap and drop the pre-mount seeding recompose'
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - perf
  - watchlists
priority: low
---

## Description

Two related residuals recorded "for the controller to file" in task-15461's
Implementation Notes (input-latency burn-down's Watchlists scoped-rebuild
work). Both are about construction-order cost inside the same region-build
plumbing task-15461 converted from whole-screen recomposes to scoped
rebuilds.

1. **Cold Read-tab wall-clock did not improve despite halving DOM work.**
   Measured 75 -> 110 ms (best-of-two) for the one section switch
   (`section: Read`, cold) that has to re-mount the CONTENT region. Every
   other measured section switch improved with task-15461's scoped rebuild;
   this one regressed on wall clock because the scoped path does the CONTENT
   remount as its own discrete remove/mount pair rather than inside one
   batched recompose. Task-15461's own notes point at Textual's `batch()` as
   "the obvious next move."
2. **`_build_detail_pane`'s pre-mount seeding costs one pane recompose per
   region build.** `[] != [row]` on a freshly constructed pane triggers an
   extra recompose; pre-existing and unchanged by task-15461, invisible on
   an empty fixture (which is why it surfaced only once task-15461's review
   asked for a seeded row). Fixing it means seeding with `set_reactive`
   instead of a plain assignment, which is not safe blind: `RunsPane`'s
   seeding ORDER is load-bearing (`selected_run` clears the detail, so the
   detail must be set after it) — any fix must preserve that ordering
   per-pane, not apply one blanket change.

## Acceptance Criteria

- [ ] The cold `section: Read` switch's CONTENT-region remount is batched
      (e.g. via Textual's `batch()`) into the same pass as its other DOM
      work, and the wall-clock regression measured in task-15461 is closed
      (before/after recorded)
- [ ] `_build_detail_pane`'s pre-mount seeding uses `set_reactive` (or an
      equivalent that avoids the extra recompose) without breaking any
      pane's load-bearing seeding order — `RunsPane`'s `selected_run`-before-
      detail ordering explicitly verified by test
- [ ] Every other pane that uses `_build_detail_pane` is checked for its own
      seeding-order dependencies before the change lands, not just
      `RunsPane`
- [ ] `Tests/Watchlists/test_watchlists_scoped_rebuilds.py` and the sources/
      rules pane suites stay green
