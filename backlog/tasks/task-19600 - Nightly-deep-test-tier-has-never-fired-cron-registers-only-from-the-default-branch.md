---
id: TASK-19600
title: >-
  Nightly deep test tier has never fired: cron registers only from the
  default branch
status: To Do
assignee: []
created_date: '2026-08-21 18:15'
labels:
  - ci
  - testing
  - infrastructure
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Tests workflow's nightly deep tier (`cron: '30 8 * * *'`, added to `dev`
on 2026-07-31 by task-1465) **has never executed a single time**. GitHub
registers scheduled workflows only from the repository's DEFAULT branch;
this repo's default is `main`, and `main`'s copy of
`.github/workflows/test.yml` predates the change and carries no `schedule:`
block at all. The cron therefore exists only on a branch where GitHub never
reads it.

Evidence: `gh run list --workflow Tests --event schedule` returns ZERO runs
across the three weeks since it was added.

This matters because it is the second half of a two-part design: the PR run
is the merge gate, and the nightly deep tier was meant to be the periodic
whole-suite verdict on `dev` (serial, `--run-slow`, thorough Hypothesis
profile, cache-off, Windows+macOS breadth -- coverage the PR gate
deliberately skips). Only the gate half has ever run.

Compounding it, the `dev` PUSH run cannot substitute: `cancel-in-progress`
is true for every non-`main` ref and merges land on `dev` every 20-40
minutes against an ~80-minute run, so each merge kills the run the previous
merge started. Measured 2026-08-21: of the last 40 Tests runs, 25 cancelled
and 15 in flight -- ZERO completed. So `dev` currently has no automatic
post-merge signal from either mechanism.

The one path that does work today is manual `workflow_dispatch` against
`dev`: it forms its own concurrency group (the group key includes
`github.event_name`), so pushes cannot cancel it, and the nightly job is
already gated on `schedule || workflow_dispatch`. That is how a `dev`
verdict was obtained on 2026-08-21 (run 32511976568).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A scheduled Tests run appears in the Actions history without anyone triggering it manually (`--event schedule` returns a run).
- [ ] #2 The mechanism does not depend on remembering to dispatch by hand.
- [ ] #3 `dev` push runs either produce a usable verdict or stop consuming runners for runs that are structurally guaranteed to be cancelled.
- [ ] #4 The workflow shape contract test pins whatever mechanism is chosen, so this cannot silently rot again.
<!-- AC:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
Three options, all owner decisions -- deliberately NOT chosen here:

1. **Merge `dev` -> `main`.** Fixes it as a side effect and is presumably
   wanted eventually, but it is a release action with far wider blast
   radius than this task.
2. **Put a small dispatcher workflow on `main`** whose only job is
   `workflow_dispatch`-ing Tests against `dev` on a cron. Restores the
   nightly tier without releasing the rest of `dev`. Narrowest fix.
3. **Accept manual dispatch** as the periodic mechanism and delete the
   inert `schedule:` block so the workflow stops advertising a tier that
   cannot run.

Note that `Tests/CI/test_github_actions_test_workflow.py` asserts
`"- cron:" in workflow`, which passes on `dev` while the schedule is inert
-- a contract test that pins the TEXT but not the EFFECT. Whichever option
is chosen, AC#4 should close that gap.

Separately, if the `dev` push trigger is kept, it needs either its own
non-cancelling concurrency group (with the caveat that ~80-minute runs
arriving every 20-40 minutes would QUEUE unboundedly) or removal in favour
of the PR gate, which already tests the merge commit.
<!-- SECTION:NOTES:END -->
