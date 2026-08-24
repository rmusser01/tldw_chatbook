---
id: TASK-21411
title: The core test job cannot finish and the summary check cannot fail
status: Done
assignee: []
created_date: ''
updated_date: '2026-08-24 00:28'
labels:
  - ci
  - testing
  - test-integrity
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The repository's test workflow has not produced a verdict since 2026-06-26. Two independent
faults keep it that way, and neither is visible from the outcome alone.

The core job — everything outside the UI directory, roughly three quarters of the test files
— has never finished. It is killed by its own time budget on every run, and because GitHub
reports a timeout as a cancellation rather than a failure, this reads from the outside like
a run being superseded by a newer push. It is not. The job is simply larger than any single
container the budget allows, and the fix is to split it the way the UI job was already split
for the same reason.

Independently, the summary check that aggregates the test jobs cannot fail. It is scheduled
to run after them regardless of what they did, but it never inspects what they did — so it
reported success on a run where every one of the twelve UI shards was red. That is the check
most likely to be marked as required, which makes a summary that cannot go red worse than no
summary at all.

Fixing these does not make the suite green. It makes the suite legible, which is the
precondition for anything else.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The core suite runs in containers it can finish in, sized from measured evidence rather than from raising a cap again
- [x] #2 The diagnosis distinguishes "too slow" from "stalled" on evidence, because the two have different fixes and the outcome looks identical
- [x] #3 The split covers every test exactly once and stays deterministic, and parallelism within each slice is retained
- [x] #4 The change does not worsen contention for the scarcest runner pool, and any coverage it moves is shown to still exist elsewhere
- [x] #5 The summary check fails when a job it gates on failed, and still publishes its report when that happens
- [x] #6 The workflow-shape test is updated in the same change and each new assertion is proven to fail when what it pins is removed
- [x] #7 Anything that cannot be fixed from a branch to the development line is stated as such rather than left implied
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read a killed job's own log to establish whether it was progressing or stalled at the kill.
2. Size the split from how far it got and the runner it got there on.
3. Split the job; keep the deterministic slicing contract the UI job already uses.
4. Give the summary check a verdict step, placed so the report is still published.
5. Update the shape test in lockstep, then mutate each new assertion to prove it bites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Split `core-tests` six ways with pytest-shard, and gave `test-summary` a verdict step.

**AC#2 — too slow, or stalled?** These look identical from the outcome and have different
fixes, so it was read from the killed job's own log (run 32647831275, ubuntu leg). The job
started at 15:39:14, pytest began emitting at 15:44:55 after ~5.7 min of checkout and
install, and was **still printing progress steadily** — no stall, no gap — when the
120-minute cap killed it at 17:39 having reached **60%**. Steady progress at the kill means
too slow, not hung. Extrapolated: ~190 minutes on a 4-vCPU runner (public repos get 4, so
`-n auto` is 4 workers; the same work takes 30.8 min locally on 8 faster cores).

This also **corrects the baseline document's guess**, which said a hang was likely because
190 min seemed too far from 30.8. The runner arithmetic accounts for it without a hang.

**AC#1/#3 — the split.** Six shards at ~32 min each, roughly a quarter of the budget, so the
cap is headroom rather than a target. Verified empirically rather than assumed: collecting
each shard and unioning them gives **42,811 node ids, zero overlap between shards, and a set
identical to the unsharded collection**. Nothing is dropped and nothing runs twice. xdist
still parallelizes within each slice.

(An early version of that check reported 35 duplicates. They were seven
`PytestCollectionWarning` location lines repeated in every shard's warning summary, which a
`^Tests/` grep caught alongside real node ids. Real ids contain `::`.)

**AC#4 — runner contention.** ubuntu only. macOS is the scarce pool here: on run
32673270908 the macOS core job waited **42 minutes** to start, and in an earlier run a UI
shard waited ~90. Sharding the scarcest pool six ways would spend more in queueing than it
buys. macOS breadth already lives in nightly-deep, and
`test_core_tests_job_does_not_multiply_the_scarcest_runner_pool` pins both halves of that
trade — ubuntu-only here, macOS still present there.

**AC#5 — the summary.** It declared `needs:` and `if: always()`, which schedules it after
those jobs whatever they did; it never inspected what they did, and reported `success` while
all twelve UI shards were red. The verdict is now its own step, placed **last** so a red
suite still leaves the PR comment behind rather than exiting before it is posted.

**AC#6 — the shape test, and a finding inside it.** Three new tests, each mutation-proven:
removing the verdict step, breaking the shard partition to ids 1..6, moving the verdict ahead
of the PR comment, and putting macOS back — each fails exactly one test, and the restored
tree is 18/18 green.

Fixing an existing test was itself a finding. `test_ui_job_is_sharded_to_fit_its_time_budget`
located the shard id list by searching the **whole workflow** for the first `shard: [`, which
was the UI job only because it was then the only sharded job. With core sharded it began
checking core's ids against the UI job's divisor. A partition assertion that can silently
start describing a different job is worse than no assertion; it now reads from the UI job
block.

**AC#7 — what this cannot fix.** `nightly-deep` has still never run, and no branch to `dev`
can change that. GitHub registers `schedule:` only from the **default branch**, which is
`main`; `main`'s workflow has no schedule block, and `main` is **11,528 commits behind
`dev`**. Since cron would execute *main's* workflow, main would have to carry the
nightly-deep job itself. That is a decision about `main`, not a workflow edit, and is left to
the owner. Until it is made, `--run-slow` (41 marks), the thorough Hypothesis profile, the
CSS-cache-off soak, Windows, and Python 3.11/3.13 run nowhere.

This change does not make the suite green. It makes it legible, which everything else needs.

Modified: `.github/workflows/test.yml`, `Tests/CI/test_github_actions_test_workflow.py`.
<!-- SECTION:NOTES:END -->
