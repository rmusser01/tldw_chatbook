---
id: TASK-21411
title: >-
  The core test job cannot finish and the summary check cannot fail
status: In Progress
assignee: []
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
- [ ] #1 The core suite runs in containers it can finish in, sized from measured evidence rather than from raising a cap again
- [ ] #2 The diagnosis distinguishes "too slow" from "stalled" on evidence, because the two have different fixes and the outcome looks identical
- [ ] #3 The split covers every test exactly once and stays deterministic, and parallelism within each slice is retained
- [ ] #4 The change does not worsen contention for the scarcest runner pool, and any coverage it moves is shown to still exist elsewhere
- [ ] #5 The summary check fails when a job it gates on failed, and still publishes its report when that happens
- [ ] #6 The workflow-shape test is updated in the same change and each new assertion is proven to fail when what it pins is removed
- [ ] #7 Anything that cannot be fixed from a branch to the development line is stated as such rather than left implied
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read a killed job's own log to establish whether it was progressing or stalled at the kill.
2. Size the split from how far it got and the runner it got there on.
3. Split the job; keep the deterministic slicing contract the UI job already uses.
4. Give the summary check a verdict step, placed so the report is still published.
5. Update the shape test in lockstep, then mutate each new assertion to prove it bites.
<!-- SECTION:PLAN:END -->
