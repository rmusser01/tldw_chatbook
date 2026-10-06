---
id: TASK-34413
title: Merge queue retry strands a PR whose required check failed once
status: Done
assignee:
  - '@claude'
created_date: '2026-10-06 07:36'
updated_date: '2026-10-06 07:36'
labels:
  - ci
  - merge-queue
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When the required check fails once on the front PR, the merge queue retries by dispatching a fresh run. That run opens a second check suite, and the failed check stays the latest in its own suite, so branch protection keeps counting the failure: the PR stays BLOCKED on a green head until the queue evicts it as stuck. #2874 was evicted this way on 2026-10-06 (03:17Z) and #3026 was stranded the same way. Re-running the failed run's jobs adds the new attempt to the same suite and clears it, verified on #3019 on 2026-10-05.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A first failure of the required check is retried by re-running that run's failed jobs, so a passing retry leaves the PR mergeable
- [x] #2 A re-run that fails again still counts as the second failure and evicts the PR, never re-running forever
- [x] #3 When no required-workflow run is behind the failed check, or GitHub refuses the re-run, the queue falls back to a fresh dispatch
- [x] #4 The spec states the corrected retry design and the evidence
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
scripts/merge_queue.py: a retry now re-runs the failed run's failed jobs instead of dispatching a fresh run. decide_front puts the failed check's suite_id on the retry Action; _rerun_failed finds the required-workflow run with that check_suite_id and POSTs actions/runs/{id}/rerun-failed-jobs (the queue job already has actions: write). A re-run adds the new attempt to the same check suite, so protection (latest per suite) stops counting the failure. read_checks (filter=all) still lists it, so a second failure still evicts, verified live on #3019 head 6a11342309. Fallback to the old fresh dispatch: no required-workflow run behind the suite (e.g. a broken run's stand-in, which has no suite), or GitHub refuses the re-run. The comment says 're-running its failed jobs' or 'retrying with a fresh run' accordingly.

Evidence for the bug: #3026 head d35d1fb595 had a failed and a passing required check in two suites, both returned by filter=latest, and stayed BLOCKED while green. #2874 was evicted as stuck at 03:17Z the same way.

Tests (Tests/CI/test_merge_queue_actions.py): the re-run path, three fallbacks (no run / other workflow / refused), and a failed re-run still evicting. FakeGh learned rerun-failed-jobs. Tests/CI 463 passed; removing the re-run fails its test. The spec (Docs/superpowers/specs/2026-10-03-merge-queue-design.md) is corrected: its line calling the re-run API undocumented was my 10-03 design error, not an owner decision. V3 records the evidence. The queue's GITHUB_TOKEN re-run path is first exercised live on the next retry (the evidence used the owner's token).
<!-- SECTION:NOTES:END -->
