---
id: TASK-274
title: 'CI guard: fail on duplicate backlog task IDs'
status: Done
assignee: []
created_date: '2026-07-17 00:31'
updated_date: '2026-07-17 00:31'
labels:
  - tooling
  - backlog
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backlog task IDs collided three times in one week (152+, 196-203, 246-256): two branches each mint the next free ID concurrently and both merge, making backlog CLI lookups ambiguous and cross-references unreliable. Add an automated guard so a PR introducing (or a merge producing) duplicate task IDs fails visibly. Root-cause fix decided with the owner on 2026-07-17.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CI fails when backlog/tasks contains two task files sharing the same task ID
- [x] #2 Guard runs on PRs touching backlog/tasks and post-merge on dev/main so merge-time races are caught immediately
- [x] #3 Failure output names the colliding IDs and files and states the renumber procedure
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Renumber colliding perf batch 246-256 to 263-273 (unblocks ambiguity)
2. Add .github/workflows/backlog-guard.yml duplicate-ID check on PR + push
3. Verify detection with a planted duplicate
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented as .github/workflows/backlog-guard.yml: a single-job workflow that extracts IDs from backlog/tasks filenames (dotted subtask IDs handled) and fails with the colliding IDs, files, and the renumber procedure when uniq -d finds duplicates. Runs on pull_request (paths backlog/tasks/**) and on push to dev/main so a merge-time race between two individually-clean PRs is flagged immediately after merge rather than lingering. Verified locally: clean tree passes; a planted duplicate task-246 file is detected. Shipped in the same PR as the third-collision cleanup (perf batch renumbered to 263-273) and ADR-005. Residual risk: two colliding PRs merging within one CI cycle are caught by the push run, not blocked pre-merge; acceptable given the repo cancels in-flight CI intentionally.
<!-- SECTION:NOTES:END -->
