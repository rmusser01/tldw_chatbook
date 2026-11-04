---
id: TASK-38716
title: "PR Fast Lane and the Console P0 gate run at their time caps"
status: To Do
assignee: []
created_date: 2026-11-04 00:59
labels:
- follow-up
- ci
- needs-owner-decision
priority: high
references:
- .github/workflows/derived-artifacts.yml
- backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 6 of the 2026-10-23 follow-up burn-down. Checked on dev c2cdb6b82e, 2026-11-03. The figures below are job wall times (`started_at` to `completed_at`), read with `gh api` from the 200 most recent `derived-artifacts.yml` pull-request runs (jobs started 2026-11-01 09:13Z to 2026-11-04 00:43Z).

**PR Fast Lane** (`pr-fast-lane`, `timeout-minutes: 35`, derived-artifacts.yml:255):
- Jobs started on 2026-11-03: 44 successful runs, median 31.5 min, p95 34.5, max 34.7. Including the 25 failed runs: median 31.8, p95 34.3. Two runs were cancelled at 35.0 and 35.5 min, which is the cap.
- The whole window: 52 successful runs, median 31.4, p95 34.5, max 34.7.

**Console P0 regression gate** (`console-p0-gate`, `timeout-minutes: 20`, :940):
- Jobs started on 2026-11-03: 56 successful runs, median 18.2 min, p95 19.6, max 20.0. Nine more were cancelled at 20.2-20.9 min, consistent with the cap cancelling them.

**Stale timing comments.**
- The `pr-fast-lane` comment (:246-254) still records TASK-34916's replay: "median 16.1 / max 20.8 min, 14 min under the cap". The inline note on its timeout cites a 27m46s worst case.
- The `console-p0-gate` comment (:924-936) estimates about 9 min and records TASK-34986's "median 11.6 / max 13.9".

Both jobs are back where TASK-34916 and TASK-34986 started. Those tasks fixed it by moving steps into a new job, and the comment's own rule is "add a job (a shard), not minutes". Every new job costs one more concurrent runner per PR (ADR-103). Runner capacity is the owner's call, so the remedy needs the owner's decision first: a shard, a split, or a cap change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision on the remedy (shard, split or cap change) is recorded in the task notes
- [ ] #2 Over at least 20 pull-request runs after the change, the PR Fast Lane's p95 job time is at least 5 minutes under its cap and the Console P0 gate's is at least 4 minutes under its cap; the window and the figures are recorded in the task notes
- [ ] #3 No test leaves the required check to achieve it
- [ ] #4 Both jobs' timing comments in the workflow state the measured median and maximum and the window they came from, and the contract tests that pin the lanes reflect the new layout
<!-- AC:END -->
