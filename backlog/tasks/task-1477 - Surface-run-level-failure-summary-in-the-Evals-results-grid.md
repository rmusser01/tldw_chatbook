---
id: TASK-1477
title: >-
  Surface run-level failure summary in the Evals results grid
status: Done
assignee: []
created_date: '2026-07-30 10:00'
labels:
  - evals
  - word-bench
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by live UAT (2026-07-30). On a fresh install the config template pre-fills `api_settings.llama_cpp.api_url` (config.py:2482), so the sample-bench gate is always true and the invited one-click golden path runs against a dead server — producing a grid of em-dashes headed "4 failed" with no visible explanation. The reason ("Failed: unreachable — All connection attempts failed") appears only if the user discovers cell-focus. There is no run-level banner, and no next step at the moment of failure.

The grid already loads every cell's failure reason; deriving a run-level summary needs no engine change. Related: task-703 (preflight verdicts through the API) — note `WordBenchRunner.run()` now returns the per-target `PreflightResult` map, so part of 703's first AC already exists at the engine seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] A run group whose cells all failed renders an always-visible callout naming the dominant failure reason and a concrete next step (start the server, then run the bench again)
- [x] A partially failed run states how many cells failed and the dominant reason
- [x] The callout follows the readiness vocabulary (`.ds-recovery-callout`, no hover-only content, no color-only signal)
- [x] Tests cover the all-failed and mixed cases
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Compute a run-level failure summary in the grid's existing single cell pass
2. Render #evals-grid-failure-callout between the state line and the canary callout
3. Pin exact strings for all-failed and partial cases; tie-break test for the dominant reason
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Commits d9ec57249, 5e3c9984c. `_failure_summary` (computed once in compose(), shared with the header meta line so counts can never disagree) derives failed count and the dominant reason (most frequent, ties first-seen); all-failed renders "All N cells failed — <reason>. Check that the target's server is running and reachable, then run the bench again.", partial renders the count-of-total form without the next-step sentence, zero failures renders nothing (DOM-absence tested). Reasons come from CellError's fixed vocabulary (server text never reaches the callout) and the Static is markup=False regardless. A review fix round added the dedicated 2-vs-2 tie + majority test after the reviewer showed a Counter.most_common() refactor would silently flip the displayed reason. Verified live: the callout rendered verbatim over a dead-server sample run. Known limitation filed as task-1511 (in-flight/cancelled runs can overstate failure).
<!-- SECTION:NOTES:END -->
