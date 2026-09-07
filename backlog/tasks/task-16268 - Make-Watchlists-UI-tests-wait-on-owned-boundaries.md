---
id: TASK-16268
title: Make Watchlists UI tests wait on owned boundaries
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 20:29'
updated_date: '2026-08-14 20:40'
labels:
  - testing
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Watchlists layout and local feed-server tests deterministic under the repository network guard and asynchronous section layout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Section-width comparison waits for the destination pane's observable layout.
- [x] #2 The real feed-server teardown test explicitly opts into loopback network access.
- [x] #3 Non-network Watchlists tests remain subject to the default network denial.
- [x] #4 Focused tests, the original 25-file checkpoint, and static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve and attribute the suite-only layout failure and sandbox-denied local-server failures.
2. Replace the fixed layout pause with bounded observable convergence and mark only the genuine loopback teardown test.
3. Run focused repetitions, the original checkpoint with its loopback permission, and static checks.

ADR required: no
ADR path: N/A
Reason: test-only synchronization and existing network-guard metadata; no product or security-boundary change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced a fixed post-navigation delay with a bounded wait for the destination pane to exist and receive non-zero layout before comparing widths.
- Marked only the real feed-server teardown test for loopback access; all other Watchlists tests retain the default network-denial guard.
- Preserved strict RED evidence: the original checkpoint failed four tests, the first immediate-query revision failed both viewport cases, and omitting the network marker reproduced the teardown guard failure.
- Verified the five affected cases (5 passed) and the original 25-file checkpoint (563 passed). `git diff --check` passes. Ruff reports the same 11 pre-existing diagnostics on both `HEAD` and the changed file, and Ruff format is likewise already red on `HEAD`; no unrelated cleanup was included.
- ADR required: no. This is test-only synchronization and metadata for an existing loopback boundary.
<!-- SECTION:NOTES:END -->
