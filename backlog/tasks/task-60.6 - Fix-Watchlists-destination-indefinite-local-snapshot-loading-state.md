---
id: TASK-60.6
title: Fix Watchlists destination indefinite local snapshot loading state
status: Done
assignee: []
created_date: 2026-05-20 15:16
labels:
- ux
- hci
- qa
- screens
dependencies: []
parent_task_id: TASK-60
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure the Watchlists destination resolves its local snapshot load into an actionable ready, empty, or error state so users can understand whether watchlist runs are available.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Watchlists screen leaves loading state deterministically in an empty local profile.
- [x] #2 Watchlists screen shows an actionable empty or service error state instead of permanent loading copy.
- [x] #3 Open and Console controls reflect actual run availability.
- [x] #4 Mounted regression waits deterministically without fixed sleeps.
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Re-check current merged `dev` before editing because Watchlists screen QA was already completed in a later PR.
2. Inspect Watchlists destination tests and screen loading/error/empty-state code paths.
3. Run the focused Watchlists destination regression slice to verify empty, error, timeout, control-state, and deterministic wait coverage.
4. If the bug still reproduces, add a failing regression and implement the smallest fix; otherwise close the stale follow-up with verification evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Verified the issue is already resolved on current `origin/dev` after the Watchlists screen QA work.
- Current Watchlists destination coverage includes empty local snapshots, service failure recovery copy, timeout-to-recovery behavior, distinct initial loading copy, policy-denied recovery selectors, disabled Console attach state, and deterministic polling via `_wait_for_wc_snapshot`.
- No production code change was needed; the task is closed as verified by the focused Watchlists destination regression slice.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py -k 'watchlists_collections' --tb=short` returned `13 passed, 89 deselected`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed as already fixed on current `dev`; focused mounted Watchlists destination regressions pass and cover the original indefinite loading failure mode.
<!-- SECTION:FINAL_SUMMARY:END -->
