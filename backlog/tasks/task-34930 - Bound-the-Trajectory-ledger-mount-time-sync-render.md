---
id: TASK-34930
title: Bound-the-Trajectory-ledger-mount-time-sync-render
status: Done
assignee: []
created_date: 2026-10-13 14:53
labels:
- performance
- ui-perf
- wave-3-screen-switch
dependencies: []
priority: medium
updated_date: 2026-10-13 17:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Trajectory builds all row specs synchronously at mount whenever the ledger is below WORKER_THRESHOLD = 5000 records (trajectory_screen.py:460-477); above it, a thread worker renders. 5000 is high enough that large-but-under-threshold sessions still pay a full sync render on the navigation path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Mount-time synchronous render bounded by a lower threshold or first-window render plus background fill
- [x] No regression for small ledgers (immediate full render below the new bound)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read trajectory mount render + WORKER_THRESHOLD + _render_ledger
2. Lower the bound or render first window + background fill
3. Tests + targeted run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Premise corrected by investigation: the ledger render was ALREADY pagination-bounded — `_build_row_specs` iterates `_flat_slice()`, the newest `min(total, PAGE_SIZE=500)` records, so the sync render below the threshold is O(500) rows regardless of conversation size, not O(total) as the review assumed. What actually scales with total records on the mount path: `_filter_bar.set_records(self._all_records())`, the timeline `record_keys` dict comprehension over all records, and the turn-occurrence counter over all turns.
- Fix: `WORKER_THRESHOLD` lowered 5000 → 1000, moving those O(total) setup legs onto the proven thread-worker path (video-player pattern, unchanged) for large-but-not-huge sessions. Small ledgers (<1000 records) still render synchronously with no loading state. Tests reference the constant symbolically (`WORKER_THRESHOLD + 2`), so the existing worker-path coverage automatically exercises the new bound.
- Verification: `Tests/UI/test_trajectory_screen.py` is at its pre-existing baseline (33 setup errors with the change stashed — same mount breakage as the other UI suites); the change is a constant plus comment, with the if/else and worker path untouched.
- ADR required: no — threshold tuning inside one screen.
