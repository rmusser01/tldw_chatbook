---
id: TASK-28023
title: >-
  Trajectory ledger - render repeated causal turn segments without duplicate row
  keys
status: Done
assignee: []
created_date: '2026-09-02 04:16'
updated_date: '2026-09-02 04:27'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent the Trajectory screen from crashing when the causal projection preserves a logical turn as multiple non-contiguous segments.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A trajectory snapshot containing repeated non-contiguous segments for the same logical turn mounts without DuplicateKey and renders every segment and record.
- [x] #2 Turn-level collapse and inspection continue to resolve segment headers to the original logical turn identifier.
- [x] #3 Focused Trajectory screen tests cover the repeated-segment regression and pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a mounted TrajectoryScreen regression test with logical turns t1, t2, t1; assert every record renders, header row keys are unique, and each segment maps back to its logical turn.
2. Run the focused test and confirm it fails with Textual DuplicateKey before changing production code.
3. Generate a stable segment-specific header key during row-spec construction while preserving explicit header-key-to-logical-turn mapping for collapse and inspection.
4. Run the focused regression and the Trajectory screen, timeline-integration, and live test modules, then run scoped static checks.

ADR required: no
ADR path: N/A
Reason: This is a localized bug fix that preserves the existing projection, UI ownership, and cross-module interfaces.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented segment-aware Trajectory ledger row identity without altering causal projection output. The first occurrence retains its existing turn:<id> key for live cursor stability; later occurrences use turn-segment:<occurrence>:<id> and map back to the logical turn for inspection and collapse. Logical turn numbering now follows first occurrence, inspector counts aggregate records across repeated segments, and paused live refresh expands the pagination window when needed to restore a selected segment header.

Added mounted Textual regression coverage for repeated t1 -> t2 -> t1 rendering, unique headers, complete record rendering, logical-turn actions, live first-header stability, and page-boundary restoration with a colon-bearing turn ID.

Modified files: tldw_chatbook/UI/Screens/trajectory_screen.py; Tests/UI/test_trajectory_screen.py. No user documentation or ADR update was required because this preserves existing behavior and boundaries.

Verification: 67 targeted Trajectory screen, timeline-integration, and live tests passed; Ruff format/check and git diff --check passed.
<!-- SECTION:NOTES:END -->
