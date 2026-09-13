---
id: TASK-24199
title: Stop redundant timer-path Static relayouts
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29 12:50'
updated_date: '2026-08-29 12:59'
labels:
  - performance
  - ui
  - tests
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent recurring Scheduling and Folder Files Notes refresh clocks from arming whole-screen relayouts when their visible Static content has not changed, while preserving layout when variable-size content does change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repeated timer-path renders with unchanged Static content do not call Static.update
- [x] #2 Variable-size content changes still use normal layout-aware Static.update behavior
- [x] #3 The timer-path Static.update architecture census passes with every remaining reachable update explicitly classified
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: localized timer-path performance correction within existing UI ownership and layout contracts. Use the failing architecture census as the red gate; add a tiny per-class content-equality helper that skips Static.update for unchanged content but retains the default layout-aware update for changed content. Route the timer-reachable Scheduling notices and Folder Files status/path projections through it, keep the fixed-height path label on layout=False, classify the equality-gated helper sites and external-change-only path, add focused helper tests, then run exact, containing, targeted UI, static, and diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the timer-path relayout issue rather than merely weakening the census. Scheduling and Folder Files Notes now compare projected copy with Static.content before calling Static.update: unchanged clock-driven renders are true no-ops, while changed auto-height content still uses Textual's default layout-aware update. Routed Scheduling empty/legend notices and Folder Files exact-path, save/detail, authority, and path-label projections through the gate. Pinned the newly discovered Scheduling repeating-clock root and replaced stale per-call classifications with two explicit equality-gated helper classifications. Self-review changed the planned fixed-height path-label layout=False optimization to the equality gate, preserving normal layout on changes without needing a geometry assumption. Added real-Static tests for both owners. ADR required: no; ADR path: N/A. Verification: helper plus full timer census 18 passed in 17.79s; full Scheduling workbench file 40 passed in 13.05s; focused Folder Files subset 7 passed (150 deselected) in 2.47s; Ruff check passed; compileall passed; git diff --check passed. Ruff format baseline ratchet is unchanged: the same two files are format-red at HEAD and current, while the other three touched files are formatted.
<!-- SECTION:NOTES:END -->
