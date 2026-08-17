---
id: TASK-16315
title: Trajectory brushable timeline strip
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 00:21'
updated_date: '2026-08-15 06:55'
labels: []
dependencies:
  - TASK-16314
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up: brushable/zoomable duration timeline widget over TrajectorySnapshot (seq+timestamps), brush-to-filter synced with the ledger. Out of scope of the v1 plan; see spec follow-ups section in Docs/superpowers/specs/2026-08-14-console-trajectory-view-design.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Timeline renders start/duration per record,Brush filters ledger rows,Zoom and pan work,Selection syncs both ways
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TrajectoryTimeline widget: time-domain render model (lanes, bars by kind), zoom/pan math, brush->range math; unit tests. 2. Integrate into TrajectoryScreen: layout above ledger, brush filters ledger records (composable with search), click-select syncs ledger cursor both ways, clear-brush control; ADR-031 hints + governance tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented brushable timeline strip: TimelineModel + TrajectoryTimeline widget and TrajectoryScreen integration.

Approach: pure `TimelineModel` (no Textual symbols) backed by a thin `TrajectoryTimeline` widget — greedy 4-lane interval packing for bars, 1.25x focal-point zoom (focal fraction clamped once), pan as fraction-of-span shift, brush via mouse drag with `capture_mouse()` so gestures survive leaving the strip. Screen integration composes brush AND search through the existing generation-guarded render pipeline; `apply_brush` re-applies settled brushes across live refreshes (cleared when the new domain no longer intersects), and the axis/brush clocks format local time to match the ledger. Selection syncs both ways via a pull-only `set_selected` seam (bar→ledger moves/grows the cursor; ledger cursor moves highlight the bar).

Key files: `tldw_chatbook/UI/Widgets/trajectory_timeline.py`, `tldw_chatbook/UI/Screens/trajectory_screen.py`, `Tests/UI/test_trajectory_timeline.py`, `Tests/UI/test_trajectory_timeline_integration.py`.

Deviations/deferrals: widget zoom/pan keys ([ ] , .) unadvertised at screen level to keep footer hints 1:1 with actions (ADR-031); brush matching only unloaded records shows just the load-earlier row; mid-drag gesture still aborted by a live refresh (settled brush is preserved); one redundant re-render per tick while brushed. Governance: ADR-066 (console trajectory view); keybinding hints per ADR-031.

Tests: 64 passed across widget/screen/integration suites; ruff check + format clean. Commits: ebab888e9 (widget), ace2447ca + 8c79a8ee0 (integration + review fixes).
<!-- SECTION:NOTES:END -->
