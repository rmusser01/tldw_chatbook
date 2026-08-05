---
id: TASK-2042
title: >-
  Library ingest first click swallowed by canvas recompose (scope recomposes to dynamic regions)
status: Done
assignee: []
created_date: '2026-08-03 02:00'
labels:
  - library
  - ingest
  - ux
  - architecture
priority: high
dependencies: []
---

## Description (the why)

Round-2 critique P0: clicking "Start ingest" (or Expand all / Browse…)
right after typing does nothing — the second click fires. The debounced
pre-flight apply and every job tick recompose the whole canvas, remounting
every widget; a mouse-down delivered to the old Button instance can never
pair with its mouse-up on the new one. The while-typing pre-flight
(task-2015) made these recomposes land exactly in the type-then-click
window (~0.8–1.2s after the last keystroke). The same whole-canvas
recompose also yanks the viewport off the queue on completion repaint
(round-2 anomaly A6) — the scroll/focus capture-restore of task-2010 was a
mitigation for a problem this removes at the source.

The fix the architecture has been converging on (task-2010 notes, Qodo
review alternative #1): extract the two DYNAMIC regions — the pre-flight
summary lines and the queue — into render-from-state child widgets, and
have the pre-flight apply and registry ticks recompose ONLY those children.
The path input, action buttons, and metadata fields then never remount on
the hot paths: no swallowed clicks, no scroll jump, focus preserved for
free. Whole-canvas recompose remains for structural changes (type-group
set changes, canvas switches).

## Acceptance Criteria (the what)

- [x] A pre-flight apply for an unchanged type-group set does not remount
      the Start/Browse buttons or the path/metadata inputs (widget
      identity preserved, pinned by test).
- [x] A registry job tick does not remount the form widgets; queue rows
      still update live.
- [x] Canvas scroll position genuinely holds across job transitions and
      batch completion (no viewport jump).
- [x] A type-group set change (e.g. a PDF newly staged) still rebuilds the
      options panels.
- [x] Existing ingest UI suites stay green.

## Implementation Notes

The two dynamic regions are now render-from-state child widgets
(`LibraryIngestPreflightSummary`, `LibraryIngestQueuePanel` in
`library_ingest_canvas.py`); `_update_library_ingest_dynamic_regions`
recomposes ONLY them for pre-flight triggers/results, registry job
ticks, path-clear, and clear-finished arming, with the context-preserving
full recompose kept for structural changes (type-group set, runtime
header lines) and as fallback. To keep the hot paths non-structural: the
Clear button and intro Statics are always mounted and display-managed,
and options panels stay mounted during a re-analysis (three canvas tests
updated to pin the display-based contract). Also extended the
prompts-editor rail-only `sync_state` protection in
`_apply_local_source_snapshot` to the ingest canvas -- that snapshot
lands after every completed job and was the remaining full-recompose
source (found by the widget-identity test, not by reading).
Verified: three new identity tests (apply preserves Start/path identity;
job tick preserves identity + updates queue; group-set change still
rebuilds panels); 228/228 canvas+state+runner, 83/83 shell ingest
subset, full-tree collect-only clean (29,068). Live: ONE click on Start
fires after typing (was two), focused labels readable, dedup toast
honest, zero first-submit noise.
