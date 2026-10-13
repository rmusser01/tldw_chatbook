---
id: TASK-34939
title: Repaint-EnhancedStatusWidget-instead-of-recomposing-per-message
status: Done
assignee: []
created_date: 2026-10-13 14:58
labels:
- performance
- ui-perf
- wave-4-blocking-rebuilds
dependencies: []
priority: low
updated_date: 2026-10-13 19:08
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
EnhancedStatusWidget.messages is the last data reactive in the repo with recompose=True (status_widget.py:82); add_message assigns a fresh list per message, and the chatbook importer calls add_warning/add_error once per conversation/file in import loops (chatbook_importer.py:1115-1768) — a 500-item import recomposes the widget 500 times. The widget renders via render() and has no children, so a plain repaint would suffice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] messages reactive drops recompose=True (refresh-only) or children update incrementally
- [x] Per-message cost is a repaint not a remove/remount cycle (test with N messages)
<!-- AC:END -->


## Implementation Notes

- `messages` reactive dropped `recompose=True`: the widget renders via `render()` (a Panel over `_messages`) and composes no children, so per-message recomposes (a full remove/remount cycle) were pure scheduling overhead; the reactive's default refresh repaint is the correct primitive. Empirically verified on Textual 8.2.8 that a refresh-only reactive still resizes a `height: auto` widget, so nothing relied on the recompose for sizing.
- Tests: `Tests/UI/test_status_widget_refresh_only.py` (declaration is refresh-only; 50 add_message calls never recompose, buffer trims correctly). 2/2 pass. Note: the behavioral test runs on a bare shell with the reactives shadowed, so the repaint itself is pinned by the declaration test; a mounted run_test harness would be a nice follow-up.
- ADR required: no — one reactive flag.