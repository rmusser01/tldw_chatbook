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
