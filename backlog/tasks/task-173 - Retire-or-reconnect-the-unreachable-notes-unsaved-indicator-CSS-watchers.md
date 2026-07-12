---
id: TASK-173
title: Retire or reconnect the unreachable notes unsaved-indicator CSS/watchers
status: To Do
assignee: []
created_date: '2026-07-11 23:53'
labels:
  - follow-up
  - notes
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The .unsaved-indicator/.has-unsaved/.auto-saving/.saved CSS family plus app.py's watch_notes_unsaved_changes/watch_notes_auto_save_status watchers are practically unreachable (the target widget is never composed since the standalone Notes screen was retired) but still grepped-live in app.py, so the F2/quick-wins CSS sweeps kept them. Adjudicate: either remove the dead watchers + CSS together, or reconnect them to the Library notes editor's real save-state indicator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The unsaved-indicator CSS + watchers are either removed together or reconnected to a live widget,No dead grep-live references remain
<!-- AC:END -->
