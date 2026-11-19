---
id: TASK-42014
title: "Containers other than Library canvases drop a queued button press when they recompose"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:54
labels:
- follow-up
- ui
priority: low
references:
- tldw_chatbook/Widgets/Library/library_canvas_sync.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41703 (round 17). Textual removes a recompose's old children top down, so a queued `Button.Pressed` below a removed child is dropped. TASK-41703 forwards such presses only for `PostRecomposeCallback` canvases; any other container that recomposes while a child holds a queued press still drops it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes list which other recomposing containers can hold a queued press, and each one keeps it or is shown unable to
- [ ] #2 Each container that can hold one has a pilot test that holds a press across its recompose, red before the fix
<!-- AC:END -->
