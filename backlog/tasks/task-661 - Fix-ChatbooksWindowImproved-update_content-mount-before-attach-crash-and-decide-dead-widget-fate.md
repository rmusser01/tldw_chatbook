---
id: TASK-661
title: Fix ChatbooksWindowImproved update_content mount before attach crash and decide dead widget fate
status: To Do
assignee: []
created_date: '2026-07-26 12:00'
labels:
  - followup
  - ui
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during task-637: ChatbooksWindowImproved._update_content() constructs a local Grid/ListView and mounts cards/items into it BEFORE the container itself is attached to the DOM - Widget.mount() raises MountError when not is_attached, so any non-empty chatbooks list crashes the recompose path (task-637's test deliberately routes around it via the empty-state branch). Separately, decide the fate of two effectively-dead widgets task-637 also had to guard: ResultsDashboardWindow.py cannot even import (missing eval_shared_components module) and Mindmap_Viewer_Window.py has no live call site - delete or properly wire them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Non-empty chatbooks list renders without MountError (regression test with 2+ chatbooks)
- [ ] #2 ResultsDashboardWindow and Mindmap_Viewer_Window are each either deleted or wired to a live call site (decision documented)
- [ ] #3 Existing chatbooks tests stay green
<!-- AC:END -->
