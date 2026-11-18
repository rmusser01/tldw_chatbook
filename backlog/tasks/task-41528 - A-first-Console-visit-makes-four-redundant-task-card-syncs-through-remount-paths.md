---
id: TASK-41528
title: "A first Console visit makes four redundant task-card syncs through remount paths"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- console
- performance
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the console stream (round 15). A first visit to Console runs the task-card sync four times through remount paths; each sync is cheap and the stream left them alone. Measured cost is small; worth one look if first-visit time is revisited.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The number of task-card syncs on a first Console visit is measured and recorded
- [ ] #2 Redundant syncs are removed, or the notes record that the saving is under 5 ms
<!-- AC:END -->
