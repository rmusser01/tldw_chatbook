---
id: TASK-37000
title: >-
  Console: a first send rebuilds the collapsed Inspector's run section and lays
  out hidden widgets
status: To Do
assignee: []
created_date: '2026-10-26 12:30'
labels:
  - console
  - performance
dependencies:
  - TASK-36012
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while classifying the send-transition requesters for TASK-36012 (headless, 160x45, collapsed Inspector, 2026-10-26). A chat's first send persists the conversation, which turns the Inspector's dictionary and world-book rows from "No active chat" into "No ... in play" and enables their Attach actions. `ConsoleRunInspector.sync_state` treats that as a structural change and recomposes, re-mounting the run section, the More control and every bounded section and viewport. The Inspector rail is collapsed (`display: none`), so none of it is visible, yet each mount and the rail cascade that then hides the new children ask Textual for a whole-screen layout: on TASK-36012's branch the first send's start still made about 190 layout requests over 14 full reflows, against 92 over 9 for a later send in the same chat.

The same is true, more mildly, of every send: the run inspector's live-work row and the selected-turn rows update their auto-height copy inside the hidden rail, and each update lays out the whole screen. Those widgets' heights can genuinely change, so the request is not wrong in itself; it is wrong only while nothing of the rail is drawn.

Fix direction to evaluate (not prescribed): let the collapsed Inspector defer its rebuilds and auto-height copy until it opens (opening already resizes the rail and reconciles it), keeping its state current for readers and tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Measured first, paired against the base build: layout requests and full reflows from collapsed-Inspector widgets during a first send and a later send
- [ ] #2 While the Inspector is collapsed, a send causes no layout request from widgets inside it, and opening it afterwards shows the same rows, actions and geometry as a rail that was open throughout
- [ ] #3 A deterministic test pins both halves and fails on the base build
<!-- AC:END -->
