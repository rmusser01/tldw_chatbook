---
id: TASK-37000
title: >-
  Console: a first send rebuilds the collapsed Inspector's run section and lays
  out hidden widgets
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-26 12:30'
updated_date: '2026-10-27 15:53'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure first (headless, 160x45, collapsed Inspector): count layout/recompose requests and full reflows from widgets inside #console-right-rail during a first send and a later send on origin/dev; A/B the rail opened after a collapsed send against one open throughout.
2. Tests first: one test counting in-rail layout requests across a first and a later send (positive control with the rail open), one A/B witness (rows, actions, classes, geometry) of an Inspector opened after the send vs one open throughout. Both must fail on origin/dev for the right reason.
3. Fix: while the rail's collapse cascade hides them, ConsoleRunInspector and ConsoleInspectorSection keep their state current, re-write row copy without a layout, and hold any rebuild for the rail's opening; the cascade's opening runs the held rebuild. Display writes into the hidden rail (the Agents section) record into the cascade's marker instead of bypassing it.
4. Mutation-check the fix against the tests; re-measure headless; live paired spot-check with the pf2probe kit (base vs branch).
5. Related suites by test id vs origin/dev, ratchets, preflight, rebase onto newest origin/dev.
<!-- SECTION:PLAN:END -->
