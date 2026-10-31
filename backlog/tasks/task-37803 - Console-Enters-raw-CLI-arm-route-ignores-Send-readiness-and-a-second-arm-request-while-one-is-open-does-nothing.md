---
id: TASK-37803
title: >-
  Console: Enter's raw-CLI arm route ignores Send readiness, and a second arm
  request while one is open does nothing
status: To Do
assignee: []
created_date: '2026-10-29 17:14'
labels:
  - console
  - raw-cli
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33622.9 routes an unarmed '! ' draft's Enter, Send and Workbench send to an in-place arm confirmation in request_visible_send. Two small gaps remain. The arm route runs before Enter's send_action_ready guard, so Enter opens the arm dialog even when the Send action is hidden or pruned (for example while another surface owns the composer). And ConsoleRawCliController.request_arm returns False with no feedback when an arm confirmation is already in flight, so a repeat press looks ignored.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Enter on an unarmed raw draft opens the arm confirmation only when the Send action is ready, the same guard a normal Enter send uses
- [ ] #2 A request to arm while an arm confirmation is already open gives visible feedback (or focuses the open confirmation) instead of doing nothing
- [ ] #3 Both are pinned by tests on the real ChatScreen
<!-- AC:END -->
