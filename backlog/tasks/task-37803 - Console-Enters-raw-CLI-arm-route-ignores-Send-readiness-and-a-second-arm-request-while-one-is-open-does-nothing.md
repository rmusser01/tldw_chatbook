---
id: TASK-37803
title: 'Console: Enter''s raw-CLI arm route ignores Send readiness, and a second arm
  request while one is open does nothing'
status: Done
assignee: []
created_date: 2026-10-29 17:14
labels:
- console
- raw-cli
dependencies: []
priority: low
updated_date: 2026-11-01 14:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33622.9 routes an unarmed '! ' draft's Enter, Send and Workbench send to an in-place arm confirmation in request_visible_send. Two small gaps remain. The arm route runs before Enter's send_action_ready guard, so Enter opens the arm dialog even when the Send action is hidden or pruned (for example while another surface owns the composer). And ConsoleRawCliController.request_arm returns False with no feedback when an arm confirmation is already in flight, so a repeat press looks ignored.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Enter on an unarmed raw draft opens the arm confirmation only when the Send action is ready, the same guard a normal Enter send uses
- [x] #2 A request to arm while an arm confirmation is already open gives visible feedback (or focuses the open confirmation) instead of doing nothing
- [x] #3 Both are pinned by tests on the real ChatScreen
- [x] #4 Unavailable visible session and a failed arm-worker start refuse visibly without modifying the draft or arming the runtime.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
2026-11-01 PR3319 integration: follow Docs/superpowers/plans/2026-11-01-pr3319-backlog-integration.md. Reproduce delayed/banked raw restoration and oversized paste/file rendering before repair; preserve original received-input/acknowledgement/native ownership. ADR required: no new ADR; existing ADR093/031/230 apply. Targeted tests and independent requesting-code-review required; hosted CI and size limits waived.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR3319 integration restores visible-session and Send-readiness checks before Arm; repeated, unavailable and failed-start requests notify without changing the draft or runtime authority. Existing ADR093/031/230 apply. Real ChatScreen regressions and preservation tests pass. Exact RED/GREEN receipts, scoped lint and review evidence: Docs/superpowers/qa/2026-11-01-pr3319-backlog-integration.md. Hosted CI and size limits waived by user; targeted tests only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
