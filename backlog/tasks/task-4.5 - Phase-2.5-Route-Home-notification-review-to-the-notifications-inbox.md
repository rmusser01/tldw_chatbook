---
id: TASK-4.5
title: 'Phase 2.5: Route Home notification review to the notifications inbox'
status: Done
assignee: []
created_date: '2026-05-03 18:47'
updated_date: '2026-05-03 18:50'
labels:
  - unified-shell
  - phase-2
  - home
  - notifications
  - navigation
dependencies: []
parent_task_id: TASK-4
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Home notification next-best actions lead to the existing local notifications inbox instead of looping on Home or requiring users to remember where notifications live.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Home notification next-best action targets a real notifications review workflow.
- [x] #2 Clicking the Home notification primary action opens the subscriptions notifications tab context.
- [x] #3 The subscriptions surface can honor a pending notifications initial-tab request and then clear it.
- [x] #4 Notification review routing does not create active-work controls or change approval priority.
- [x] #5 Focused tests and Phase 2 QA evidence verify the end-to-end routing seam.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing dashboard and Home screen tests for notification review routing.
2. Add failing subscriptions test for honoring a pending notifications tab request.
3. Implement the smallest Home action and SubscriptionWindow initial-tab seam.
4. Add Phase 2 QA evidence and tracking updates.
5. Run focused Home, subscription, and tracking verification plus git diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed Home notification review from a self-looping Home route to the existing subscriptions notifications inbox path. Home primary-action clicks now stage a one-shot `pending_subscription_initial_tab = "notifications"` request before navigation, and `SubscriptionWindow` consumes that validated initial tab and clears it. Added regression coverage for dashboard routing, mounted Home primary-action behavior, SubscriptionWindow tab consumption, and Phase 2 tracking evidence.
<!-- SECTION:NOTES:END -->
