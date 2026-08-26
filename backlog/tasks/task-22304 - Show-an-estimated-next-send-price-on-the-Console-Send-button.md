---
id: TASK-22304
title: Show an estimated next-send price on the Console Send button
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-26 04:42'
updated_date: '2026-08-26 04:43'
labels: []
dependencies:
  - TASK-22303
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users an at-a-glance indication that a composed Console message has an estimated processing cost, with a hover tooltip that explains the projected input and maximum reply costs before they send.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An empty Console draft keeps the normal Send or Queue label and its existing non-price guidance.
- [ ] #2 A non-empty sendable draft shows `Send | $` or `Queue | $` without exposing a misleading numeric amount in the compact button label.
- [ ] #3 Hovering the priced action shows an estimated next-request total plus separate input and maximum-reply token and cost details.
- [ ] #4 The estimate includes current conversation context, system and staged context, the live draft, and the configured maximum reply tokens.
- [ ] #5 Unknown or unconfigured pricing keeps the dollar affordance and explains that a cost estimate is unavailable.
- [ ] #6 The estimate refreshes when the draft, active session/provider/model, staged context, pending attachments, or relevant settings change, while accumulated-spend behavior remains unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture the approved interaction, cost-estimation boundaries, refresh triggers, unavailable-pricing behavior, and test strategy in a written design spec.
2. Obtain independent spec review, address any planning-blocking issues, and ask the user to review the committed spec.
3. After written-spec approval, produce a TDD implementation plan before changing production code.
4. Implement and verify the pure estimate model, Console integration, dynamic button presentation, tooltip copy, and focused regressions.

ADR required: no
ADR path: N/A
Reason: The feature derives ephemeral presentation state from existing session settings, pricing catalog data, and composer/context inputs; it adds no storage, schema, ownership, service-contract, security, or long-lived application-structure decision.
<!-- SECTION:PLAN:END -->
