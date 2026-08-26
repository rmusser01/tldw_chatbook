---
id: TASK-22304
title: Show an estimated next-send price on the Console Send button
status: Done
assignee:
  - '@codex'
created_date: '2026-08-26 04:42'
updated_date: '2026-08-26 14:41'
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
- [x] #1 An empty Console draft keeps the normal Send or Queue label and its existing non-price guidance.
- [x] #2 A non-empty sendable draft shows `Send | $` or `Queue | $` without exposing a misleading numeric amount in the compact button label.
- [x] #3 Hovering the priced action shows an estimated next-request total plus separate input and maximum-reply token and cost details.
- [x] #4 The estimate includes current conversation context, system and staged context, the live draft, and the configured maximum reply tokens.
- [x] #5 Unknown or unconfigured pricing keeps the dollar affordance and explains that a cost estimate is unavailable.
- [x] #6 The estimate refreshes when the draft, active session/provider/model, staged context, pending attachments, or relevant settings change, while accumulated-spend behavior remains unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture the approved interaction, cost-estimation boundaries, refresh triggers, unavailable-pricing behavior, and test strategy in a written design spec.
2. Obtain independent spec review, address any planning-blocking issues, and ask the user to review the committed spec.
3. After written-spec approval, produce a TDD implementation plan before changing production code.
4. Implement and verify the pure estimate model, Console integration, dynamic button presentation, tooltip copy, and focused regressions.

ADR required: yes
ADR path: backlog/decisions/088-console-lightweight-next-send-history-projection.md
Reason: The feature adds a cross-module detached store snapshot and shared pre-serialization history projection so per-keystroke pricing can observe buffered text and admitted media without writes or base64 work.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the live Send | $ and Queue | $ affordance with blocker-first native hover guidance, text-token and maximum-reply pricing details, honest unavailable/media fallbacks, and synchronous refresh from the canonical Console state. ADR-088 defines the detached lightweight history projection used by the pure cached estimator; accumulated-spend behavior is unchanged. Verification: 232 focused tests passed, mounted 80x24 and 160x40 visual/tooltip cases passed, Ruff lint passed, all new and task-owned files passed Ruff format, git diff checks passed, and the one-time Impeccable detector returned no findings. The existing chat_screen.py size ratchet remains red at 20,071 lines versus its 17,727-line budget; the feature adds only the composer callback binding there. Ruff format reports the same eight pre-existing drifted files at the pre-task base and this branch. Added a testing-evidence lesson for Textual layout settlement and explicitly enabling tooltips in run_test.
<!-- SECTION:NOTES:END -->
