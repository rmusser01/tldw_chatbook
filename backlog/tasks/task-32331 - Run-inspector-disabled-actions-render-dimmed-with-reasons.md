---
id: TASK-32331
title: >-
  Run inspector disabled actions render dimmed with reasons
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review C2. _button_for_action (console_run_inspector.py ~320-352) fully hides disabled action buttons AND their disabled_reason (display:none), so users cannot discover an action exists or why it is unavailable; TASK-1843 removed a permanently-dead button for exactly this. Render disabled actions dimmed with their one-line reason; reserve hiding for actions that are never applicable.

Filed from the 2026-09-10 Console rail UX review (review item C2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Disabled run-inspector actions render visibly but dimmed, with their disabled_reason text on the same row or adjacent line
- [x] #2 Actions with no reason text and never-applicable actions remain hidden
- [x] #3 Enabled action rendering is unchanged (variant, height, tooltip)
- [x] #4 Widget tests cover enabled, disabled-with-reason, and hidden cases
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
**Close-out.** Closed without code change: re-verification against dev tip found TASK-24606 already shipped visible disabled actions with reason rows at the exact site this review flagged.

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

ALREADY FIXED ON DEV by TASK-24606: disabled actions render as visible disabled Buttons and disabled_reason renders as its own Static row (console_run_inspector.py:588-628). Closing.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
