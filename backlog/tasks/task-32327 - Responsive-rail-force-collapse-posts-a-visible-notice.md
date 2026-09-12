---
id: TASK-32327
title: >-
  Responsive rail force-collapse posts a visible notice
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review B3. Width-band rules silently force-collapse rails (<150 Inspector, <100 Context, <84 single-pane) and resolve both-open conflicts (console_rail_state.py resolve_console_rail_priority). Users see their rail disappear with no explanation. Post a transient notice the first time per session a responsive override hides a rail the user had open.

Filed from the 2026-09-10 Console rail UX review (review item B3).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: resize test asserts one notice naming Context + reopen, none on repeat crossings. 2. Capture pre-sync visibility in the width adapter; add the once-per-session helper. 3. Document in Small terminals. 4. Run narrow layout + edge geometry suites.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When a responsive width rule closes a rail the user had open, a transient notice names what happened and how to reopen (once per session per rail, not per resize tick)
- [x] #2 Resize spam does not produce repeated notifications (debounced/once-per-session)
- [x] #3 Preferences are still not rewritten by responsive overrides (existing behavior kept)
- [x] #4 Unit tests cover the notice firing once and not on subsequent collapses
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach.** `_adapt_console_workspace_to_width` now captures which rails
were visibly open before a width-band crossing, syncs visibility, then
calls `_notify_console_responsive_rail_collapse`: for each rail that WAS
open and is now force-closed (and whose notice has not fired this
session), posts a transient warning notice naming the rail and its reopen
route (Context handle / Alt+I), 6s timeout. Once-per-session-per-rail via
a screen set — resize spam and repeat crossings cannot re-notify.
Stored preferences remain untouched (responsive rules still never rewrite
them); a rail whose preference already says closed produces no notice
(rule and user agree). `app_instance.notify` failures are swallowed to
never break layout.

**ADR check.** Not required — transient feedback on an existing seam.

**Modified.** `UI/Screens/chat_screen.py` (adapter capture + helper +
session set), `Tests/UI/test_console_narrow_layout.py` (+1 test: notice
fires once, does not re-fire on repeat crossings),
`Docs/User_Guide/console.md` (Small terminals section). Verified:
narrow-layout + edge-rail-geometry suites — 42 passed (1 known
pre-existing failure).

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED LIVE: rail collapsed with zero feedback (Enter on focused collapse button). No notify() anywhere near _adapt_console_workspace_to_width (chat_screen.py:21718-21780).
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
