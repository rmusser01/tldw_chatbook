---
id: TASK-32321
title: >-
  F6 rail focus lands on first content control not the collapse button
status: To Do
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
UX review A2. CONSOLE_FOCUS_TARGETS_BY_PANE (chat_screen.py ~636-644) makes the rail collapse button the first F6 focus target inside each rail, so a reflexive Enter hides the pane the user just entered. The target should be the first content control; the collapse button stays one Tab away.

Filed from the 2026-09-10 Console rail UX review (review item A2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 F6 into the Context rail focuses a content control (first section header or equivalent), not #console-context-rail-collapse
- [ ] #2 F6 into the Inspector rail focuses a content control, not #console-inspector-rail-collapse
- [ ] #3 The collapse buttons remain reachable by Tab from the content target
- [ ] #4 Existing F6 pane-cycle order (left rail -> transcript -> right rail -> composer) is unchanged
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED LIVE: one F6 press focused the left rail; a reflexive Enter collapsed the whole rail silently. Current map chat_screen.py:876-884 still targets the collapse buttons. alt+i open path bypasses to #console-send-authority-summary (TASK-24703) - follow that pattern.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
