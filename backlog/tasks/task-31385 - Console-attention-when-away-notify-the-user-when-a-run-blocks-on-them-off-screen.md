---
id: TASK-31385
title: >-
  Console attention when away: notify the user when a run blocks on them
  off-screen
status: To Do
assignee: []
created_date: '2026-09-04 19:29'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When an agent run blocks on the user -- an approval, a skill or worktree confirm, or an ask_user question -- the only attention affordances are inside Console: the rail badge, the parked-round toast, and the wake badge. A user on another screen (Library, Settings) or in another terminal window learns nothing until they come back, while the run sits paused for as long as ADR-067's indefinite default allows. Sub-project D of the design spec (2026-08-19-console-user-interaction-design.md section 4): a terminal bell and/or OSC notification, and a cross-screen badge on the Console nav item, raised when a round mounts or parks while Console is not the visible screen, and cleared when it resolves. Both reference implementations (Claude Code, Codex) ring or notify on a blocking prompt.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A blocking round raised while Console is not the visible screen produces a terminal notification (bell or OSC 9/777) governed by a [console] setting that defaults on
- [ ] #2 The Console entry in the app navigation shows a pending-interrupt badge until the round resolves
- [ ] #3 A round raised while Console is visible produces no bell
- [ ] #4 Headless and test runs never emit terminal control sequences
<!-- AC:END -->
