---
id: TASK-23104
title: Settings State banner renders doubled and self-colliding
status: To Do
assignee: []
created_date: '2026-08-28 14:05'
labels:
  - ux
  - settings
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The banner composition at settings_screen.py:6413 prepends 'State: {badge} | ' to scope strings that already embed their own 'State: ...' (6466, 6479-6482), producing 'State: Read-only here | State: Active | ...'; and the whole banner renders twice on Overview and all 11 domain categories (pinned at 16631 plus in-card via 12405/15116). The banner is the sole carrier of the save contract (task-1717 built it because five save models coexist); a stuttering duplicated contract line teaches users to stop reading it, un-fixing the original problem. P1 from the 2026-08-28 critique (.impeccable/critique/2026-08-28T06-32-49Z__tldw-chatbook-ui-screens-settings-screen-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each settings category renders exactly one State banner
- [ ] #2 The banner text contains exactly one 'State:' segment; scope text no longer embeds a second
- [ ] #3 Verified at runtime on Overview, one domain-defaults category, and one draft-model category
<!-- AC:END -->
