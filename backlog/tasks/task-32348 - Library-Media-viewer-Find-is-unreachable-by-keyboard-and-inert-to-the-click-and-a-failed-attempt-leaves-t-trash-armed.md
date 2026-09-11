---
id: TASK-32348
title: >-
  Library Media viewer: Find is unreachable by keyboard and inert to the click,
  and a failed attempt leaves 't trash' armed
status: To Do
assignee: []
created_date: '2026-09-11 06:15'
labels:
  - library
  - media
  - keyboard
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Find is neither in the Tab order (10 Tabs each way) nor opened by the same SGR click that its sibling buttons accept; typing a query after the failed click fired the t accelerator and armed 'Delete this media?' (B D4/D4a, caps 27/28). No Find key binding exists in the screen's bindings. Docs promise a focused search bar (media-and-conversations.md:529). Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Find opens from the keyboard (a binding with a footer chip) and is in the viewer's Tab order
- [ ] #2 No single-key destructive accelerator fires while the viewer's Find gesture is pending
- [ ] #3 Pinned
<!-- AC:END -->
