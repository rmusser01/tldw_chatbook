---
id: TASK-37021
title: >-
  Console: revealing a hidden rail renders the deferred Character snapshot after
  its first frame
status: To Do
assignee: []
created_date: '2026-10-26 13:55'
labels:
  - console
  - performance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-35970 review (2026-10-26). TASK-35970 defers the Character browser's render while it cannot be seen. When the Character section is opened by its own disclosure, the rail starts the deferred render before the body is displayed and holds allocation passes until it has mounted, so the first frame shows the newest chats and the open's reveal survives. When it was the whole Context rail that was hidden (a narrow terminal, or the rail closed) and the section was already open, the browser renders from its Show event instead: that happens after the first frame is painted and holds no allocation passes, so one frame can show the chats from before the rail was hidden, and a pass can measure the half-mounted body (the reveal-drop TASK-35970 fixed for the disclosure path).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Revealing the Context rail with the Character section open shows the newest Character snapshot in every painted frame
- [ ] #2 No allocation pass measures the Character body while that render is mounting, and the section's reveal behaves as on a disclosure open
<!-- AC:END -->
