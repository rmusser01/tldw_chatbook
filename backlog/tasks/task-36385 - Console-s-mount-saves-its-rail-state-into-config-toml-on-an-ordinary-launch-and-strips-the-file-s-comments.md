---
id: TASK-36385
title: Console's mount saves its rail state into config.toml on an ordinary launch and strips the file's comments
status: To Do
created_date: 2026-10-22 23:58
labels:
- follow-up
- console
- config
priority: medium
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/config.py
dependencies:
- TASK-35835
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-35835 (PR #3216, 2026-10-22). Once TASK-35835 lands, a quit no longer rewrites config.toml. A whole boot and quit still does not leave the file byte-identical. When the Console mounts, it saves `console.rail_state` (chat_screen.py around :12145-12153) through the normal config writer. That write dumps the file, so on an ordinary launch it strips the comments a user or the template left. Its only skip today is `unchanged_selected_config`, which holds for recovered profiles only. A default rail state written on mount is not a user change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An ordinary launch that changes no setting leaves config.toml byte-identical
- [ ] #2 A rail-state change the user makes is still persisted
- [ ] #3 A test pins both
<!-- AC:END -->
