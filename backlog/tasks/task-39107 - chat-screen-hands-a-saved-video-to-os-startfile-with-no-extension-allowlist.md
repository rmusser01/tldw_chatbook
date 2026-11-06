---
id: TASK-39107
title: 'chat_screen hands a saved video to os.startfile with no extension allowlist'
status: To Do
assignee: []
created_date: '2026-11-06 09:00'
labels:
  - security
dependencies: []
references:
  - tldw_chatbook/UI/Screens/chat_screen.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`chat_screen.py:19070` hands a saved video to `os.startfile` with no extension allowlist (Windows only). It exists on dev today; this task's PR does not fix it. Found by the Roleplay B2b follow-up burn-down (2026-11-05), read on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Only an allowlist of video extensions reaches os.startfile; anything else is refused with a notice
- [ ] #2 A test pins an accepted and a refused extension
<!-- AC:END -->
