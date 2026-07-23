---
id: TASK-413
title: 'Fix [World Info: N entries] indicator one-send lag (mounted before resolve)'
status: To Do
assignee: []
created_date: '2026-07-21 16:21'
labels:
  - bug
  - chat
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In handle_chat_send_button_pressed (chat_events.py), the [World Info: N entries activated] indicator ChatMessage is mounted (~line 1180, reads app.current_world_info_active/count) BEFORE the send path resolves world-info and updates those reactives (~line 1395). So the indicator on send K reflects send K-1's world-info, not the current send. Pre-existing (the mount-before-resolve ordering predates P2g-3); surfaced by the opus whole-branch review + Qodo on PR #740. Fix by reordering — resolve world-info before mounting the indicator, or move the indicator mount after the resolution — so the indicator reflects the CURRENT send.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The [World Info: N entries] indicator reflects the CURRENT send's matched-entry count/active state (not the previous send's),No regression to the world-info injection itself (byte-parity preserved),import tldw_chatbook.app OK; a test covers the corrected ordering
<!-- AC:END -->
