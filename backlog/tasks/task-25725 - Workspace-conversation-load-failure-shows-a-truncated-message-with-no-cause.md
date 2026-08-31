---
id: TASK-25725
title: Workspace conversation load failure shows a truncated message with no cause
status: Done
assignee: []
created_date: '2026-08-31 05:09'
updated_date: '2026-08-31 06:36'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When the workspace conversation list fails to load, the rail shows a clipped string and a bare Retry control. Two distinct root causes are collapsed into one generic sentence, the real exception is swallowed into a debug log, and the visible text is cut off by the rail width so the user cannot read even the generic message.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The rail error is legible at the rail's actual width
- [ ] #2 Distinct failure causes produce distinct user-facing messages
- [ ] #3 The underlying exception is recorded at a level the user can be pointed to
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two unrelated failures shared one sentence: membership-token-unknown and a swallowed list_conversations exception both rendered 'Workspace conversations are unavailable.', which the rail then clipped to 'Workspace conversations a...' -- naming neither cause. Split into WORKSPACE_CONVERSATIONS_ACCESS_UNKNOWN ('Workspace access unknown.') and WORKSPACE_CONVERSATIONS_LOAD_FAILED ("Couldn't load conversations."), both short enough to read at rail width, and raised the swallowed exception log from debug to warning so 'check the app log' actually leads somewhere. Four pinned tests updated to their real cause. Verified against baseline: the 13 failures in test_console_workspace_controller.py are pre-existing on clean dev, unchanged by this.
<!-- SECTION:NOTES:END -->
