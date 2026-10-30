---
id: TASK-38000
title: 'Console: Save chat''s capture promise ignores the runtime durable-capture gate'
status: To Do
assignee: []
created_date: '2026-10-30 04:58'
labels:
  - console
  - copy
dependencies: []
references:
  - >-
    backlog/tasks/task-37811 -
    Console-Temporary-chat-capture-copy-promises-capture-that-saving-would-not-turn-on.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-37811 makes the composer Menu's Save chat row say '(enables capture)' only when the saved chat resolves Capture On from the capture policy (the chat's own Capture choice, else the global setting). A send also needs the runtime to support durable capture (TASK-25814): the provider gateway must have its durable trace seam and the normalized-write rollout flag must be on. Without that, every send falls back to Capture Off. In that configuration the row promises traces that never come. The Trace Privacy dialog's effective-capture line has the same shape. The configuration is rare; raised by the second pre-merge review of TASK-37811.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With durable capture unsupported at runtime, the Save chat row makes no capture promise, and a test pins it against a gateway without durable capture
- [ ] #2 The Trace Privacy dialog's effective-capture line says capture cannot be recorded in that configuration, or this task records why it should not
- [ ] #3 With durable capture supported, the Save chat copy is unchanged and its existing pins stay green
<!-- AC:END -->
