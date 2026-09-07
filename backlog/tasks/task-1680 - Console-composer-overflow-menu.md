---
id: TASK-1680
title: 'Console composer overflow menu'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
updated_date: '2026-07-31'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description (the why)

The composer action row is width-bounded and already carries Send/Stop/Mic/Attach/Save. New actions need a home that does not grow the row further.

## Acceptance Criteria (the what)

- [x] A hamburger button sits before Send
- [x] It opens a menu with Generate Image, Generate Caption, Narrate Entire Conversation, Impersonate
- [x] The action row still fits every existing button

## Implementation Notes

See the batch commit. Live-verified in tmux, with Impersonate additionally
proven against a payload-logging stub provider (instruction + transcript
reach the model; suggestion lands in the composer; a second click replaces
rather than stacks).
