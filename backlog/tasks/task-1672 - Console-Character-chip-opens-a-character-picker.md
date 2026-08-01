---
id: task-1672
title: 'Console Character chip opens a character picker'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description (the why)

The Character/Assistant chip was inert; swapping character meant leaving Console for Roleplay. User asked for a picker that can swap or start a new chat.

## Acceptance Criteria (the what)

- [x] The chip opens a searchable character picker
- [x] The user chooses per pick: swap this chat, or start a new one
- [x] A greeting seeds only into an empty chat

## Implementation Notes

See the batch commit; live-verified in tmux.
