---
id: TASK-1671
title: 'Console chips show 25 chars of the model name'
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

The chip cap of 22 cells cut model ids mid-token ("Model: claude-3-hai…"). User asked to see up to 25 characters of the model NAME.

## Acceptance Criteria (the what)

- [x] The model name renders up to 25 chars before ellipsis
- [x] Short chips are unchanged (width is auto, the cap only bites long labels)

## Implementation Notes

See the batch commit; live-verified in tmux.
