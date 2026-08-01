---
id: task-1670
title: 'Console Provider-Model chips open the model popover'
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

Provider and Model read as inert labels; the quick model popover was reachable only via Alt+M, which is undiscoverable. Both chips are two views of one setting, so either is a reasonable click target.

## Acceptance Criteria (the what)

- [x] Provider and Model chips activate on click and Enter/Space
- [x] They open the SAME popover Alt+M opens, not a fork

## Implementation Notes

See the batch commit; live-verified in tmux.
