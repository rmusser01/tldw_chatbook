---
id: TASK-1683
title: 'Impersonate drafts the user's next reply'
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

Users wanted the model to propose their own next message for review and editing before sending.

## Acceptance Criteria (the what)

- [x] The current model drafts the USER's next message from the transcript
- [x] The suggestion is appended on a new line and never replaces existing user text
- [x] Clicking again replaces the previous suggestion instead of stacking
- [x] If the user edited the suggestion the new one is appended instead

## Implementation Notes

See the batch commit. Live-verified in tmux, with Impersonate additionally
proven against a payload-logging stub provider (instruction + transcript
reach the model; suggestion lands in the composer; a second click replaces
rather than stacks).
