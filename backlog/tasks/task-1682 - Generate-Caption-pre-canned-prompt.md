---
id: TASK-1682
title: 'Generate Caption pre-canned prompt'
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

Captioning an attached image meant writing the prompt by hand every time.

## Acceptance Criteria (the what)

- [x] The menu entry inserts a ready caption prompt into the composer
- [x] The entry is disabled with an explanation when nothing is attached

## Implementation Notes

See the batch commit. Live-verified in tmux, with Impersonate additionally
proven against a payload-logging stub provider (instruction + transcript
reach the model; suggestion lands in the composer; a second click replaces
rather than stacks).
