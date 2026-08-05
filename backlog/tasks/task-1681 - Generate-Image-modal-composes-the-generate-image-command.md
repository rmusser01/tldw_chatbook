---
id: TASK-1681
title: 'Generate Image modal composes the /generate-image command'
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

Building the command by hand means knowing the `:backend`/`@style` grammar.
A modal should collect the options and hand back the command for review.

## Acceptance Criteria (the what)

- [x] A modal collects prompt, backend and style
- [x] It previews the exact command as you type
- [x] Accepting pastes the command into the composer rather than generating
      immediately, so the existing /generate-image handler stays the single
      execution path

## Implementation Notes

See the batch commit; live-verified (preview updated per keystroke, and the
composed command landed in the composer).
