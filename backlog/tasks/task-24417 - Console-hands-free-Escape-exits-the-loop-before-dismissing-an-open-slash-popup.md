---
id: TASK-24417
title: >-
  Console hands-free Escape exits the loop before dismissing an open slash
  popup
status: To Do
assignee: []
created_date: '2026-08-29'
updated_date: '2026-08-29'
labels:
  - console
  - defect
  - ux
priority: low
dependencies: []
---

## Description (the why)

Found in the 2026-08-29 `/` command trigger review (same session as
TASK-24415 / TASK-24416), lower severity: in `ChatScreen.on_key`, the
hands-free/realtime branch claims Escape *before* any popup-dismissal logic
runs (the branch sits above `_should_capture_console_input`, and the popup's
Up/Down/Enter claims come after). While a hands-free or realtime loop is
active with the slash-command popup open, pressing Escape exits the entire
loop instead of just closing the popup — the user asked the overlay to go
away and lost the loop's mode with it.

## Acceptance Criteria

- [ ] With a hands-free (or realtime) loop active and the slash popup open,
      Escape dismisses the popup first; a second Escape exits the loop.
- [ ] Escape with no popup open still exits the loop from any point (the
      documented loop promise is preserved).
- [ ] Targeted tests for both orders.

## Implementation Plan

1. In the hands-free/realtime Escape branches, dismiss an open slash popup
   first and return early when one was open.
2. Tests: popup open + Escape → loop still active, popup closed; no popup +
   Escape → loop exits.

## Implementation Notes

(added after implementation)
