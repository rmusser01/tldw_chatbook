---
id: TASK-24417
title: >-
  Console hands-free Escape exits the loop before dismissing an open slash
  popup
status: Done
assignee:
  - @zcode
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

- [x] With a hands-free (or realtime) loop active and the slash popup open,
      Escape dismisses the popup first; a second Escape exits the loop.
- [x] Escape with no popup open still exits the loop from any point (the
      documented loop promise is preserved — the pre-existing
      `test_esc_exits_loop_and_restores_normal_esc_semantics` and
      `test_barge_in_and_esc_work_with_focus_off_the_composer` stay green).
- [x] Targeted tests for both orders (RED first: the unfixed code exited the
      loop AND left the orphaned popup open; then GREEN).

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: key-claim ordering fix inside existing screen seams; no contract or
boundary decision.

1. In the hands-free/realtime Escape branches, dismiss an open slash popup
   first and return early when one was open.
2. Tests: popup open + Escape → loop still active, popup closed; no popup +
   Escape → loop exits.

## Implementation Notes

Fixed 2026-08-29, TDD (RED reproduced a finding WORSE than filed: the first
Escape exited the loop AND left the orphaned popup floating open over the
dead loop). GREEN after the fix; 123 hands-free/realtime wiring + toggle
tests green (including the pre-existing exit-from-any-point guarantees).

- **Where the claim lives (the non-obvious part)**: the hands-free exit is
  delivered by a `priority=True` BINDING (`escape → exit_console_hands_free`)
  that fires BEFORE key bubbling, so `on_key` alone could never intercept
  it. The popup-first claim went into `ChatScreen.action_exit_console_
  hands_free` (the binding's destination; it covers BOTH engines per the
  wave-2 controller) plus the `on_key` hands-free Escape fallback branch.
- **Realtime**: the dev-tip `on_key` routes realtime keys through
  `self._realtime.handle_key`; a session-gated popup claim sits immediately
  before that call (`event.key == "escape" and self._realtime.session is
  not None and self._dismiss_console_command_popup()`), so the non-loop
  Escape path is untouched. Covered by symmetry with the tested hands-free
  path (the AC's "or"); the realtime rig is heavier than this claim
  warranted.
- Files: `tldw_chatbook/UI/Screens/chat_screen.py`,
  `Tests/UI/test_console_popup_etiquette.py` (1 new test: popup open →
  first Escape closes popup only, second exits the loop, exactly one
  session stop).
- ADR: not required (interaction-ordering fix inside existing seams).
