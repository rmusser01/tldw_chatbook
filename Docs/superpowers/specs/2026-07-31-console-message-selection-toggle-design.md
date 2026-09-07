# Console Message Selection Toggle Design

**Date:** 2026-07-31
**Status:** Approved

## Goal

Let a Console user deselect the active transcript message by activating that
same message again with either the mouse or keyboard, without changing how
message navigation or contextual actions work.

## Interaction Contract

- Clicking an unselected message selects it and shows its contextual action
  row.
- Clicking the selected message again clears selection and hides the action
  row.
- Pressing Enter while no message is selected keeps the existing behavior of
  selecting the first message.
- Pressing Enter while a transcript message is selected clears that selection.
- Pressing Enter while a contextual action button has focus continues to invoke
  that action; it does not clear the message selection.
- Up/Down and J/K remain absolute navigation commands. Reaching a boundary or
  programmatically selecting the current message must not toggle it off.
- Escape and transcript negative-space clicks retain their existing deselection
  behavior.

## Design

`ConsoleTranscript` will gain a focused `toggle_message_selection(message_id)`
operation. It will validate the message ID through the same local message list
used by `select_message()`. When the requested ID is already selected, it will
delegate to `action_clear_selection()`; otherwise it will delegate to
`select_message()`.

`ConsoleTranscriptMessage.on_click()` will call the toggle operation instead of
the absolute selection operation. `ConsoleTranscript.action_confirm_selection()`
will keep selecting the first message when no selection exists, but will call
the toggle operation for an existing selection.

The existing `select_message()` method remains idempotent and unchanged for
arrow-key navigation, action focusing, and other internal callers. This keeps
toggle semantics limited to explicit user activation of a message.

Both toggle branches reuse the existing refresh and selection-change
notification paths. An unknown or stale message ID remains a no-op, matching
current selection behavior. No CSS, persistence, message model, or screen-level
state changes are required.

## Testing

Focused mounted Textual regressions will verify:

1. Clicking a message selects it; clicking the same row again clears
   `selected_message_id` and removes the contextual action row.
2. Keyboard navigation can select a message; Enter on that selected message
   clears `selected_message_id` and removes the action row.
3. Enter with no selection still selects the first message, and focused action
   buttons retain their existing Enter behavior through the unchanged button
   handler coverage.

The focused Console transcript test module will run first, followed by the
relevant broader Console UI tests and `git diff --check`.

## Scope and Decision Record

ADR required: no

ADR path: N/A

Reason: This is a routine interaction correction within the existing Console
transcript selection boundary. It does not change storage, ownership, service
contracts, security policy, dependencies, or long-lived application structure.

Out of scope: multi-select, persistent selection, visual restyling, changes to
message actions, and changes to transcript navigation boundaries.
