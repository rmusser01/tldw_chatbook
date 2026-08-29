---
id: TASK-24416
title: >-
  Slash-command popup etiquette: sticky Escape dismissal, bare-slash Enter
  guard, undo-safe accept
status: To Do
assignee: []
created_date: '2026-08-29'
updated_date: '2026-08-29'
labels:
  - console
  - defect
  - ux
priority: high
dependencies: []
---

## Description (the why)

From the same user report ("the `/` command trigger … funky in a bad way",
2026-08-29 live review) as TASK-24415 — three popup behaviors that make the
trigger feel grabby, all confirmed live in the real terminal:

1. **Escape dismissal is not sticky.** Escape closes the popup, but the very
   next keystroke re-opens it: every `DraftChanged` runs the un-gated
   `_sync_console_command_popup` (`chat_screen.py`, the
   `_handle_console_composer_draft_edit` path), and backspacing through a
   command token re-opens it on every character. There is no way to compose a
   slash-prefixed draft with the popup dismissed. TASK-3790's gate only
   protects *navigation* keys after dismissal — its test
   (`test_escape_dismissal_survives_a_following_arrow_key`) asserts arrows
   don't re-open; nothing covers typing, and live behavior shows typing does.
2. **Bare `/` + Enter silently stages the first command.** With the popup open
   on a bare `/` (empty prefix, full list), Enter accepts the top suggestion —
   the draft becomes `/prompt ` with no other visible effect. A user probing
   the trigger gets a command staged instead of a send or a no-op.
3. **Accept wipes undo.** Accepting routes through `composer.load_draft()`,
   which by design (TASK-1281) clears the undo/redo stacks — an accidental
   accept cannot be Ctrl+Z'd back to the pre-accept draft.

## Acceptance Criteria

- [ ] After Escape dismissal, subsequent draft *edits* (typing, backspace)
      do NOT re-open the popup while the draft stays in the same completion
      context; the dismissal latch re-arms when the draft leaves slash/skills
      -arg context (e.g. a space is typed, the draft is emptied or cleared).
- [ ] The existing TASK-3790 guarantee holds: navigation keys after dismissal
      do not re-open the popup (existing test stays green).
- [ ] Bare `/` + Enter no longer silently stages the first command: Enter on
      an empty-prefix (unfiltered) popup list falls through to the ordinary
      send path, where the unknown-command escape applies; Enter with a
      non-empty filtered prefix still accepts as today.
- [ ] Accepting a suggestion preserves undo: Ctrl+Z immediately after an
      accept (Tab or Enter) restores the pre-accept draft.
- [ ] Targeted tests cover each behavior; the existing popup and
      draft-changed suites stay green.

## Implementation Plan

1. Add a completion-context helper (command mode vs skills-arg mode vs none)
   next to `suggestions_for_draft` so the screen can key a dismissal latch on
   context, not draft text.
2. Latch in `_sync_console_command_popup`: dismissed stays dismissed while
   the context is unchanged; leaving context clears the latch. Dismissal
   sites (`_dismiss_console_command_popup`) set the latch.
3. Enter guard in `on_key`'s popup branch: accept only when the draft's
   command-mode prefix is non-empty (bare `/` falls through to send).
4. Undo-safe accept: snapshot/record the pre-accept draft onto the undo stack
   around the draft replacement in `_accept_console_command_popup` (avoid
   `load_draft`'s history wipe; reuse the composer's undo-recording seams).
5. Tests first for each behavior (RED), then implement (GREEN); re-verify the
   Escape/re-open and bare-slash paths live in tmux.

## Implementation Notes

(added after implementation)
