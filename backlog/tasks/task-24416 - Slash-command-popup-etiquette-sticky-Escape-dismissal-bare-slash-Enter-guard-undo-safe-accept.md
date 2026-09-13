---
id: TASK-24416
title: >-
  Slash-command popup etiquette: sticky Escape dismissal, bare-slash Enter
  guard, undo-safe accept
status: Done
assignee:
  - @zcode
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

- [x] After Escape dismissal, subsequent draft *edits* (typing, backspace)
      do NOT re-open the popup while the draft stays in the same completion
      context; the dismissal latch re-arms when the draft leaves slash/skills
      -arg context (e.g. a space is typed, the draft is emptied or cleared).
- [x] The existing TASK-3790 guarantee holds: navigation keys after dismissal
      do not re-open the popup (existing test stays green).
- [x] Bare `/` + Enter no longer silently stages the first command: Enter on
      an empty-prefix (unfiltered) popup list falls through to the ordinary
      send path, where the unknown-command escape applies; Enter with a
      non-empty filtered prefix still accepts as today.
- [x] Accepting a suggestion preserves undo: Ctrl+Z immediately after an
      accept (Tab or Enter) restores the pre-accept draft.
- [x] Targeted tests cover each behavior; the existing popup and
      draft-changed suites stay green (159 neighboring tests).

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: interaction-behavior fixes inside existing popup/screen seams; the
completion-context helper extends an existing pure module, no new contract
or boundary.

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

Fixed 2026-08-29, TDD (4 RED tests reproduced the three live findings —
popup re-opened after Escape + typing, bare `/`+Enter staged `/prompt `,
accept wiped undo — then GREEN; the two control tests, re-arm and
filtered-accept, passed before and after). 159 neighboring popup/
draft-changed/history/composer tests green; all three behaviors re-verified
live in tmux (Escape+type stays closed with draft visible; bare `/`+Enter
shows the honest "Unknown command /" transcript row with the draft retained;
accept then Ctrl+Z restores the pre-accept draft).

- **Sticky dismissal**: `completion_context_for_draft` (new, pure, in
  `Chat/console_command_suggestions.py`) reports which context
  (`command`/`skills_arg`) and filter prefix a draft carries.
  `_dismiss_console_command_popup` latches the context it dismissed in;
  `_sync_console_command_popup` keeps the popup hidden while the draft
  stays in that context and clears the latch the moment the draft leaves
  completion context (so a fresh `/` re-arms). Keyed on context, not draft
  text — text moves every keystroke.
- **Bare-slash Enter guard**: `on_key`'s popup Enter branch accepts only
  with a non-empty filter prefix; on an empty prefix (bare `/` or bare
  `/skills `) it dismisses the popup (latched, so the restored
  unknown-command draft does not re-open it) and falls through to the
  ordinary send path. Tab-accept is unchanged (Tab is the deliberate
  completion key).
- **Undo-safe accept**: new `ConsoleComposerBar.replace_draft_via_completion`
  records the pre-accept draft onto the undo stack
  (`_record_undo_snapshot(coalesce=False)`), banks the stacks across
  `load_draft`'s intentional wipe, and reinstates them — accept now has
  typed-edit undo/redo semantics instead of session-switch semantics.
- Files: `tldw_chatbook/Chat/console_command_suggestions.py`,
  `tldw_chatbook/UI/Screens/chat_screen.py`,
  `tldw_chatbook/Widgets/Console/console_composer_bar.py`,
  `Tests/UI/test_console_popup_etiquette.py` (6 tests: sticky, re-arm,
  bare-slash guard, filtered-accept control, Tab-undo, Enter-undo).
- ADR: not required (interaction fixes inside existing seams; the context
  helper extends an existing pure module).
