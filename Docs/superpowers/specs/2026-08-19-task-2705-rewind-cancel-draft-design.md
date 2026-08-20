# TASK-2705 — Consume the Console `/rewind` draft

## Status

Approved in conversation on 2026-08-19.

## Problem

The Console restores a keypress-stashed slash-command draft before dispatching
the command handler. Most handled commands then replace or clear their own
invocation text. `/rewind` opens `ConsoleRewindModal` but leaves `/rewind` in
the composer, so Escape or **Never mind** returns focus to a draft that is no
longer useful and the next text is appended to the command.

The modal result is not the right place to infer an earlier arbitrary draft.
The recognized `/rewind` command is the current composer payload; there is no
separate pre-command text owned by this flow. The existing composer clear seam
is therefore the smallest truthful boundary.

## Goals

- Consume the exact `/rewind` invocation once the menu successfully opens.
- Leave the composer empty after Escape or **Never mind** when it contained
  only `/rewind`.
- Preserve **Restore to here**, which replaces the cleared draft with the full
  selected prompt text.
- Preserve **Summarize up to here**, which keeps the handled command consumed.
- Preserve the no-target refusal: when there is nothing to rewind, notify and
  leave the draft untouched.
- Remove the obsolete User Guide workaround.

## Non-goals

- Do not change generic slash-command dispatch or every command's draft policy.
- Do not reconstruct text from composer undo history.
- Do not change the rewind modal, tree mutation, summary worker, focus policy,
  command grammar, or prompt-row ordering.
- Do not introduce state, a helper abstraction, dependency, or configuration.

## Chosen design

After `_console_command_rewind` resolves an active session and at least one
prompt row, call the existing `_clear_console_composer_draft()` immediately
before pushing `ConsoleRewindModal`. This existing seam clears the composer and
synchronizes the command popup.

The modal callback remains unchanged:

- `None` only restores composer focus; the command is already consumed.
- `restore` replaces the empty draft with the selected prompt's full text.
- `summarize-up-to` starts the existing guarded worker and leaves the draft
  empty.

The early no-row return stays before the clear, so a refused command is not
silently consumed.

## Alternatives rejected

1. **Clear only in the cancellation callback.** This duplicates cleanup across
   result branches and leaves handled command text behind the modal until it
   closes.
2. **Restore from undo history.** Undo history does not identify a canonical
   "draft before `/rewind`" boundary, and replay would risk restoring unrelated
   edits or structured paste state.
3. **Consume every recognized slash command centrally.** Existing commands
   deliberately have different replacement/refusal behavior; changing their
   shared dispatcher is broader than this defect.

## Verification

- Add mounted product-path tests that dispatch an exact `/rewind` draft through
  the Console command route and assert it is cleared when the menu opens.
- Exercise both terminal negative paths: Escape and visible **Never mind**.
  Each must close only the modal, leave the draft empty, and return focus to the
  composer.
- Exercise **Restore to here** through the mounted modal and assert the selected
  full prompt replaces the cleared draft.
- Assert a no-prompts refusal leaves `/rewind` untouched and emits the existing
  warning.
- Run the rewind modal, restore wiring, command-dispatch, and safe-dismissal
  tests only, plus targeted Ruff, formatting, compilation, and diff checks.
- Remove the TASK-2705 quirk bullet from the User Guide after GREEN evidence.

## Failure handling and privacy

This change adds no I/O, persistence, logging, or user-data flow. The existing
no-target warning and modal mutation guards remain authoritative. Tests use
synthetic in-memory Console sessions.

## ADR check

ADR required: no

ADR path: N/A

Reason: this is a localized command-draft cleanup bug fix using an existing
Console seam. It changes no storage, ownership, service contract, security
boundary, dependency, or long-lived interaction architecture.
