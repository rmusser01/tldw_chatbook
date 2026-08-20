# TASK-2705 — Consume the Console `/rewind` draft

## Status

Approved in conversation on 2026-08-19; amended after independent review to
preserve post-Enter input and dispatch-failure rollback.

## Problem

The Console restores a keypress-stashed slash-command draft before dispatching
the command handler. Most handled commands then replace or clear their own
invocation text. `/rewind` opens `ConsoleRewindModal` but leaves `/rewind` in
the composer, so Escape or **Never mind** returns focus to a draft that is no
longer useful and the next text is appended to the command.

The modal result is not the right place to infer an earlier arbitrary draft.
The Enter path already owns the exact command invocation as a
`ConsoleDraftStash`, separately from any keystrokes that land in the fresh
composer before the queued send handler runs. The fix must preserve that
separation instead of clearing the live composer.

## Goals

- Consume an argument-free `/rewind` invocation once the menu successfully
  opens.
- Leave the composer empty after Escape or **Never mind** when it contained
  only `/rewind`, while preserving text typed after the Enter keypress.
- Preserve **Restore to here**, which replaces the current draft with the full
  selected prompt text.
- Preserve **Summarize up to here**, which keeps the handled command consumed.
- Preserve the no-target refusal: when there is nothing to rewind, notify and
  leave the draft untouched.
- Remove the obsolete User Guide workaround.

## Non-goals

- Do not change every command's draft policy; keep the send-path exception
  local to argument-free `/rewind`.
- Do not reconstruct text from composer undo history.
- Do not change the rewind modal, tree mutation, summary worker, focus policy,
  command grammar, or prompt-row ordering.
- Do not introduce persistent state, a helper abstraction, dependency, or
  configuration.

## Chosen design

The existing send path already captures a keyboard invocation with
`ConsoleComposerBar.stash_draft_for_send()` before it posts the Send button
message. Any later typing therefore lives in the fresh composer. For the exact
registry result `kind == "command"`, `name == "rewind"`, and `args == ""`, the
send handler will keep the captured invocation separate instead of restoring
it before dispatch:

1. If the command came from Enter, keep the already-captured stash separate
   and dispatch `_console_command_rewind` without putting that invocation back
   in front of the live draft.
2. If it came from the visible Send button, leave the live argument-free
   command in place while `_console_command_rewind` runs and retain the exact
   composer reference, dispatched text, and the `(edit_serial, generation)`
   pair from its existing immutable `capture_draft_snapshot()`. The handler
   currently has no suspension point before its synchronous
   `push_screen`/return.
3. Have `_console_command_rewind` return whether it actually opened the modal.
   A session with prompt rows returns `True` after `push_screen`; the existing
   no-row warning returns `False`.
4. On `True`, discard the keyboard invocation stash. For visible Send, clear
   through the existing `_clear_console_composer_draft()` seam only when the
   same mounted composer still has the captured edit serial, generation, and
   dispatched text. The edit serial detects ordinary typing (including an
   edit-and-retype of identical bytes); generation detects programmatic draft
   scope replacement. Neither treats harmless focus/cursor movement as draft
   replacement. If identity, either counter, or text changed, preserve the
   draft rather than risking user-data loss.
   On `False`, restore the keyboard stash with `restore_stashed_draft`,
   preserving the existing rejected-send ordering in front of any text typed
   after Enter; the visible-Send draft was never changed. If dispatch raises
   before opening succeeds, perform the same keyboard-stash restoration before
   propagating the existing error; this path must not turn a modal-launch
   failure into draft loss.

All other command parses keep the current restore-before-dispatch path.
In particular, `/rewind anything` has non-empty `parse.args`, remains outside
this cleanup, and retains its current behavior. This task does not redefine the
command grammar or add argument validation.

The modal callback remains unchanged:

- `None` makes the existing immediate composer-focus request; the shared safe
  dismissal policy then restores the actual opener. A keyboard invocation's
  opener remains the composer, while a visible-Send invocation's opener is the
  Send button. This task does not change that focus policy.
- `restore` deliberately replaces the current draft (including any post-Enter
  text) with the selected prompt's full text, preserving the existing
  `replace=True` contract.
- `summarize-up-to` starts the existing guarded worker and leaves the draft
  as it was after the invocation was consumed.

The early no-row return reports `False`, so a refused command is not silently
consumed.

## Alternatives rejected

1. **Clear the live composer when the modal opens.** This loses keystrokes typed
   after Enter because `restore_stashed_draft` prepends the captured command to
   the fresh live draft before the queued handler runs.
2. **Clear only in the cancellation callback.** This duplicates cleanup across
   result branches and leaves handled command text behind the modal until it
   closes.
3. **Restore from undo history.** Undo history does not identify a canonical
   "draft before `/rewind`" boundary, and replay would risk restoring unrelated
   edits or structured paste state.
4. **Consume every recognized slash command centrally.** Existing commands
   deliberately have different replacement/refusal behavior; changing their
   shared dispatcher is broader than this defect.

## Verification

- Add mounted product-path tests that dispatch an argument-free `/rewind`
  through both Enter and visible-Send routes and assert only its captured
  invocation is consumed when the menu opens.
- Interleave text after the Enter keypress but before the queued send handler;
  assert Escape and **Never mind** preserve that text exactly.
- Exercise both terminal negative paths: Escape and visible **Never mind**.
  Each must close only the modal, leave no command invocation in the draft,
  and preserve any post-Enter text. Assert the existing opener-specific focus:
  keyboard dispatch returns to the composer; visible-Send dispatch returns to
  the Send button.
- Exercise **Restore to here** through the mounted modal and assert the selected
  full prompt replaces the current draft.
- Assert a no-prompts refusal restores `/rewind` ahead of any post-Enter text
  and emits the existing warning.
- Inject a modal-launch exception and assert the keyboard command stash is
  restored before the existing error propagates; the visible-Send draft stays
  unchanged.
- Simulate visible-Send dispatches whose composer identity, edit serial, or
  generation changes before success returns—including edit-and-retype to
  identical text—and assert the newer draft is not cleared.
- Assert `/rewind anything` stays on the ordinary command-dispatch path and is
  not auto-consumed by this fix.
- Run the rewind modal, restore wiring, command-dispatch, and safe-dismissal
  tests only, plus targeted Ruff, formatting, compilation, and diff checks.
- Remove the TASK-2705 quirk bullet from the User Guide after GREEN evidence.

## Failure handling and privacy

This change adds no I/O, persistence, logging, or user-data flow. The existing
no-target warning and modal mutation guards remain authoritative. A dispatch
exception restores the keyboard stash before following the existing error
path. Tests use synthetic in-memory Console sessions.

## ADR check

ADR required: no

ADR path: N/A

Reason: this is a localized command-draft cleanup bug fix using an existing
Console seam. It changes no storage, ownership, service contract, security
boundary, dependency, or long-lived interaction architecture.
