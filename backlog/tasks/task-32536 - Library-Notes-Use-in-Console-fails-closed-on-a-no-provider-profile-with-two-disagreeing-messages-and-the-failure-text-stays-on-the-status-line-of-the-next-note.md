---
id: TASK-32536
title: >-
  Library Notes: "Use in Console" fails closed on a no-provider profile with two
  disagreeing messages, and the failure text stays on the status line of the
  next note
status: Done
assignee:
  - '@claude'
created_date: '2026-09-13 06:45'
updated_date: '2026-09-14 14:47'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors, personas researcher/student and Alex, Edit workflow. P1 (A) / D1 + D2 (B).

**What happened.** Fresh profile (the default first-run state, no provider): Tab to "Use in Console" (footer chip "enter use in Console") → status "Use in Console failed — check Console readiness and try again. · Next: Review the error, then keep editing." plus a toast "Copy or link blocked Library sources into the active workspace before using them in Console." (B 14; A 55/56). No link to set up a provider, no statement that nothing was staged, and no error anywhere on screen to review. Escape, open note B, edit, let it save: fifteen minutes later B's status line reads "Saved 06:03 · Use in Console failed — check Console readiness and try again. · Next: Review the error, then keep editing." and the list status line carries "…failed — check Console readiness and…" too (B 33, 35, 53; A 68 on the power profile). Captures: A 55, 56, 68; B 14, 33, 35, 53.

**Cause.** PROVEN (wave 4, task-4 Step 1, fresh profile `notes-crit/wave4/console-handoff/fresh`, captures `wave4-caps/console-handoff/handoff-00..03`). (1) The blocker is the workspace gate, not Console readiness: headless on the same profile, `screen._library_workspace_depth_state(refresh=True)` reports `workspace_name="Local Default"`, `context_handoff_enabled=False`, and every note row `reason_code="not_in_active_workspace"` (`handoff-03-headless-depth-state.txt`) -- `display_state.py` gates the whole hand-off on `blocked_count == 0` and `eligibility.py` blocks a note that is not a member of the active workspace, which no fresh note is. `_open_selected_library_note_handoff` then toasts `workspace_state.context_handoff_tooltip` ("Copy or link blocked Library sources...") and returns False, and `handle_library_note_use_in_console` reports the literal `failure_next_action="check Console readiness and try again"` -- two messages naming two different blockers (`handoff-00-two-messages.txt`). (2) "Next: Review the error" comes from `library_notes_canvas.py:969`, which appends it whenever the word "failed" is in the status. (3) Persistence: `_finish_library_notes_operation` keeps the terminal `failed` state and neither `_reset_library_note_editor_state` (note switch) nor a successful save clears it -- opening another EXISTING note from the list (a row press; ctrl+n's create path clears it via `_select_library_rail_row`) carries the failure onto that note's status line, still there 2 min 19 s and one autosave later (`handoff-01-stale-failure.txt`). (4) The list pane beside the editor carries it because `_build_library_notes_state` (`library_screen.py:16522`) reads `_library_notes_operation_for_active_region()` while the EDITOR is the active region, so an editor-region operation is painted into the navigator's status row (`handoff-02-list-status.txt`).

**Ruling (controller, product).** Use in Console links the open note to the active workspace on the way (the "Use as source" precedent, task-32107) and then stages it; a no-provider profile lands on Console's own setup card with the note staged. One message per failed hand-off, the remedy inline; the failure clears on note switch and on a successful save.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 On a profile with no ready model, Use in Console links the open note to the active workspace, stages it, and lands on Console's own setup card; when the link cannot be made it shows one sentence naming the real blocker with the remedy inline (for example "restart Chatbook, then try again")
- [x] #2 Exactly one message is shown for a failed hand-off (status line or toast, not both), and "Next: Review the error" is never offered when no error is on screen
- [x] #3 Opening another note, or saving the current one, clears a previous action failure from the editor status line, and the list status line never carries it
- [x] #4 A regression test performs a failed hand-off, opens a second note and saves it, and asserts the second note's status line reads Saved with no failure text
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live on the fresh profile at 235x52 and headlessly (captures handoff-00..03); write the PROVEN cause into the Description and reword AC#1 per the controller ruling (done before any code change).
2. RED pins in `Tests/UI/test_library_notes_w4_console_handoff.py`: auto-link + stage on a fresh profile with no toast and no "check Console readiness"; a registry-less hand-off shows exactly one line naming the blocker with its remedy and no "Review the error"; a failure clears on note change and on a successful save, and never reaches the list status.
3. Fix: `_link_open_note_to_active_workspace()` on the controller (the conversation precedent minus the receipt); `_open_selected_library_note_handoff` links when the gate is closed and returns the blocker sentence instead of toasting; `LibraryNotesOperationState.failure_line` carries the one-line failure copy; `_reset_library_note_editor_state` and a SAVED outcome drop a terminal failed operation; `_build_library_notes_state` only shows navigator-region operations; the canvas does not append its generic Next when the transfer status already names one; the hand-off log becomes metadata-only.
4. Diagnostic inventory `--write`, GREEN + dev-baseline FAILED-name comparison, live re-verify at 235x52 and 100x30, guide `notes.md` "Use a note in Console" + stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Use in Console now links the note it is given.** The refusal named the remedy
("Copy or link blocked Library sources into the active workspace") without ever
offering it, and the status line named a *different* blocker. Per the
controller ruling, `_open_selected_library_note_handoff` now calls a new
`_link_open_note_to_active_workspace()` when the workspace gate is closed --
the conversation reader's "Use as source" precedent (task-32107), minus the
receipt -- refreshes the depth state, and proceeds. Live: a fresh note lands on
Console's own Get started card with "notes evidence staged", and Library's
status reads "Use in Console complete — Linked to Local Default · staged in
Console."

**One message.** The hand-off's return type changed from `bool` to
`(blocker, linked_workspace)`; the blocker is a complete sentence with its
remedy, carried on `LibraryNotesOperationState.failure_line` and rendered
verbatim in place of the generic "{action} failed — …". Every toast on that
path is gone. `library_notes_canvas.py` no longer appends "Next: Review the
error, then keep editing." when the transfer status already carries a `Next:`
-- that generic line offered an error that was never on screen.

**Clearing.** `_reset_library_note_editor_state` alone was not enough: the
row-press path never runs it. `_begin_library_note_load` -- the real note
switch -- calls the same new `_clear_finished_library_notes_operation()`, and a
SAVED outcome drops a terminal `failed` operation. `_build_library_notes_state`
now refuses any non-navigator operation, so an editor-region failure can never
paint the list pane beside it.

**Diagnostics.** The hand-off's `logger.opt(exception=True)` became a
metadata-only `logger.warning(..., error_type=...)`; the new link helper logs
the same shape. Inventory re-pinned (10 -> 11 calls).

**Fix round 1 (review of task 4).**

- *The link is no longer written on a path that cannot succeed.* The
  `open_chat_with_handoff` seam is read before the link block, and the one
  remaining post-write failure (the seam raising) rolls its own insert back
  through `_unlink_open_note_from_workspace` -- nothing in the app removes an
  `item_type="note"` membership, so a failed hand-off must not strand one.
  Only a failed rollback names the link that stayed. Inventory re-pinned
  again (11 -> 12 calls) for the rollback's metadata-only warning.
- *The completion line stopped claiming a link it had not made:* "Linked to
  {workspace} · staged in Console" / "Already linked to {workspace} · staged
  in Console" / "Staged in Console", by branch. The no-note blocker reads
  "No note is open. Next: open a note, then try again." -- the shared "Can't
  use this note in Console — " lead-in moved into the blocker strings (same
  rendered text for every other branch) so that one branch stops refusing
  "this note" while reporting that none is open.
- *Note for a future reader (review Minor #2):* `context_handoff_enabled` is
  no longer load-bearing on this path. The gate is profile-wide
  (`display_state.py`: any unlinked source row closes it), so linking this
  note usually leaves it False and the hand-off proceeds regardless -- by the
  ruling's intent. It is read only to decide whether to attempt the link.
- Added pins: `test_a_hand_off_console_cannot_take_writes_no_workspace_link`,
  `test_a_hand_off_that_raises_rolls_back_the_link_it_wrote`,
  `test_the_status_says_already_linked_when_it_made_no_link`,
  `test_the_status_claims_no_link_when_the_gate_is_open`.

Modified: `UI/Library_Modules/library_notes_controller.py`,
`Library/library_notes_state.py`, `UI/Screens/library_screen.py`,
`Widgets/Library/library_notes_canvas.py`,
`Docs/security/production-diagnostic-inventory.json`,
`Tests/UI/test_library_notes_w4_console_handoff.py` (new),
`Tests/UI/test_library_shell.py`, `Docs/User_Guide/library/notes.md`,
`Docs/User_Guide/console.md`.

Live: 235x52 and 100x30 on a no-provider profile and a 12-note profile;
captures `wave4-caps/console-handoff/handoff-10`, `10b`, `14`, `14b`, `15`,
`15b`, `16`, `16b`, `17`.
<!-- SECTION:NOTES:END -->
