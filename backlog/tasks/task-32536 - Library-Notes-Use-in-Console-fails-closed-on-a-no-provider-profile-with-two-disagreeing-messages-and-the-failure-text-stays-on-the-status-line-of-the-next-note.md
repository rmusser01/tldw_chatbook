---
id: TASK-32536
title: >-
  Library Notes: "Use in Console" fails closed on a no-provider profile with two
  disagreeing messages, and the failure text stays on the status line of the
  next note
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-13 06:45'
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
- [ ] #1 On a profile with no ready model, Use in Console links the open note to the active workspace, stages it, and lands on Console's own setup card; when the link cannot be made it shows one sentence naming the real blocker with the remedy inline (for example "restart Chatbook, then try again")
- [ ] #2 Exactly one message is shown for a failed hand-off (status line or toast, not both), and "Next: Review the error" is never offered when no error is on screen
- [ ] #3 Opening another note, or saving the current one, clears a previous action failure from the editor status line, and the list status line never carries it
- [ ] #4 A regression test performs a failed hand-off, opens a second note and saves it, and asserts the second note's status line reads Saved with no failure text
<!-- AC:END -->

## Implementation Plan

1. Reproduce live on the fresh profile at 235x52 and headlessly (captures handoff-00..03); write the PROVEN cause into the Description and reword AC#1 per the controller ruling (done before any code change).
2. RED pins in `Tests/UI/test_library_notes_w4_console_handoff.py`: auto-link + stage on a fresh profile with no toast and no "check Console readiness"; a registry-less hand-off shows exactly one line naming the blocker with its remedy and no "Review the error"; a failure clears on note change and on a successful save, and never reaches the list status.
3. Fix: `_link_open_note_to_active_workspace()` on the controller (the conversation precedent minus the receipt); `_open_selected_library_note_handoff` links when the gate is closed and returns the blocker sentence instead of toasting; `LibraryNotesOperationState.failure_line` carries the one-line failure copy; `_reset_library_note_editor_state` and a SAVED outcome drop a terminal failed operation; `_build_library_notes_state` only shows navigator-region operations; the canvas does not append its generic Next when the transfer status already names one; the hand-off log becomes metadata-only.
4. Diagnostic inventory `--write`, GREEN + dev-baseline FAILED-name comparison, live re-verify at 235x52 and 100x30, guide `notes.md` "Use a note in Console" + stamp.
