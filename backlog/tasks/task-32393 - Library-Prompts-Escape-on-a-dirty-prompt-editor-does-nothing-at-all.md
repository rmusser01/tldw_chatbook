---
id: TASK-32393
title: 'Library Prompts: Escape on a dirty prompt editor does nothing at all'
status: Done
assignee: []
created_date: '2026-09-11 10:30'
updated_date: '2026-09-11 17:19'
labels:
  - library
  - prompts
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With an unsaved edit in the Prompts editor, Escape neither leaves nor explains. Verified live at 235x52 on a seeded profile during the task-32366 reconciliation: with the editor showing "Modified 4h - v1 - Unsaved changes" and the footer advertising `esc back to list`, four Escape presses (from an Input and after tabbing out of one) produced no exit, no notice and no visible change. A dirty veto exists in code (`_notify_prompt_dirty_veto`, `LIBRARY_PROMPT_DIRTY_VETO_COPY`) but no toast reached the screen. A key the footer names must either do what it says or say why it will not; a silent no-op reads as the app having hung. The veto itself is correct and deliberate -- `_exit_library_prompt_editor_guarded` (`library_prompts_controller.py:3456-3459`) returns False on a dirty flush -- so the defect is only the missing message. task-2702 (Done) shipped `LIBRARY_PROMPT_DIRTY_VETO_COPY` for the nav-bar veto and task-32133 is the nearest precedent for wiring a refusal to its reason; the Skills twin (`library_screen.py:24802-24810`) already calls its notifier from the same seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Escape on a dirty prompt editor either leaves the editor or states why it cannot, on the same line as the next step
- [x] #2 The footer chip and what Escape does agree in every prompt-editor state
- [x] #3 The dirty-Escape path is covered by a test that fails if the key becomes a silent no-op again
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the veto seam: every _flush_library_prompt_save caller and which ones already notify.
2. Wire the notice at the shared exit seam (_exit_library_prompt_editor_guarded), covering Escape and Back.
3. Make the footer chip state-aware so it stops promising an exit the key refuses, and re-register it at the dirty flip (which deliberately does not recompose).
4. Red-first pins through the real editor widgets; mutation-test each guard.
5. Live-verify at 235x52 on a seeded profile; guide stamp; Implementation Notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The veto was already correct; it was silent. ``_exit_library_prompt_editor_
guarded`` -- the one seam both Escape and "‹ Back to list" route through --
now calls the notifier that task-2702 shipped for the nav-bar veto, in the
Skills twin's shape (``_exit_library_skill_editor_guarded``). The third exit,
``on_prompt_block_editor_back_requested``, claimed in its docstring to "use
the Library editor's existing dirty-aware back behavior" while
re-implementing it inline one copy behind; it now actually calls that seam,
which also gives it the list-entry focus every other back-to-list exit arms.

AC#2 needed more than the toast: the footer chip said "esc back to list" while
Escape refused to go there. The prompt editor's footer set now names the
blocker (``esc save or discard first``,
``LIBRARY_PROMPT_DIRTY_ESCAPE_CHIP``) while the editor is dirty -- the same
honest-chip rule task-31271/31272 applied at the other Library seams, and the
skill editor's own state-dependent set two branches below is the shape it
follows. The dirty flip deliberately does NOT recompose (that would remount
the editor's Input/TextArea fields and re-arm-race them), so the footer is
re-registered from ``_update_library_prompt_meta_static`` -- the one seam that
already repaints the dirty marker in place. That needed
``register_footer_shortcuts`` bound into ``LibraryPromptsController``, which
the ledger in ``test_library_prompts_wiring.py`` records (42 -> 43 bound
names).

Deliberately NOT widened to every ``_flush_library_prompt_save`` veto: the
prompt-row switch, select-mode entry and the entry-reconcile path are silent
too, but two of them are different symptoms and the third is a background
reconcile where a toast would be noise. Listed as a rider instead.

**Evidence.** Two pins drive the real editor widgets (open a seeded prompt,
type into ``#library-prompt-name``, press Escape) and assert the toast and the
REGISTERED footer set, not just the selector's return value. Each of the three
guards was mutation-tested: removing the notify, forcing the chip branch off,
and dropping the footer re-registration each turn a pin red. Live at 235x52 on
a seeded profile: "• Unsaved changes" in the meta line, footer reading "esc
save or discard first", Escape raising "Unsaved Prompt changes — Save or
Discard changes first." and staying in the editor.

**Files:** ``library_prompts_controller.py``, ``screen_constants.py``,
``library_screen.py``, ``Tests/UI/test_library_prompt_dirty_escape.py`` (new),
``Tests/Architecture/test_library_prompts_wiring.py``,
``Docs/User_Guide/library.md``, ``Docs/User_Guide/library/prompts.md``.
<!-- SECTION:NOTES:END -->
