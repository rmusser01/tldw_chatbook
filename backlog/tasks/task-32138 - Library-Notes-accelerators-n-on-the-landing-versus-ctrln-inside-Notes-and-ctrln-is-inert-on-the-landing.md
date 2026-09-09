---
id: TASK-32138
title: >-
  Library Notes accelerators: n on the landing versus ctrl+n inside Notes, and
  ctrl+n is inert on the landing
status: Done
assignee: []
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 06:58'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - keyboard
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Parent verified live: the Library landing footer advertises 'n new note' and Ctrl+N does nothing there; the Notes canvas footer advertises 'ctrl+n new note'. `test_library_notes_bindings_are_inactive_outside_notes_workflow` pins the inactivity, so this is a pinned decision the critique disagrees with: the same action has two keys depending on the canvas. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One accelerator for New note is advertised and works on the landing and inside Notes, or both keys work in both places
- [x] #2 The pinning test records the decision
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Decision: BOTH keys work in BOTH places (the pinned test's rationale for gating library_notes_new to the notes workflow is sound for the OTHER 3 notes-only bindings, but New note already has a landing-safe action -- action_library_notes_new is exactly _select_library_rail_row(LIBRARY_ROW_CREATE_NOTE), the same call the bare-n landing handler already made). Widened check_action's library_notes_new branch with 'or not self._library_selected_row_id' (landing = no row selected), and rewired the bare-n on_key handler to gate through check_action('library_notes_new') instead of a landing-only literal check, so both keys now share one activation surface. Landing footer copy changed from 'n new note' to 'ctrl+n new note' (LIBRARY_LANDING_SHORTCUTS) to advertise one consistent story; bare n still fires (kept for the existing pin and muscle memory), it's just no longer the ADVERTISED key. Reconciled the pinned test (test_library_notes_bindings_are_inactive_outside_notes_workflow, Tests/UI/test_library_shell.py) by pulling library_notes_new out of the must-be-False loop and asserting it IS True on a fresh/landing screen, with a docstring explaining the task-32138 decision. Updated 3 other pinned footer-string tests to match the new landing copy (test_screen_footer_hints.py, test_app_footer_shortcut_context.py, test_library_shell.py). Files: tldw_chatbook/UI/Screens/library_screen.py, Tests/UI/test_library_shell.py, Tests/UI/test_screen_footer_hints.py, Tests/UI/test_app_footer_shortcut_context.py. Tests: Tests/UI/test_library_notes_wave_editor_keys.py (3 new tests). Live-verified at 235x52: landing footer reads 'ctrl+n new note', Ctrl+N from the landing opens Create.

Fix round 1: deferred, no action taken (reviewer-accepted, cheap-to-fix-later): Tests/UI/test_app_footer_shortcut_context.py:420's `test_footer_screen_supplied_f6_hint_survives_at_60_cols`-adjacent test constructs its own inline `shortcuts=(("/", "focus search"), ("i", "import content"), ("n", "new note"))` tuple to test generic footer TRUNCATION/rendering behaviour at 100 cols -- it does not read LIBRARY_LANDING_SHORTCUTS (that mirror, in the SAME file's `_library_landing_shortcuts_with_pane_cycle`, was already updated to ctrl+n in the original pass) and "n new note" there is arbitrary sample data, not an assertion about the real landing copy.
<!-- SECTION:NOTES:END -->
