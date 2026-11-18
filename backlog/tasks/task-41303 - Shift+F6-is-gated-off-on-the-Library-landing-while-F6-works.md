---
id: TASK-41303
title: "Shift+F6 is gated off on the Library landing while F6 works"
status: Done
assignee:
- '@claude'
created_date: 2026-11-17 00:06
updated_date: 2026-11-17 01:13
labels:
- follow-up
- library
- keyboard
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/UI/test_screen_navigation.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41127 (round 15 of the 2026-10-23 burn-down) and its review r15lt-rev1.

**Why.** On the Library landing (no rail row selected), F6 cycles the workbench panes forward. It is the app-global binding, which delegates to `LibraryScreen.action_focus_next_workbench_pane` over the landing's rail and hub targets. Shift+F6 does nothing there: `LibraryScreen.check_action` returns `bool(self._library_selected_row_id)` for `focus_previous_workbench_pane`, so the screen's own priority binding is inactive, and nothing else binds the key. The same action already walks the same targets backwards on every other Library surface.

**Origin.** The gate did not come from a product decision. It arrived in the merge resolution a62d92f590 ("Merge latest dev and resolve Console integration regressions", 2026-08-27, landed through PR #2050), after `test_library_screen_bindings_are_all_gated_or_universal` had declared Shift+F6 universal (04e29673a2). TASK-34000.28 fix round 1 (4c7d47e45f) narrowed the branch back to Shift+F6 alone without discussing it. Owner ruling via the round-15 coordinator (2026-11-17): restore symmetry, so Shift+F6 works on the landing exactly as F6 does.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 On the bare Library landing, Shift+F6 is active and moves focus to the previous workbench pane over the same targets F6 cycles forward
- [x] #2 The bindings audit (`test_library_screen_bindings_are_all_gated_or_universal`) passes with Shift+F6 declared universal, red before the fix
<!-- AC:END -->

## Implementation Plan

1. Show the bindings audit red on the Shift+F6 assertion alone (after TASK-41127's re-pin of the ctrl+n landing gate).
2. Drop the `focus_previous_workbench_pane` landing gate from `LibraryScreen.check_action` (minimal diff; the action already resolves the landing's targets).
3. Run the audit, the F1/help and Library keyboard covering tests, ruff, preflight.

## Implementation Notes

Shift+F6 now works on the Library landing the same way F6 does.

**Change.** Removed the two-line `focus_previous_workbench_pane` branch from `LibraryScreen.check_action`. The action now falls through to `return True`, and `action_focus_previous_workbench_pane` walks `_library_workbench_focus_targets()` backwards: on the landing, the rail handle or search box, the emergency return and the hub. F1 on the landing now lists "Previous pane" (shift+f6), as on every other Library surface.

**Tests** (`Tests/UI/test_screen_navigation.py`):
- New `test_shift_f6_walks_the_landing_panes_back_the_way_f6_came`. It mounts the landing, takes two F6 steps through the screen action, then presses the real `shift+f6` key and expects focus back on the first stop. A real key press goes through `check_action`, which is where the gate was.
- The bindings audit now reports every mismatch at once, so one stale gate can no longer hide another. Within TASK-41127 it re-pins ctrl+n as landing-active: d00812ac44 made `library_notes_new` pass on the landing on purpose.

**Red, then green.**
- Red, on the gated source: the new test failed with focus left on `#library-hub-action-import` rather than `#library-search-input`. The audit failed on Shift+F6 alone.
- Green: both pass on the fixed source in 56.6 s at load 72.

**Covering runs.** The Library keyboard and help files, run with a throwaway `bootstrap_profile` stamp plugin (deleted afterwards) at `-n 4`. The files: `test_library_honesty_accessibility.py`, `test_library_modal_dismissal.py`, `test_library_ingest_keyboard.py`, `test_library_artifacts_focus.py`, `test_library_notes_tree_keys_extended.py`, `test_workbench_pane_focus.py`, and three Library-shell F6 and bindings nodes.
- Result: 7 failed, 266 passed.
- 6 of the failures fail the same way on the base `library_screen.py`: a missing `DestinationHarness` import, a CSS contrast pin, a toolbar row count, a footer fake without `has_class`, and the Personas F6 ring.
- The 7th is `test_workbench_focus_lands_on_a_scrolled_out_preferred_target_and_reveals_it`, which uses a non-Library app. It passed 3 of 3 alone on the fixed tree.
- `Tests/lint/ruff_gate.py` passes.

**User Guide.** The key table in `Docs/User_Guide/index.md` said Shift+F6 is bound only on Console and Roleplay. It now names Library (landing included) and Settings too.

**Files.**
- `tldw_chatbook/UI/Screens/library_screen.py` (2 lines removed)
- `Tests/UI/test_screen_navigation.py`
- `Docs/User_Guide/index.md`

