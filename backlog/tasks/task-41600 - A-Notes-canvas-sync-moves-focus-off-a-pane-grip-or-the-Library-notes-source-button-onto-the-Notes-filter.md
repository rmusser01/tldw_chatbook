---
id: TASK-41600
title: "A Notes canvas sync moves focus off a pane grip or the Library notes source button onto the Notes filter"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-18 18:15'
created_date: 2026-11-17 19:40
labels:
- follow-up
- library
- focus
- ui
priority: medium
references:
- tldw_chatbook/UI/Library_Modules/canvas_sync.py
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- tldw_chatbook/UI/Screens/library_screen.py
- Tests/UI/test_library_shell.py
dependencies:
- TASK-41129
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the libfocus stream's reviews of TASK-41129 (round 15: review r15lf-rev3 and fix round 4, 2026-11-17). Both cases also happen on base cc6006fcf5.
- **Tab during a waiting Database Notes entry.** Tab disarms the entry and moves focus to `#library-notes-source-database`. When the held rows arrive, the canvas sync moves focus to `#library-notes-filter`.
- **A click on a pane grip during the wait.** The grip keeps focus only until the rows arrive; the sync then moves it to the filter. TASK-41129's grip pin asserts only that the entry does not take focus back, for this reason.

**Mechanism** (read at 212f015a91):
- `canvas_sync.py:986-1008`: an in-surface Notes sync captures the focus identity when it starts, and its default follow-up calls `_restore_library_notes_after_targeted_sync`.
- `library_notes_controller.py:2305-2306`: that restore is skipped only while the list entry is armed. After a user's Tab or click has disarmed it, `_restore_library_notes_focus_identity` runs.
- `library_screen.py:5248`: the `direct_roles` table in `_library_notes_semantic_role` gives the source buttons and the pane grips no role. Their identity therefore resolves to no target, and `_library_notes_fallback_focus_target` picks the filter for the navigator region (`library_notes_controller.py:2093-2094`).
- `library_notes_controller.py:2202` then focuses the filter, although the user's control is still mounted. The source strip (`library_browse_route_swap.py:175`) and the grips sit outside `#library-notes-canvas` (`:196`), so the sync never replaced them.

**Related.** TASK-34000.89 is the same mechanism for the rail search box (stage `rail`, so it lands on the Notes rail row) and the Clear filter button. Its fix needs a design decision on a rail text field and Escape. This task covers the controls outside the canvas that are not text fields. TASK-41504 covers where code-driven entries land, not a sync moving the user's own focus.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Notes canvas sync leaves focus where it is when the focused control is not a text field and is still mounted outside the repainted canvas (a pane grip, `#library-notes-source-database` or `#library-notes-source-files`); the rail search box is out of scope (TASK-34000.89)
- [x] #2 A focused control that the sync replaced is still restored as today (its role, or the region's fallback)
- [x] #3 Pilot pins press Tab, and click a grip, during a waiting Notes entry, then release the rows and assert the focused widget id; red on the code before the fix
<!-- AC:END -->

## Implementation Plan

1. In `canvas_sync.py`, remember the widget focused when an in-surface Notes sync captures its own identity, and hand it to `_restore_library_notes_after_targeted_sync`.
2. That restore leaves focus alone when the same widget still holds focus after the repaint, sits in `screen.focus_chain`, is not a text field (TASK-34000.89 keeps the rail search box), and was focused by a click or Tab; a replaced control, or a grip Textual's reset parked focus on, still takes today's role or fallback restore.
3. Pilot pins: Tab, and a real grip click, during a waiting Notes entry; release the rows; assert the focused id. Red on the base.
4. Covering tests (canvas sync, Notes focus), ratchets, preflight.

## Implementation Notes

Reproduced on the base (`67583dfa72`): during a waiting Notes entry, Tab put focus on `#library-notes-source-database` and a click on `#library-browse-library-grip` kept it there, and in both cases the rows' canvas sync then moved focus to `#library-notes-filter`. The base was tested for both starts, a Media switch and Escape from Create, at 170x48.

- **Fix.** `_sync_library_canvas` now passes the widget focused when it captured its own Notes identity. `_restore_library_notes_after_targeted_sync` leaves focus alone when four things hold:
  - that same widget still holds focus after the repaint;
  - it is in `screen.focus_chain`;
  - it is not a text field (AC#1), which keeps the rail search box on today's restore (TASK-34000.89);
  - a click or Tab put it there. `LibraryScreen._library_input_focus` records the widget from `get_focusable_widget_at` and from `action_focus_next`/`_previous`, and clears it when focus moves anywhere else.

  A control the repaint replaced is no longer the focused widget, so its role or the region's fallback is restored as before (AC#2). An identity passed in by automatic reconciliation carries no pre-sync widget, so that path is unchanged.
- **Why the click-or-Tab condition.** Without it, the covering run failed `test_library_notes_w4_import_keyboard.py::test_tab_from_the_selection_pane_cycles_the_three_actions`. In Import once's picker flow, Textual's reset off a removed control parks focus on `#library-browse-items-grip` before the next sync starts. Today's restore heals that to `note-import-body`; the first cut kept the grip. A trace showed it: `RESTORE identity=''/import before=library-browse-items-grip`. With the condition the test passes.
- **Tests** (`Tests/UI/test_library_notes_move_never_false_conflicts.py`, PR-laned, no census edit). `test_a_notes_canvas_sync_keeps_focus_on_a_control_outside_the_canvas`:
  - `[tab]` presses Tab during a waiting entry, releases the rows and asserts `library-notes-source-database`.
  - `[grip-click]` does a real `pilot.click` on the Library grip and asserts `library-browse-library-grip`.
  - `[rail-search]` is the scope fence: a click into the rail search box still ends on `library-row-browse-notes`.
  - Each case asserts the focused id and `screen.focus_chain` membership.
  - `test_a_notes_canvas_sync_still_restores_a_grip_the_user_did_not_choose` sets focus on the grip directly, the way Textual's reset leaves it. It asserts the sync still moves focus to the navigator's fallback, the filter. It passes on the base.
  - AC#2 (a replaced control is still restored) stays pinned by `test_new_note_keeps_focus_after_a_targeted_notes_refresh` in the same file, which passed.
- **Red evidence.**
  - On the base source, `[tab]` and `[grip-click]` failed with focus on `library-notes-filter`, 2 runs each. `[rail-search]` passed there, as it should.
  - Without the text-field clause, only `[rail-search]` failed: focus stayed in `library-search-input`.
  - Without the click-or-Tab clause, the grip pin failed, with focus left on `library-browse-library-grip`. The import keyboard test failed too.
- **Runs.** Final source, the four nodes x 10 with 16 CPU burners at `-n 6`: all passed (load 14-43, in the 170-node run noted in TASK-41504). Covering runs and lane times are recorded in TASK-41504.
- **Files.** `UI/Library_Modules/canvas_sync.py`, `UI/Library_Modules/library_notes_controller.py`, `UI/Screens/library_screen.py` (`_library_input_focus`, the delegator), `Tests/UI/test_library_notes_move_never_false_conflicts.py`, `Docs/User_Guide/library/notes.md`, `backlog/docs/lessons-textual.md` (an incident added to the TASK-34000.28 entry).
- **Docs.** `notes.md`: a refresh of the Notes list leaves focus on a pane grip or the source strip.

**Fix round 1 (review r16fo-rev1, 2026-11-18).**
- **The restore reads the current focus, not the focus from when the sync began (Minor 3a).** A click or Tab that lands while the sync repaints is kept: the rule is now "focus is the control the user last clicked or Tabbed to, reachable, not a text field". The `focused_before` parameter is gone, so `canvas_sync.py` and the screen delegator are back to their base form.
- **The narrow emergency Tab is recorded (Minor 3b).** Below 64 columns the canvas-only stage handles Tab in `on_key`, and that branch now records `_library_input_focus` as well.
- **Tests** (`Tests/UI/test_library_notes_move_never_false_conflicts.py`):
  - `test_a_click_while_a_notes_sync_repaints_keeps_its_target` does the click's focus half inside the sync, before its restore, the way Textual's MouseDown forwarding does it (`get_focusable_widget_at`, then `set_focus`). Red on `b0cd65a9b8`: the restore put focus back on `library-notes-tree-note-1`.
  - `test_a_narrow_emergency_stage_tab_counts_as_the_users_focus` (60x24) sets the canvas-only stage as `test_emergency_tab_from_a_note_field_still_reaches_on_key` does, Tabs onto `#library-browse-items-grip`, then runs a sync. Red on `b0cd65a9b8` and without the new line: the sync moved focus to `library-notes-filter`.
  - The stage has to be set by hand: at 60x24 and 50x20 the Notes route never enters it (a probe found the stage `None`).

**Fix round 2 (review r16fo-rev2, 2026-11-18).** The emergency-Tab line from fix round 1 and its test (`test_a_narrow_emergency_stage_tab_counts_as_the_users_focus`) are removed.
- Notes never enters the canvas-only stage: `_apply_library_emergency_geometry` needs `_library_ordinary_route_active()`, which excludes Notes and Create Note. A probe had already shown that stage stays `None` at 60x24 and 50x20.
- `_library_input_focus` is read only by the Notes restore, so the line had no effect, and its test pinned a state no user can reach.
- `library_screen.py` is now 38312 lines (1380 methods), and its ratchet row says so.
