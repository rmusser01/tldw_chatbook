---
id: TASK-41600
title: "A Notes canvas sync moves focus off a pane grip or the Library notes source button onto the Notes filter"
status: To Do
assignee: []
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
- [ ] #1 A Notes canvas sync leaves focus where it is when the focused control is not a text field and is still mounted outside the repainted canvas (a pane grip, `#library-notes-source-database` or `#library-notes-source-files`); the rail search box is out of scope (TASK-34000.89)
- [ ] #2 A focused control that the sync replaced is still restored as today (its role, or the region's fallback)
- [ ] #3 Pilot pins press Tab, and click a grip, during a waiting Notes entry, then release the rows and assert the focused widget id; red on the code before the fix
<!-- AC:END -->
