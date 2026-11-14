---
id: TASK-40328
title: "A Library Collections rail press recomposes the whole screen three times, about 0.64-0.80 s of UI-loop CPU"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- library
- collections
- performance
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_collections_controller.py
dependencies:
- TASK-39170
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (cred stream review r13c-rev2, measured on the Collections action node). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** One rail press costs 0.64-0.80 s of UI-thread CPU across three full Library rebuilds, each re-mounting the rail, canvas and reader: visible jank and focus churn on a common navigation step.

**Path** (`_select_library_rail_row_after_source_admission`): the route swap (`library_screen.py:23481`, `_replace_library_browse_canvas`) or a whole-screen `recompose()` (`:23484`), then `run_worker(self._load_library_collections_capture_entry())` (`:23504-23508`). That worker calls `_refresh_library_collections_capture_reader()` at `library_collections_controller.py:467`, again after `await controller.load_page` (`:503`), and again after `await controller.load_selected_now()` (`:506`). Each is `self.refresh(recompose=True)` (`:403-411`), and on this controller `refresh` is `Screen.refresh` (`:191-193`), so each is a whole-screen recompose. TASK-281 (the Library whole-screen-recompose umbrella) does not name this path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Collections rail press settles with at most one Library recompose; the capture loads update the reader pane in place
- [ ] #2 The press's UI-loop CPU is measured before and after on the same host and recorded
- [ ] #3 A test counts the recomposes of one press and is red before the fix
<!-- AC:END -->
