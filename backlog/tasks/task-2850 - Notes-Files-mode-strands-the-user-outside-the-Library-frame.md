---
id: TASK-2850
title: Notes Files mode strands the user outside the Library frame
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 01:10'
updated_date: '2026-08-07 02:27'
labels:
  - library
  - notes
  - ux
  - uat-2026-08-06
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 (LIB-01, dual-agent critique `.impeccable/critique/2026-08-07T01-01-42Z__tldw-chatbook-ui-screens-library-screen-py.md`, observed at dev `6ffa56516`).

Notes canvas → "Database | Files" strip → clicking "Files" replaces the ENTIRE Library screen —
rail, search, groups, canvas frame all vanish. What remains is "Choose a notes folder." top-left
and a "Choose folder…" button ~150 columns away top-right, over ~40 blank rows. Escape does
nothing; the only exits are the small "Database" text link or the folder picker. Reproduced 100%.

A first-time user reads this as the app breaking. It is total context loss on a surface whose
sibling states all keep the rail + canvas frame, and it violates the product principle that
recovery paths stay visible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Entering Notes ▸ Files mode keeps the Library rail and canvas frame visible
- [x] #2 The folder chooser renders as a normal canvas empty state: prompt text and its action button adjacent, not separated by blank columns
- [x] #3 Escape (or an equally advertised key/control) returns from Files mode to the Notes Database view
- [x] #4 Live TUI verification confirms the above at 170×50 and 100×30
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the reported bug at HEAD (read compose_content in library_screen.py + LibraryFileNotesWorkspace root-status CSS) before touching code.
2. Fix the owning seam in tldw_chatbook/UI/Screens/library_screen.py: compose_content() currently returns early with just the source-toggle strip + workspace as direct screen children when notes_source=="files", bypassing the rail/canvas shell_grid entirely. Remove the early return; add a files-mode branch inside canvas_host (same seam as the editor/sync/list notes branches) so the workspace renders INSIDE the existing rail+canvas frame like every other notes view.
3. Fix the empty-state adjacency in library_file_notes_workspace.py: the root-status Static uses width:1fr, pushing "Choose folder..." far from "Choose a notes folder." Add a -empty-root CSS class (width:auto) toggled in compose() and _update_root_surface() for the root-less state only.
4. Add a Files-mode Escape binding to LibraryScreen (second "escape" BINDINGS entry, since Textual supports multiple bindings per key resolved via check_action), gated via check_action to _file_notes_active() so it never fires/advertises outside Files mode -- mirrors the existing library_skill_back pattern. Refactor the Database button handler into a shared _return_to_library_database_notes() helper reused by both the button and the new action, and register/unregister the footer's "esc" hint on entry/exit of Files mode.
5. TDD: write failing tests first (rail/canvas frame persists, empty-state adjacency, check_action gating, escape delegates to the shared return path, production pilot at 170x50/100x30), watch RED (via git apply -R of the implementation diff), then implement and confirm GREEN.
6. Fix any pre-existing tests whose assertions baked in the old (buggy) full-screen behavior (found: one test asserted rail ABSENCE in Files mode; two others relied on Files mode getting the FULL screen width for narrow-layout/text-elision assertions -- converted those to _WorkspaceHarness, matching their siblings, since their real intent is the workspace's own logic, not Library-shell integration).
7. Live tmux verification at both 170x50 and 100x30 per the task's required recipe.
8. Update Docs/User_Guide's Library page verified-against stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: two independent defects, both in the Files-mode seam.

1) library_screen.py's compose_content() returned EARLY when
   _library_notes_source == "files", yielding only the Database|Files
   strip + the LibraryFileNotesWorkspace as direct screen children --
   the rail and the shell_grid frame (canvas_host) were never composed.
   Fix: removed the early return; added a new elif branch for Files mode
   INSIDE canvas_host, at the same seam as the existing editor/sync/list
   notes branches, so the workspace now shares the rail+canvas frame like
   every other Notes view. is_local_snapshot_canvas was also narrowed to
   exclude Files mode (disk-backed, not DB-backed) so it can't flash the
   DB "Loading local Library sources..." copy.

2) LibraryFileNotesWorkspace's #file-notes-root-status Static used
   width:1fr in its own DEFAULT_CSS, which is correct for the
   linked/offline states (a status bar with Details/Change pinned right)
   but pushed "Choose folder..." far from "Choose a notes folder." in the
   root-less empty state. Fix: added a `-empty-root` class (width:auto),
   toggled in both compose() (first paint) and _update_root_surface()
   (state changes), scoped to the root-less state only.

Escape: added a SECOND "escape" BINDINGS entry on LibraryScreen
(Textual supports multiple bindings per key, resolved via check_action in
list order -- confirmed by reading textual/binding.py and app.py's
_check_bindings/run_action). Gated to _file_notes_active() in
check_action, mirroring the existing library_skill_back pattern exactly.
The Database button handler was refactored into a shared
_return_to_library_database_notes() so the button and the new
action_library_notes_files_back both go through the identical
flush/leave-guard sequence -- one seam, not a parallel path. The footer's
"esc back to Database" hint is registered/cleared on the same
transitions (LIBRARY_NOTES_FILES_SHORTCUTS), so Escape is genuinely
advertised only while it works (avoids worsening task-2858's LIB-09).

Test fallout from the architecture fix (all pre-existing, all in
Tests/UI/test_library_file_notes_git.py): one test asserted `not
screen.query("#library-rail")` while in Files mode -- literally encoding
the bug as expected behavior; fixed the assertion. Two others (one at
40x20, one at 120x40) routed through the full production LibraryScreen
and asserted literal, un-elided status text that only fit because Files
mode got the FULL screen width before this fix; every sibling
trust/retrust/narrow-layout test in that file already uses
_WorkspaceHarness (workspace mounted alone, its own declared width) for
exactly this reason -- converted both outliers to match, which also
dropped now-dead imports (_build_test_app, LibraryScreen,
LIBRARY_ROW_BROWSE_NOTES) from that file.

Files changed:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
- Tests/UI/test_library_file_notes_workspace.py (+4 new tests)
- Tests/UI/test_screen_navigation.py (+2 new tests)
- Tests/UI/test_library_file_notes_git.py (1 assertion fixed, 2 tests
  converted to _WorkspaceHarness, dead imports removed)
- Docs/User_Guide/library.md, Docs/User_Guide/library/file-notes.md
  (prose + Verified against stamps)

Concern noted, not fixed (out of scope): test_library_shell.py::
test_landing_footer_advertises_the_landing_keyboard_story fails
identically on unmodified HEAD (confirmed via git apply -R of this
task's diff) -- a pre-existing flake unrelated to this change.
<!-- SECTION:NOTES:END -->
