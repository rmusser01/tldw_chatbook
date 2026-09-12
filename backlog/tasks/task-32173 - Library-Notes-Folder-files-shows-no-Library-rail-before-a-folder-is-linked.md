---
id: TASK-32173
title: 'Library Notes: Folder files shows no Library rail before a folder is linked'
status: Done
assignee: []
created_date: '2026-09-09 09:11'
updated_date: '2026-09-09 17:47'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - file-notes
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the task review of task-32136. task-32136 AC#1 shipped the Library rail and
the source strip staying visible inside Folder files once a folder is
linked, but only partially: `#file-notes-body` is hidden entirely while
`_root is None`, so the pre-link empty state still drops to a full-width
layout with only the source strip surviving — the same gap the file-notes
guide's own pre-link sentence already documents as a known limitation
rather than a fixed behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At wide sizes, the Library rail is visible in the empty (no root
  linked) state, not just after a folder is linked
- [x] #2 Compact-terminal behavior is unchanged
- [x] #3 The file-notes guide's pre-link sentence is updated to match
- [x] #4 The behavior is pinned in a test
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Probe the pre-link Folder files layout at 235x52 and 60x24 (done: the rail is hidden because `#file-notes-body` is display-gated on a linked root, which leaves the reader shell zero width so its adaptive layout never resolves at all).
2. Failing test: the rail is visible pre-link at 235x52; a sibling test pins that nothing inside the body paints at 60x24.
3. Stop hiding the body; hide only the panes that need a linked folder (items, work, both grips) and let the existing resolver decide the rail -- it already closes it below the Library compact breakpoint.
4. Docs: file-notes.md pre-link sentence; re-tick task-32136 AC#1 now the qualifier is gone.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Folder files keeps its Library rail from the first frame; only the file panes wait for a folder.

Root cause was not the rail. `_update_root_surface` set `body.display = self._root is not None`, and hiding `#file-notes-body` left `LibraryAdaptiveReaderShell` with zero width -- so `_sync_library_file_notes_reader_layout_from_shell` returned early on `width <= 0` and the adaptive layout stayed degenerate (`library_open=False, library_width=0` even at 235x52). Un-hiding the rail alone could never have worked; the shell had to be measurable first.

- The `body.display` gate is deleted (three assignments). New `_sync_body_panes()` gates `shell.items` (AND the resolved `items_open`), `shell.items_grip`, `shell.library_grip` and `shell.work` on a linked root, and deliberately leaves `shell.library` to the resolver. Called from `_update_root_surface` and from `sync_reader_layout` after `shell.sync_layout` (which restores `items.display` from a layout that knows nothing about linking).
- Compact is unchanged for free: the resolver already reports `library_open=False` at 60 and 100 columns and True from 120 -- the same LIBRARY_NOTES_COMPACT_BREAKPOINT the screen uses -- so the unlinked body holds nothing to paint below it. Measured, not assumed. `#file-notes-body.-no-root { min-height: 0 }` keeps the now-visible empty body from claiming eight rows from the empty-state copy on a short terminal.
- task-32136 AC#1 re-ticked `[x]`; its `[~]` qualifier and its recorded rider are both closed here.

Tests: Tests/UI/test_library_notes_riders_r_file_notes.py -- `test_folder_files_keeps_the_rail_before_a_folder_is_linked` (235x52, rail painted, items/work/items_grip hidden) and `test_compact_folder_files_paints_no_shell_before_linking` (60x24, no shell pane displayed, empty state intact). Both RED first.

Live (fresh + power profiles, socket nw2-r-file-notes): caps 01, 06, 07, 08 -- pre-link rail at 235x52 on both profiles, full-width empty state at 60x24, and the panes returning when Use file_notes links.

Files: tldw_chatbook/Widgets/Library/library_file_notes_workspace.py, Tests/UI/test_library_notes_riders_r_file_notes.py, Docs/User_Guide/library/file-notes.md, Docs/User_Guide/library/notes.md, backlog/tasks/task-32136*.md.
<!-- SECTION:NOTES:END -->
