---
id: TASK-38982
title: "File Notes narrow navigator: ‹ Files leaves no focus, and below 60 columns Session Git and background refresh failures are out of reach"
status: To Do
created_date: 2026-11-06 01:42
labels:
- follow-up
- library
- file-notes
- ux
- accessibility
priority: medium
references:
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
- Tests/UI/test_library_file_notes_workspace.py
- Docs/User_Guide/library/file-notes.md
dependencies:
- TASK-38757
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b and on the TASK-38757 branch (its post-merge state), 2026-11-05. All lines below are in that branch's `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`.

**Session Git has no entry from the narrow navigator.**
- Manage (`:1896`) and "Review session changes" (`:2003-2007`) live only in the work pane (`:2019-2021`).
- Below 60 columns the navigator hides that pane. See `_NARROW_WORK_MIN_WIDTH = 18` (`:344`), `sync_reader_layout` (`:2048-2087`) and `_navigator_hides_work` (`:2089-2096`).
- No navigator control or key leads there. The workspace binds only escape and ctrl+end/home (`:363-374`).
- TASK-38757's notes accept "below 60 columns, Manage and Session Git are reached by opening a file first" as a trade-off, but nothing tracks it.

**‹ Files leaves focus on nothing.**
- `_back_to_navigator` (`:7786-7790`) switches to the navigator. `_apply_responsive_layout` then hides the focused Back button (`back.display = False`, `:3736-3737`), and nothing moves focus elsewhere.
- A base probe at load 13.7 pressed Enter on Back: focus was None at all 6 widths tried (44, 50, 59, 60, 70, 79). So this is not limited to widths below 60.

**A background refresh failure is invisible while the work pane is hidden.**
- `refresh_files` (`:6804-6849`) reports a failure with `_set_action_status(f"Refresh failed: {error}")` (`:6837`). That writes only `#file-notes-action-status` (`:5799-5803`), a Static in the work pane (`:1914`).
- It never calls `_bring_work_forward`.
- A failed reconcile does not set `_root_offline`, so the authority line and the folder row stay unchanged. Both are composed above the shell (`:2555-2590`) and could carry the message without moving the view.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After ‹ Files at every width from 40 to 79, focus is on a visible navigator control, never None; red before: the dev probe (None at 6 of 6 widths)
- [ ] #2 Below 60 columns with no file open, Session Git is reachable from the navigator by keyboard without opening a file, and the User Guide describes the route
- [ ] #3 A failed background reconcile while the work pane is hidden (50x24) is shown without moving the view, on the authority line or the folder row, and clears on the next healthy pass
- [ ] #4 Tests pin #1-#3 and are red before the change
<!-- AC:END -->
