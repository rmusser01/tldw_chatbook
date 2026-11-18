---
id: TASK-38982
title: "File Notes narrow navigator: ‹ Files leaves no focus, and below 60 columns Session Git and background refresh failures are out of reach"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:42
updated_date: 2026-11-06 04:19
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
- [x] #1 After ‹ Files at every width from 40 to 79, focus is on a visible navigator control, never None; red before: the dev probe (None at 6 of 6 widths)
- [x] #2 Below 60 columns with no file open, Session Git is reachable from the navigator by keyboard without opening a file, and the User Guide describes the route
- [x] #3 A failed background reconcile while the work pane is hidden (50x24) is shown without moving the view, on the authority line or the folder row, and clears on the next healthy pass
- [x] #4 Tests pin #1-#3 and are red before the change
<!-- AC:END -->

## Implementation Plan

1. Tests first, on the real Library screen: ‹ Files at every width 40-79 lands focus on a visible navigator control; at 50x24 with no file open, a keyboard route from the navigator opens Session Git; at 50x24 a failed reconcile shows on the authority line without moving the view and clears on the next healthy pass. Show each red on base.
2. ‹ Files: hand focus to the files tree (or search results) synchronously before the view switch hides Back.
3. A navigator **Manage** control, shown only while the narrow navigator hides the work pane, that opens Manage (and with it Session Git) in the narrow editor view, focusing the work pane's Manage first so its own hide never drops focus.
4. Refresh failures: keep the failure on the workspace, show "Refresh failed" in the authority line, and clear it (and its stale action-status line) on the next healthy reconcile or scan.
5. User Guide, census tail, size ratchet, covering runs.

## Implementation Notes

- **AC#1, ‹ Files.** `_back_to_navigator` parks focus on the Files grip synchronously, before the view switch hides Back, and then hands it to the files tree, or the search results while a search is active, through `_focus_once_shown`. A direct `set_focus(tree)` does nothing there: on the Library screen the Files pane is still closed, and the shell disables a closed pane until the screen re-resolves it a message later.
- **AC#2, Session Git.** The navigator header gains **Manage** (`#file-notes-navigator-manage`), shown only while the narrow navigator hides the work pane, so from 40 to 59 columns (`_sync_body_panes`). It focuses the work pane's Manage and then runs the same handler, which opens Manage in the narrow editor view. Session Git's **Review session changes (N)** is there, with or without a file open. The Button is kept as a reference because `_sync_body_panes` can run before the workspace is in the DOM.
- **AC#3, refresh failures.** A failed `refresh_files` records `_refresh_failure`. The authority line, above the shell, then reads "· Refresh failed" without moving the view. The next healthy reconcile (`_apply_reconcile`) or scan (`_adopt_scan_state`) clears it, along with the stale "Refresh failed: …" action-status line. An offline folder still outranks it ("Folder unavailable").
- **Tests (AC#4)**, all on the real Library screen:
  - `test_file_notes_back_leaves_focus_on_a_visible_navigator_control[40|50|60|70]` sweeps every width 40-79, ten per case. At each width it opens a file, presses Enter on ‹ Files, and requires focus inside the Files pane with visible cells. It was 38.7 s as one case, so it is split to keep each census entry short.
  - `test_file_notes_session_git_is_reachable_from_the_narrow_navigator` (50x24, no file open): Tab to a visible Manage, Enter, Tab to Review session changes, Enter; the Session Git panel paints.
  - `test_file_notes_failed_refresh_shows_beside_the_hidden_work_pane` (50x24): a raising `reconcile` paints "Refresh failed" on the authority line while the navigator still hides the work pane, and a healthy pass clears it.
- **Red on the base source (HEAD's workspace file, with the tests).** All 6 fail: ‹ Files "left focus on None" at 40, 50, 60 and 70 (the first width of each case); "Tab never reached a visible Manage"; "the failed reconcile was not shown beside the navigator". Green: 17-23 s for each decade case and 4-9 s for the other two, at `-n 4`, load 22 on 18 CPUs.
- **Covering runs** (`-n 4`, load 21-37).
  - `test_library_file_notes_workspace.py`: 197 passed, 6 failed. The 6 are TASK-32185's known reds, the same as on `fu-base-r10`.
  - The other File Notes files (the same set as TASK-38983): 665 passed, 55 failed, the identical 55 that fail on `fu-base-r10`.
- The module-size row goes 9835 -> 9873 under the owner's waiver. The six nodes are appended at the census tail, floor 1707. The User Guide covers ‹ Files' focus, the navigator's Manage, and "Refresh failed" on the authority line.
- Files: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`, `Tests/UI/test_library_file_notes_workspace.py`, `Tests/Architecture/test_module_size_ratchet.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `Docs/User_Guide/library/file-notes.md`.

