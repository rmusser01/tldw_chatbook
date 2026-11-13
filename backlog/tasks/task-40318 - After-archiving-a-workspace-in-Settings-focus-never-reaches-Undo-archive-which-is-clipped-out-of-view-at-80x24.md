---
id: TASK-40318
title: "After archiving a workspace in Settings, focus never reaches Undo archive, which is clipped out of view at 80x24"
status: Done
assignee:
- '@claude'
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 03:16
labels:
- follow-up
- settings
- workspaces
- focus
priority: medium
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- Tests/UI/test_settings_workspace_lifecycle_journeys.py
dependencies:
- TASK-39169
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (cprod stream, TASK-39169 review r13p-rev1). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**What the user sees.** After confirming an archive in Settings > Workspaces, the "Undo archive" button is the next action, but focus does not land on it, and at 80x24 it is clipped out of its pane. At base ae9f9e8d21 focus fell to the Domain Defaults toggle; after TASK-39169's M2 fix a parked rebuild that lost its control lands on the category's rail row (`settings_screen.py:4857-4859`). Whether the real flow through the ConfirmationDialog parks is not verified; either way focus never reaches Undo (`:21880-21884`).

**Why.** The replacement map (`settings_screen.py:4954-4958`) has no `settings-workspace-archive` entry, though its own comment says to "keep keyboard context on the next action in that same card". The existing `reveal()` would focus and scroll Undo into view once mapped.

**Node.** `Tests/UI/test_settings_workspace_lifecycle_journeys.py::test_workspace_lifecycle_feedback_is_visible[archive]` (`:23`): red 3 of 3 at base and at the cprod head; `settings-workspace-archive-undo` at `Region(x=39, y=21, width=16, height=1)` is not in the compositor's visible set. The file runs on no lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a confirmed archive through the real dialog, focus is on Undo archive and it is painted inside its pane at 80x24; a test is red before the fix
- [x] #2 test_workspace_lifecycle_feedback_is_visible passes for every case, and its file runs on a PR lane
<!-- AC:END -->

## Implementation Plan

1. Re-run the lifecycle journeys file on base 26b2e677dd; read where focus lands after a confirmed archive and where Undo is painted.
2. Trace the archive completion: it refreshes the panes inside the ConfirmationDialog's callback, while the dialog still owns focus, so the swap captures no focus to restore.
3. After a current archive completes, queue a focus on Undo archive behind the pane swap (the `_after_category_panes(self._restore_category_pane_focus, ...)` pattern the folder-remove handler uses), so it is focused and revealed whichever order the swap and the dialog's dismissal run in. Cover the real app's eager task factory as well as `run_test`'s.
4. Show the archive node red before, green after; make the whole file pass and append it to the UI census tail with its local serial time.
5. Update the Settings page of the User Guide where it describes archive/Undo.

## Implementation Notes

**Cause.** `ConfirmationDialog` awaits `confirm_callback()` before it dismisses itself, and the archive commits and refreshes the Workspaces panes inside that callback. The pane swap reads `app.focused`, which is still the dialog's button, so `_focused_id_in_category_panes` returns None: nothing is parked or restored, and Textual's own reset moves the covered screen's focus off the removed Archive button. At base it landed on the Domain Defaults group toggle. The replacement map the description pointed at is never consulted on this path: with only a `"settings-workspace-archive": "settings-workspace-archive-undo"` entry, the archive case stayed red under `run_test` and under the eager task factory.

**Fix.** The archive's receipt presentation moved from `SettingsScreen.handle_workspace_archive` into `show_receipt()` in `UI/Settings_Modules/workspace_archive.py`, which already holds Undo. It clears the selection, sets the "Archived <name>…" result, refreshes the panes, and queues `_restore_category_pane_focus("settings-workspace-archive-undo")` behind the swap (`_after_category_panes`, the pattern folder Remove uses). If the swap runs under the dialog, `Screen.set_focus` sets the covered screen's focus to Undo, and the resume's own same-category rebuild restores and reveals it. If it runs after the dismissal, the queued call focuses and reveals it directly. The completion still runs only while the archive's request is current, so leaving the screen never steals focus. `settings_screen.py` shrinks by 3 lines; its row in `Tests/Architecture/test_module_size_ratchet.py` is lowered to 35,020.

**Tests.**
- `test_workspace_lifecycle_feedback_is_visible[archive]` now also asserts `host.focused is` Undo archive before its painted check. Red at base 26b2e677dd: focus was `settings-category-group-domain-defaults`; before the assertion was added it failed on Undo's `Region(x=39, y=21, width=16, height=1)` outside the compositor's visible set (three base runs). Green with the fix, serially and at `-n 4`.
- `Tests/UI/test_workspace_archive_async_completion.py`: the Settings stand-in gains `_after_category_panes`/`_restore_category_pane_focus`, and the cancellation test pins that Undo is queued only while Settings is current.
- All five feedback cases pass at `-n 4`; the whole file, 13 passed serially with the lane's flags.

**`test_partial_create_keeps_the_committed_workspace_clear` (3 cases) fails locally at `-n 4` only, from the path length, not code.** The retry error prints the folder twice, and under the scratch wrapper's basetemp plus the private-profile nesting at `-n 4` that `tmp_path` is about 245 characters, so the error is 11 rows and runs one row past the modal's clip at 80x24. A temporary edit pointing the folder at a 136-character path (CI's is about 131) passed all three at `-n 4`, and the serial lane command passed them under the wrapper as well. Proposed as a follow-up, not changed here.

**Lane.** The file's 13 cases are appended to `scripts/ui_pr_gate_census.txt`, one entry each (every case is over the per-entry budget), and `MINIMUM_FILES` is the measured count. Local serial times (`-p no:xdist --timeout=180 --durations=0`, load 55-67 on 18 CPUs, 623.8 s in all): feedback rename_error 52.5, rename_success 49.2, restore_error 43.4, archive 46.0, create_error 41.1; partial_create retry 62.2, keep 61.4, escape 60.5; completion activate 40.0, restore 38.7, literal_archive 39.0; theme textual-dark 34.1, textual-light 34.0.

**Docs and lessons.** `Docs/User_Guide/settings.md` (the Archive row says where the receipt appears and that focus moves to Undo archive). Lessons: `lessons-textual.md` (a pane rebuilt inside a modal's confirm callback finds the modal's focus) and `lessons-testing-evidence.md` (a temp path shown in the UI makes a geometry test depend on basetemp length).

Files: `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/UI/Settings_Modules/workspace_archive.py`, `Tests/Architecture/test_module_size_ratchet.py`, `Tests/UI/test_settings_workspace_lifecycle_journeys.py`, `Tests/UI/test_workspace_archive_async_completion.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `Docs/User_Guide/settings.md`, `backlog/docs/lessons-textual.md`, `backlog/docs/lessons-testing-evidence.md`.
