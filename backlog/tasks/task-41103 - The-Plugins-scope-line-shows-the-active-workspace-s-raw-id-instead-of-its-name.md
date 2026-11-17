---
id: TASK-41103
title: "The Plugins scope line shows the active workspace's raw id instead of its name"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-15 18:34'
created_date: 2026-11-15 02:20
labels:
- follow-up
- plugins
- ui
priority: low
references:
- tldw_chatbook/UI/Screens/plugins_screen.py
- tldw_chatbook/Workspaces/display_state.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** "Activation in: 6f1c…" tells the user nothing about which workspace a plugin will be active in; every other surface names the workspace.

**Evidence.** `plugins_screen.py:224-233` (`_workspace`) returns `active.workspace_id`, and `:295-299` paints `"Activation in: " + display_text(workspace or "Global default")`. The display-name helper the Library uses is `Workspaces/display_state.library_workspace_display_name` (imported by `Workspaces/link_on_use.py:34`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Plugins scope line names the active workspace by its display name, and 'Global default' when none is active
- [x] #2 The id still drives the token and staleness checks, so a workspace rename does not invalidate a selection
- [x] #3 A mounted test pins the named line, red before the change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the active workspace record once in the load; keep its id for the browse state, the token and `_current`, and paint the scope line with `library_workspace_display_name` (or "Global default" when none is active).
2. Keep the scope Static's `markup=False` seam (the CLAUDE.md markup rule for a markup-free surface: literal render, no escaping), and pin a markup-shaped name rendering literally.
3. Extend the mounted workspace-change test: the line names the workspace, a rename repaints it while the id, the token and the selection stay; show it red on the old code.
4. Update the User Guide Plugins page (scope line text; drop the "shows the ID" known issue).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Summary.** The Plugins scope line now reads "Activation in: <workspace name> | Installation: this device", using the Library's `library_workspace_display_name` ("Local Default" for the built-in workspace), and "Global default" when no workspace is active. The id still scopes everything else.

**Approach.** `PluginsScreen._workspace()` now returns the active record (or None) instead of its id. `_load` keeps the id in `browse.workspace_id`, so the token (`_token()[1]`) and `_current`'s staleness comparison are unchanged, and paints the name. `_current` compares the record's id. `_workspace` stays the single registry read, so the held-workspace boundary test still gates the same call.

**Markup.** The scope `Static` is `markup=False`, which renders text literally, the same convention as `notify(..., markup=False)` in CLAUDE.md. `escape_markup` would be wrong here: on a markup-free widget it shows a literal backslash. The test pins a markup-shaped name, `[b]Team[/b] B`, painting exactly as typed.

**Tests** (`Tests/UI/test_plugins_browse.py`, both already in the UI PR lane as node ids; no census change):
- `test_changed_workspace_refuses_pending_readme_and_refreshes_scope` now checks that the line reads "Activation in: B". It then renames the workspace to `[b]Team[/b] B` and checks four things: the line repaints with that name literally, `browse.workspace_id == _token()[1] == "workspace-b"`, `_current(token)` holds, and the selection and its details survive. After `clear_active_workspace()`, the line reads "Global default".
- `test_keyboard_details_back_resize_and_restore_keep_stable_selection` pins "Activation in: A" (it pinned the raw "workspace-a" before).
- `test_navigation_during_child_prune_settles_original_load[workspace-success]` now reads the gated record's `.workspace_id`.
- Red before the change: 3 of 3 nodes failed on the named-line assertion (`'Activation in: A | …' in paint`, and the same for B) at load 42-45. After: the whole file passed, 20 of 20 at `-n 4`, load 45-57 (one run). The seam node then failed once on `gate.result == 'workspace-a'`, which the record change made stale; after updating that assertion, 5 of 5 targeted nodes passed at load 53-57.

**Docs.** `Docs/User_Guide/plugins.md`: the scope-line bullet says the line shows the workspace by name ("Local Default" for the built-in one) and that a rename keeps the selection. The "scope line shows the workspace's ID" known issue is removed.

**Review r15pb-rev1.** The page's screenshot `Docs/User_Guide/images/plugins/overview.svg` was regenerated with `scripts/capture_user_guide.py plugins` (through run.sh, on a fresh profile). Its scope line now reads "Activation in: Local Default", where it read "workspace-default"; no other painted line changed.

**Files.** `tldw_chatbook/UI/Screens/plugins_screen.py`, `Tests/UI/test_plugins_browse.py`, `Docs/User_Guide/plugins.md`, `Docs/User_Guide/images/plugins/overview.svg`.
<!-- SECTION:NOTES:END -->
