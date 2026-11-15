---
id: TASK-41103
title: "The Plugins scope line shows the active workspace's raw id instead of its name"
status: To Do
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
- [ ] #1 The Plugins scope line names the active workspace by its display name, and 'Global default' when none is active
- [ ] #2 The id still drives the token and staleness checks, so a workspace rename does not invalidate a selection
- [ ] #3 A mounted test pins the named line, red before the change
<!-- AC:END -->
