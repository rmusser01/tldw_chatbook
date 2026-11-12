---
id: TASK-40318
title: "After archiving a workspace in Settings, focus never reaches Undo archive, which is clipped out of view at 80x24"
status: To Do
created_date: 2026-11-12 19:36
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
- [ ] #1 After a confirmed archive through the real dialog, focus is on Undo archive and it is painted inside its pane at 80x24; a test is red before the fix
- [ ] #2 test_workspace_lifecycle_feedback_is_visible passes for every case, and its file runs on a PR lane
<!-- AC:END -->
