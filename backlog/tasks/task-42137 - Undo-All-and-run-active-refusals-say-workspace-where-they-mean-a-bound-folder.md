---
id: TASK-42137
title: "Undo All and run-active refusals say workspace where they mean a bound folder"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:45
labels:
- follow-up
- console
- change-review
- copy
priority: low
references:
- tldw_chatbook/UI/Console_Modules/review_selection.py
- tldw_chatbook/UI/Screens/change_review_screen.py
- Tests/UI/test_change_review_screen.py
- Docs/User_Guide/console/agent-runs-and-tools.md
- tldw_chatbook/Workspaces/change_revert.py
- tldw_chatbook/Workspaces/change_revert_ownership.py
- tldw_chatbook/Workspaces/git_workspace.py
- Tests/UI/test_change_review_commit_ui.py
- Tests/UI/test_change_review_push_ui.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review of the docs stream's undo section (TASK-42003, round 18). Undo All's refusals name a change snapshot's `root` folder as a "workspace":
- `tldw_chatbook/UI/Console_Modules/review_selection.py:395`: "Undo All is unavailable because a change snapshot has no workspace.";
- `review_selection.py:400`: "This turn has multiple change windows for the same workspace." (the same text at `UI/Screens/change_review_screen.py:167`, pinned by `Tests/UI/test_change_review_screen.py:1323`);
- `review_selection.py:406` and `:466`: "A run is active on this workspace — finish or stop the run first."

The same run-active refusal appears on the Review screen's other paths, so after a change to the Console notices alone the Review screen would say "folder" in one place and "workspace" in another:
- `tldw_chatbook/Workspaces/change_revert.py:185`, `Workspaces/change_revert_ownership.py:382` and `Workspaces/git_workspace.py:924`: "a run is active on this workspace — finish or stop the run first";
- `UI/Screens/change_review_screen.py:100`, `COMMIT_RUN_ACTIVE_REFUSAL`. Tests use it by name (`Tests/UI/test_change_review_commit_ui.py:507`, `Tests/UI/test_change_review_push_ui.py:467` and `:474`), so they follow a wording change. (`test_change_review_commit_ui.py:853` asserts its own fake's text, not the product's.)

The checks are per root folder (`row.get("root")`), and a workspace can bind several folders. The undo section TASK-42003 restored (round-18 docs branch) says "folder".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each Undo All refusal that refers to a change snapshot's root names it as a folder, consistent with the User Guide's undo section
- [ ] #2 The run-active refusals in `change_revert.py`, `change_revert_ownership.py`, `git_workspace.py` and `COMMIT_RUN_ACTIVE_REFUSAL` use the same wording, so Console and the Review screen name the same thing the same way
- [ ] #3 Tests that pin the old wording are updated, and a test pins each new notice
- [ ] #4 Where `Docs/User_Guide/console/agent-runs-and-tools.md` quotes these notices, it quotes them as they now read, including the commit refusal quoted verbatim at lines 1229-1230
<!-- AC:END -->
