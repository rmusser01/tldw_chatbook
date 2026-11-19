---
id: TASK-41648
title: "The Console workspace tree raises KeyError for conversation-attention-running when mounted during a running run"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- console
- ui
priority: medium
references:
- tldw_chatbook/Widgets/Console/console_workspace_tree.py
dependencies:
- TASK-41313
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** In the burn-down's UI runs, mounting the Console while a chat held a running or retained run logged `KeyError: "No 'conversation-attention-running' key in COMPONENT_CLASSES"`. It is raised from `get_component_styles` in the workspace tree's projection sync. The key is declared (`Widgets/Console/console_workspace_tree.py:280`), so the likely cause is a style lookup that runs before styles are applied. A mid-run rebuild in the real app could hit it too.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Mounting the Console with a running chat does not raise or log this KeyError
- [ ] #2 A test mounts with a running chat and asserts that the running attention style applies once styles are ready
<!-- AC:END -->
