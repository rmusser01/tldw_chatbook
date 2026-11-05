---
id: TASK-38945
title: "Console rebuilds the workspace context on every sync because its build fingerprint cannot prove the inputs unchanged"
status: To Do
created_date: 2026-11-05 21:35
labels:
- follow-up
- console
- performance
priority: low
references:
- tldw_chatbook/UI/Console_Modules/workspace.py
- tldw_chatbook/Workspaces/display_state.py
- Tests/UI/test_console_projection_reads_off_loop.py
- Tests/UI/test_console_workspace_display_reads_off_loop.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05.

TASK-36511.2 (Done) brought the Console down to one workspace-context build per sync. It narrowed its AC#2 to that, and its notes leave a cross-sync cache "as a follow-up for the coordinator to file". Nobody filed it, so here it is.

- The build cache (`ConsoleTickWorkspaceBuilds`) lives only for one scope: `tick_workspace_build_scope` (`tldw_chatbook/UI/Console_Modules/workspace.py:6973`) and `live_workspace_build_scope` (`:7003`).
- The cache cannot simply outlive the scope. `_console_workspace_build_fingerprint` (`workspace.py:7510`) says in its docstring (`:7523`) that it is "Deliberately NOT exhaustive": stars, unseen markers and conversation-browser collapse preferences are not in it. The build reads them at `:4652` (`_console_conversation_browser_collapse_preferences`, defined at `:1705`) and through `fleet_unseen_ids_accessor` (`:697`).
- TASK-36511.2 measured each build at 3.2-3.6 ms of CPU per sync at load 15. Whether that adds up to enough to justify a cache is not known. AC#1 measures it first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Workspace-context builds per minute, their CPU, and how many had inputs identical to the previous build are measured for an idle Console and for one during a run; if the saving does not justify a cache, the notes record the numbers and the task closes as Won't Do
- [ ] #2 Consecutive syncs with unchanged inputs reuse one build, shown by a build_console_workspace_state call count over N syncs before and after the change
- [ ] #3 Changing any single build input between two syncs produces a fresh build: every input the fingerprint covers today, plus a star, an unseen marker and a group collapse preference; each of those last three is shown red with that input temporarily removed from the reuse key
- [ ] #4 test_a_control_bar_sync_builds_the_workspace_context_once and test_a_control_bar_sync_reads_a_moved_registry_live_during_a_run pass unchanged
<!-- AC:END -->
