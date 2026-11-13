---
id: TASK-40717
title: "Startup workflow recovery runs with workflow runs off, and its failure stops sub-agent result delivery to ordinary chats"
status: To Do
created_date: 2026-11-13 06:45
labels:
- follow-up
- agents
- console
- bug
priority: medium
references:
- tldw_chatbook/Chat/console_runtime.py
- tldw_chatbook/Chat/console_fleet_wake.py
- tldw_chatbook/Agents/workflow_run_service.py
- https://github.com/rmusser01/tldw_chatbook/pull/3480
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-11-10 review of PR #3480 (TASK-39147, native phased workflows), which merged on 2026-11-12. It is still present on dev 7579f75ea9, re-checked on 2026-11-13 by reading the code. Line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** Workflow runs are off by default: `[agents] workflow_runs_enabled` in `Config_Files/default_config.toml:377`, read by `UI/Console_Modules/wiring.py:191-195`.

Even so, `Chat/console_runtime.py:4988` installs `_project_workflow_recovery` as the fleet wake's recovery projection without checking that flag. The projection (:5319-5327) does three things:
- builds the workflow service with `ensure_workflow_service()`, which raises "Native workflow stores are unavailable" at :5303;
- runs `project_recovery`, which issues SQL updates (`Agents/workflow_run_service.py:1844-1859`);
- runs `recovery_protections`.

**If the projection raises:**
- `Chat/console_fleet_wake.py:1494-1504` marks wake recovery not ready, with reason `history_unavailable`.
- `_attempt` returns early while it is not ready (:829).
- Recovery runs once per session (:453-454).

So for the rest of the session, finished sub-agent results are not delivered to ordinary chats. TASK-38971 fixed a different recovery failure, the missing runs database.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With `workflow_runs_enabled` off, startup recovery builds no workflow service and runs no workflow SQL or projection.
- [ ] #2 A failing workflow projection does not stop sub-agent result delivery to chats that have no workflow runs.
- [ ] #3 A test with workflow runs off and a raising workflow projection shows a sub-agent result still delivered to an ordinary chat.
<!-- AC:END -->
