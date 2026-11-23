---
id: TASK-42511
title: "The project-chat UAT fails all four variants locally on macOS with a generic file-tool worker failure"
status: To Do
assignee: []
created_date: 2026-11-23 01:58
updated_date: 2026-11-23 01:58
labels:
- follow-up
- tests
- ci
- agents
priority: medium
references:
- Tests/QA/test_project_chat_uat.py
- Tests/uat_project_chat_support.py
- tldw_chatbook/Agents/local_tool_provider.py
- tldw_chatbook/Tools/workspace_tool_executor.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-11-22 hosted-runner flake list (item 4) and filed in the 2026-11-23 backlog hygiene pass. Checked on dev 2f7790873a; paths start at the repository root.

**Why.** Run locally on macOS, `Tests/QA/test_project_chat_uat.py` fails in all four of its variants (`fenced`, `native`, `compact-recovery`, `wrong-family`; `Tests/uat_project_chat_support.py:22`), 4 of 4, on both dev ddd3c02052 and the branch of #3653. Each fails with `fs_list failed: ERROR: The local file-tool worker failed before returning a result.` That line is `LOCAL_WORKER_FAILED_REFUSAL` (`tldw_chatbook/Agents/local_tool_provider.py:215-217`). `_workspace_execution_error_reason` (`:696`) classes every executor code other than `root_pin_failed`, `invalid_request` and `tool_failure` as unavailable authority, and `_workspace_execution_error_result` (`:707-719`) answers all of them with that one line (`worker_crashed`, `worker_timed_out`, `spawn_failed`, `protocol_failure`, `containment_unavailable`, `cleanup_unproven` and the rest; TASK-33940.2). So the failure that actually happened is not shown. The record does not say whether the local run was inside an agent sandbox. The UAT is a step of the required "PR Fast Lane (admission-sensitive 2)" job (`.github/workflows/derived-artifacts.yml:683-709`), so a developer on a Mac cannot use it to check a change before CI does.

**The CI half is not a flake.** The flake list named this UAT as the cause of #3653's eviction (run 44977790794, `[fenced]`). The job logs say otherwise. In both failing attempts (attempt 2, job 137095986403; attempt 3, job 137178739290) the "Run project-chat UAT" step passed: "Project-chat UAT: all four cases passed with complete receipts". The `FAILED Tests/QA/test_project_chat_uat.py::test_project_chat_uat[fenced]` lines in those logs come from the next step, "Prove missing project evidence fails UAT", whose run is meant to fail ("README read did not supply a verification code"). Both jobs failed in "Run admission-sensitive suites 2", on `Tests/UI/test_console_edge_focus_restyle_scope.py` (TASK-42416). The flake list says CI passes the UAT for most PRs (about 30 merged since that run); no failure of the UAT step itself on CI is known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes name the executor failure behind the local 4-of-4 failure and its cause, from a captured macOS run that records whether it ran inside an agent sandbox
- [ ] #2 Run on macOS outside any agent sandbox, the four variants pass; where the worker cannot run in an environment, the UAT fails or skips with that reason named, not only the generic worker-failure line
- [ ] #3 The notes record how often the "Run project-chat UAT" step itself failed in recent Derived Artifacts runs over a stated window, counting that step's outcome and not the negative-control step's expected FAILED lines
<!-- AC:END -->
