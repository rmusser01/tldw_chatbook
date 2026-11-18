---
id: TASK-40717
title: "Startup workflow recovery runs with workflow runs off, and its failure stops sub-agent result delivery to ordinary chats"
status: Done
created_date: 2026-11-13 06:28
labels:
- follow-up
- agents
- console
- bug
priority: low
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

**Why.** Workflow runs are off by default. `[agents] workflow_runs_enabled` is read by `UI/Console_Modules/wiring.py:191-195` with a `False` fallback; `Config_Files/default_config.toml:377` has it commented out.

Even so, `Chat/console_runtime.py:4988` installs `_project_workflow_recovery` as the fleet wake's recovery projection without checking that flag. The projection (:5319-5327) does three things:
- builds the workflow service with `ensure_workflow_service()`, which can raise "Native workflow stores are unavailable" at :5303;
- runs `project_recovery`, which issues one SQL update per workflow in the startup audit (`Agents/workflow_run_service.py:1844-1859`), so none when there are no workflow rows;
- runs `recovery_protections`, which protects the worktrees of workflow runs left by an earlier session that had the flag on.

The projection only runs when a runs database exists (`Chat/console_fleet_wake.py:1446-1461`).

**If the projection raises anything:**
- `Chat/console_fleet_wake.py:1494-1504` marks wake recovery not ready, with reason `history_unavailable`.
- `_attempt` returns early while it is not ready (:829).
- Recovery runs once per session (:453-454).

So for the rest of the session, finished sub-agent results are not delivered to ordinary chats. No concrete trigger has been reproduced on a default install, so the risk is the coupling: flag-off sessions depend on workflow code succeeding. TASK-38971 fixed a different recovery failure, the missing runs database.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With `workflow_runs_enabled` off and no workflow runs in the startup audit, startup recovery builds no workflow service. Workflow runs left by an earlier session still get their worktree protections.
- [x] #2 On an install with no saved workflow runs, a failing workflow projection cannot stop sub-agent result delivery. When saved workflow runs exist and the projection fails, delivery stays stopped (fail closed), because worktree protections are runtime-wide (owner decision 2026-11-13).
- [x] #3 A test with workflow runs off and a raising workflow projection shows a sub-agent result still delivered to an ordinary chat.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Reason: a no-op skip inside the existing recovery projection; no new contract, store or policy.
1. Reproduce: with workflow runs off, no workflow rows, and a workflow projection that raises ("Native workflow stores are unavailable"), show a finished sub-agent result is never delivered to an ordinary chat.
2. The startup audit records whether any saved (non-removed) workflow run exists, in the same transaction that already reads `workflow_runs`. This is the exact set `recovery_protections` and `project_recovery` read, so a legacy closed run with live evidence still counts.
3. `_project_workflow_recovery` returns before building the workflow service when the audit saw no saved workflow run.
4. Pin protections with the flag off (including the closed-but-active shape the audit's `workflows` tuple leaves out).
5. Mutation check, covering files base vs head, preflight, Tests/Architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The startup audit (`RuntimeRecoveryResult`) now records `workflow_history`: whether any non-removed workflow run exists, read in the same ledger transaction that already reads `workflow_runs` (`DB/automatic_work.py`). `ConsoleRuntime._project_workflow_recovery` returns before building the workflow service when it is false, so a profile that never ran a workflow cannot have wake recovery fenced by workflow code, whatever the flag says.

Why not "the audit's `workflows` tuple is empty": that tuple leaves out closed, completed and stopped runs, but `recovery_protections` still protects a closed run with live evidence (the `legacy_closed` shape). A mutation to `bool(workflows)` fails three protection cases, including the new flag-off one.

Tests: `test_workflow_projection_without_saved_workflows_cannot_fence_delivery` (flag off, workflow service unbuildable, sub-agent result delivered) failed on base with "wake recovery failed (exception_type=RuntimeError)" and passes now; it fails again with the guard removed. The protection test gains two flag-off cases (plain, legacy closed); both keep the folder protected.

AC #2 was reworded to the owner's 2026-11-13 decision: keep failing closed. With saved workflow runs, a failing projection still stops delivery to every chat, because worktree protections are runtime-wide and a woken turn run without them could write into a protected folder. Installs with no workflow history are no longer affected. The two rejected options were a fence that blocks all worktree admission while deliveries continue (in practice it also blocks ordinary agent runs), and running the protections step on its own so only its failure stops delivery.

Files: `Agents/automatic_work_budget.py`, `DB/automatic_work.py`, `Chat/console_runtime.py`, `Tests/Chat/test_automatic_wake_recovery.py`, `Tests/Chat/test_workflow_recovery.py`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
