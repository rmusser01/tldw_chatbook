---
id: TASK-41320
title: "Workflow reconcile still refuses, until restart, a retained run with an open preparation, an unproven control return or an unproven metadata read"
status: To Do
created_date: 2026-11-16 19:55
labels:
- follow-up
- agents
- workflows
- console
priority: medium
references:
- tldw_chatbook/Chat/console_workflow_runs.py
- tldw_chatbook/Agents/workflow_run_service.py
- backlog/decisions/285-native-phased-agent-workflows.md
- https://github.com/rmusser01/tldw_chatbook/pull/3543
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. These are the limits TASK-40718 (PR #3543) recorded. Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code; line numbers are that tree's, and paths are under `tldw_chatbook/`.

**Why.** TASK-40718 lets a user close a failed native workflow phase as reviewed history once its work has returned, so the chat is usable again without a restart. Three retained states still refuse that reconcile with `workflow_resources_still_live`. In each, the chat keeps its intake claim, new turns are refused ("A turn is already preparing."), and only a restart frees it.

**Evidence.**
- **An open, never-launched preparation** counts as live work. `WorkflowRunService._has_live_workflow_resources` (`Agents/workflow_run_service.py:2267-2300`) returns True for any entry in `runtime._workflow_preparations` for the workflow (`:2276-2284`), whether or not it ever launched. Reconcile does not close such preparations.
- **An unproven control return.** `ConsoleWorkflowCoordinator.reconcile` (`Chat/console_workflow_runs.py:2426-2492`) refuses at `:2446-2460` when `run.control_unknown` is set, or `run.control_return` exists but is not set.
- **An unproven metadata read.** It refuses at the same place when the run's final-metadata cell's `disposed` future is not done, was cancelled, raised, or did not return `True`. App shutdown also refuses while that cell's marker stays in `_metadata_cells` (`:2658-2669`, `workflow_original_metadata_shutdown_unresolved`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A retained run whose only live resource is an open, never-launched preparation can be reconciled: the preparation is closed without effects, the run is recorded as reviewed history, and the chat accepts a new turn, shown by a test that is red on dev
- [ ] #2 For an unproven control return and an unproven metadata read, the chat can be released without a restart through an explicit, recorded user decision, or the task notes and ADR-285 record why only a restart may release it and the UI says so
- [ ] #3 Reconcile still refuses while any launched work, admitted owner or live native thread remains (the existing TASK-40718 tests still pass)
<!-- AC:END -->
