---
id: TASK-35413
title: A project-instruction status sync that lands before the row is built fails its Console worker
status: Done
assignee:
- '@claude'
created_date: 2026-10-18 09:30
updated_date: 2026-10-18 09:40
labels:
- console
- bug
- flaky
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
While TASK-35026's split was being verified, the Console session-close pending journeys test failed once in six runs under load: a Console worker raised NoMatches for '#console-project-instruction-status-button' on the Inspector's project-instruction status row. The row was already in the screen but had not yet built its button, and a status sync landing in that gap failed the worker. In the app this ends a worker with an error at random; in CI it fails a required test on unchanged code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A status sync that lands before the row has built its button no longer raises, and the row shows the synced status once built
- [x] #2 A test fails deterministically without the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the failure's stack to the query that raised.
2. Make the row's sync tolerate a missing button; compose already builds it from the stored state.
3. Pin it with a sync before compose.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Cause: ConsoleProjectInstructionStatusRow.sync_state (tldw_chatbook/Widgets/Console/console_project_instructions.py) stored the new state and then queried its own button. Its caller, sync_project_instruction_status_row, already returns quietly when the row is absent, but a row that is attached before its compose has run has no button yet, and the query raised NoMatches inside the worker that synced it.

Fix: sync_state keeps the state and returns when the button is not there (QueryError, as its caller does); compose builds the button from that stored state, so the row shows the synced status. Every caller goes through this one method.

Evidence: Tests/UI/test_console_project_instruction_choose_folder.py::test_status_row_synced_before_its_compose_runs_paints_that_state syncs a row before compose and checks the built button's label (no app). It is appended to scripts/ui_pr_gate_census.txt, so it runs in the UI Fast Lane (the census floor in scripts/check_ui_pr_gate_census.py goes 322 -> 323); the row's own test file, Tests/UI/test_console_project_instructions.py, is in no pull-request lane. Run as a standalone script, the same steps raise "NoMatches: No nodes match '#console-project-instruction-status-button' on ConsoleProjectInstructionStatusRow" without the fix and pass with it. The session-close pending journeys test failed 1 of 6 loaded runs before the fix and passed 8 of 8 after.

Local caveat: plain (not private-profile) Tests/UI tests do not run in this machine's environment at the moment. Every one errors at setup with RecoveryRequired: raw_source_selection_changed, on dev too, so this pytest node was not run locally; the UI Fast Lane runs it.

Files: tldw_chatbook/Widgets/Console/console_project_instructions.py, Tests/UI/test_console_project_instruction_choose_folder.py, scripts/ui_pr_gate_census.txt, scripts/check_ui_pr_gate_census.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The Inspector's project-instruction status row no longer fails a Console worker when a sync arrives before it has built its button; it shows the synced status once built.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
