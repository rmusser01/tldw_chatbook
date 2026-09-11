---
id: TASK-32461
title: Fix destination-shell test drift on clean dev (5 failures + 2 flaky)
status: To Do
assignee: []
created_date: '2026-09-11 23:54'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-existing on origin/dev (verified 2026-09-11): test_destination_shells.py fails 5 on clean HEAD -- test_library_destination_service_failure_uses_recovery_copy, test_library_policy_denial_uses_runtime_recovery_taxonomy, test_library_policy_denial_uses_runtime_recovery_state_selector, test_destination_action_buttons_explain_their_outcome[schedules], test_models_shell_keeps_external_paths_inside_the_dedicated_edit_view. Additionally test_schedules_pending_run_uses_shared_approval_status_taxonomy and test_schedules_empty_state_reads_as_live_queue_with_recovery_path are flaky (fail under suite load, pass in isolation/rerun). Three unrelated subsystems (library recovery taxonomy, schedules copy, models edit view) -- triage each separately.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All 5 deterministic failures fixed with root cause,Flaky schedules pair stabilized or root-caused with a follow-up note,Suite green on two consecutive full runs
<!-- AC:END -->
