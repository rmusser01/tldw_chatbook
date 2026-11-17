---
id: TASK-41532
title: "Dev's push-to-dev Derived Artifacts run 43917600414 failed four nodes at dev 5e3c8b2bd4"
status: To Do
assignee: []
created_date: 2026-11-17 05:39
labels:
- follow-up
- tests
- ci
priority: high
references:
- Tests/UI/test_library_shell.py
- Tests/Backup_Recovery/test_operational_owners.py
dependencies:
- TASK-41505
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reported by the coordinator at the filing review (round 15). The run failed:
- `Tests/UI/test_library_shell.py::test_slash_focuses_the_prompts_filter_not_the_rail_search`
- `Tests/UI/test_library_shell.py::test_library_media_edit_save_stays_gated_through_detail_refresh`
- `Tests/UI/test_library_shell.py::test_library_media_complete_facets_commit_during_initial_recovery[loading-All]`
- `Tests/Backup_Recovery/test_operational_owners.py::test_run_log_real_bound_writer_is_excluded`
Whether the three library_shell nodes share TASK-41505's cause (a single pause racing a handler) is not known; read the run logs first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the four nodes passes in the push-to-dev job, shown by a CI run on the fix
- [ ] #2 The failure shape of each node from run 43917600414 is recorded, with any shared cause named
- [ ] #3 If the library_shell nodes share TASK-41505's cause, the notes say so and fix them there
<!-- AC:END -->
