---
id: TASK-36383
title: Three Library file-notes workspace tests are flaky on dev
status: To Do
created_date: 2026-10-22 23:58
labels:
- follow-up
- tests
- flaky
- library
priority: low
references:
- Tests/UI/test_library_file_notes_workspace.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-35649's implementation (2026-10-22). On dev 28beeb9ed5, four reruns of `Tests/UI/test_library_file_notes_workspace.py` failed 1, 1, 3 and 2 of these tests:
- `test_file_notes_navigation_and_key_guidance_use_one_phrase`
- `test_initial_root_scan_projects_checking_authority_while_actions_are_gated`
- `test_create_move_delete_protect_and_restore_use_real_service`

The cause is not yet identified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each test's failure mode is identified and recorded
- [ ] #2 Each test passes 20 of 20 serial single-node runs on dev
- [ ] #3 A fix that changes a wait shows the old form failing under the same conditions (a negative control)
<!-- AC:END -->
