---
id: TASK-42073
title: "test_skills_import.py and test_library_character_repair.py stop at RecoveryRequired while building the test app and run on no PR lane"
status: To Do
assignee: []
created_date: 2026-11-20 02:35
updated_date: 2026-11-20 02:35
labels:
- follow-up
- tests
- library
priority: medium
references:
- Tests/Skills/test_skills_import.py
- Tests/UI/test_library_character_repair.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41715 (round 17) when its touched test files were run on the restacked branch. 25 nodes in `Tests/Skills/test_skills_import.py` and 1 in `Tests/UI/test_library_character_repair.py` (`test_mounted_library_refreshes_real_cas_version_then_retry_succeeds`) fail with `RecoveryRequired: raw_source_selection_changed` while building the test app: `_build_test_app` -> `load_settings` -> `config_participants.py` -> `Backup_Recovery/raw_participants.py`. This happens on dev e446ab06eb and on dev a6df4d8d48, with or without round 17.

Two more nodes in `test_library_character_repair.py`, `test_repair_continuation_reaches_every_candidate_and_restarts_after_mutation[size0|size1]`, fail for another reason. They build a bare `App()` with their own database, so no profile marker applies, and assert `len(controller.candidates) == 20` after a single `pilot.pause()`. Run serially (alone or with the file), they fail with `assert 0 == 20`; at `-n 4` they have passed.

Neither file carries the `bootstrap_profile` or `@private_profile_test` marker that app-building census files use, and no PR lane runs either file. That is the same shape as TASK-41772, whose scope is 63 named media-series guards and does not include these files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every test in both files that builds the app gets the profile setup it needs and no longer stops at RecoveryRequired
- [ ] #2 Each of those tests that then passes 3 of 3 serially with the UI lane's flags runs on a PR lane
- [ ] #3 Any test that still fails after the setup fix has its failure recorded, with its own task or fix named
- [ ] #4 The two repair-continuation nodes wait for the candidates they assert instead of a single pause, and pass 3 of 3 serially
<!-- AC:END -->
