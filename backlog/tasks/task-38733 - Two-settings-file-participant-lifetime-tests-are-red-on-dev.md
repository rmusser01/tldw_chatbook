---
id: TASK-38733
title: "Two settings-file participant lifetime tests are red on dev"
status: To Do
created_date: 2026-11-04 02:52
labels:
- follow-up
- tests
- backup-recovery
priority: medium
references:
- Tests/Backup_Recovery/test_settings_file_participant_lifetimes.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev d204ddce70 (2026-11-03), `Tests/Backup_Recovery/test_settings_file_participant_lifetimes.py::test_runtime_serialization_is_inside_admitted_source` and `::test_writeback_failure_does_not_wedge_the_operation_file_list` fail when the file is run alone. The second raises `OSError: writeback failed` outside the `pytest.raises` it expects. Both were also red on dev 66155632ed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both tests pass on dev, with the cause recorded in the task notes
- [ ] #2 If the failure exposed a product defect, it is fixed rather than the test loosened
<!-- AC:END -->
