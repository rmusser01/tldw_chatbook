---
id: TASK-42262
title: "Two Tests/Backup_Recovery/test_settings_file_participant_lifetimes.py cases are red at base"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- tests
- backup
priority: medium
references:
- Tests/Backup_Recovery/test_settings_file_participant_lifetimes.py
- tldw_chatbook/Backup_Recovery/raw_participants.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the backupdef stream (round 19). On dev e5626ea043, through the scratch wrapper at `-n 4`:
- `test_runtime_serialization_is_inside_admitted_source` fails at `Tests/Backup_Recovery/test_settings_file_participant_lifetimes.py:129` with `assert [True, True] == [True]`.
- `test_writeback_failure_does_not_wedge_the_operation_file_list` fails with `RecoveryRequired: raw_operation_provenance_invalid`, raised from `live_state` (`tldw_chatbook/Backup_Recovery/raw_participants.py:434`) through `remove_temporary` (line 1410), after `close_descriptor` raised `raw_resources_not_retired` (line 587) in the preceding `merge_templates` call.
The stream reported `raw_operation_provenance_invalid` for both; only the second shows it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of each red node is named and fixed
- [ ] #2 Both cases pass through the scratch wrapper at `-n 4` on a branch carrying the fix
- [ ] #3 The file runs on a PR lane, or the reason it does not is recorded
<!-- AC:END -->
