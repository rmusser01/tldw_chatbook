---
id: TASK-42146
title: "test_legacy_chat_context_roundtrip fails in its seed because the citation migration is not off at start-up"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:06
labels:
- follow-up
- tests
- backup
priority: low
references:
- Tests/Backup_Recovery/test_legacy_chat_context_roundtrip.py
- Tests/Backup_Recovery/test_complete_roundtrip.py
- .github/workflows/backup-recovery-qualification.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the backup stream (round 18). At d85e90bc9e `Tests/Backup_Recovery/test_legacy_chat_context_roundtrip.py::test_native_version1_chat_context_roundtrip_uses_real_message_ids` fails (serial and `-n 4`, through the test wrapper). Its seed child exits 1 (`_run_profile_child`, `Tests/Backup_Recovery/test_complete_roundtrip.py:651`) at line 29 of the generated `seed.py`, `assert migration.writes_enabled is False and migration.ready is False`, right after `TldwCli()` (the `_SEED` script, test file line 51). So at least one of the two is already true for `app.citation_legacy_migration_service`. Not diagnosed: the product may now enable the migration earlier (and the test is stale), or the migration may be wrongly ready. The file is listed in `.github/workflows/backup-recovery-qualification.yml` (lines 55 and 277).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause is named: which of `writes_enabled` and `ready` is true at that point, what sets it, and since which commit
- [ ] #2 The cause is fixed in the product or in the test, as the cause shows, and the node passes
- [ ] #3 If the product changed on purpose, the notes say why the test's expectation was stale
<!-- AC:END -->
