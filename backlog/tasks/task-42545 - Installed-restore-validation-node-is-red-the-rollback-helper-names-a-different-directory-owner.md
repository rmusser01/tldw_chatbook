---
id: TASK-42545
title: "Installed restore validation node is red: the rollback helper names a different directory owner"
status: To Do
assignee: []
created_date: 2026-11-24 23:44
updated_date: 2026-11-24 23:44
labels:
- follow-up
- tests
- dev-red
- backup-recovery
priority: low
references:
- Tests/Backup_Recovery/test_installed_restore_validation.py
- Tests/Backup_Recovery/test_publication_crashes.py
- Tests/Backup_Recovery/test_archive_writer.py
- tldw_chatbook/Backup_Recovery/publication.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Backup_Recovery/test_installed_restore_validation.py::test_existing_owned_directory_metadata_uses_verified_node_rollback` fails on dev 75bac413f5 and on `fu/r26-reds2` with `ValueError: rollback_coverage_mismatch`, raised from `Journal.verify_rollback` (`Backup_Recovery/publication.py`, the directory-metadata branch). No PR lane runs the file, so nothing reports it. It is not part of the ChaChaNotes schema family that TASK-42517 fixed.

Cause (temporary probe at the raise, 2026-11-24): the archived `root` directory's mode and mtime match the live directory, but its producer entry has `owner_id='models.artifacts'`, while the live item being rolled back is `StorageItem("ui.state", "root", ...)`. The check requires `producer.owner_id == item.owner_id`. The test builds its rollback archive with `test_publication_crashes._encrypted_rollback`, which rewrites a `test_archive_writer.captured()` manifest and stamps `models.artifacts` on its files. The directory's producer owner therefore never matches the `ui.state` inventory this test plans against. The owner check has existed since b5251e9a6e; the commit that turned the node red was not bisected.

This is probably test drift: the helper's fixed owner should follow the caller's inventory. Confirm that before changing the product check, which is a real guard against restoring one owner's directory metadata from another owner's rollback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The node passes on dev, with the rollback archive's directory producer owned by the same owner as the live directory it covers; the product's owner check is unchanged, or its change is justified
- [ ] #2 A negative control shows the check still refuses a rollback whose directory producer has a different owner
- [ ] #3 The file runs on a PR lane, timed with that lane's flags
<!-- AC:END -->
