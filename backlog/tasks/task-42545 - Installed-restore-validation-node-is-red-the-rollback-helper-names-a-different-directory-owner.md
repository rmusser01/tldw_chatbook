---
id: TASK-42545
title: "Installed restore validation node is red: the rollback helper names a different directory owner"
status: Done
assignee:
- '@claude'
created_date: 2026-11-24 23:44
updated_date: 2026-11-26 08:30
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
- [x] #1 The node passes on dev, with the rollback archive's directory producer owned by the same owner as the live directory it covers; the product's owner check is unchanged, or its change is justified
- [x] #2 A negative control shows the check still refuses a rollback whose directory producer has a different owner
- [x] #3 The file runs on a PR lane, timed with that lane's flags
<!-- AC:END -->

## Implementation Plan

1. Confirm the drift: find the commit that changed `_encrypted_rollback`'s directory producer owner away from the `ui.state` this test's inventory uses.
2. Give `_encrypted_rollback` an `owner` keyword (default `models.artifacts`, which `_replacement`'s callers use) stamped on the owners list, the files and the directory producers; this test passes `owner="ui.state"`. The product's owner check in `publication._verify_rollback` stays unchanged.
3. Negative control: a test node that builds the rollback with a different directory owner and asserts `verify_rollback` refuses it with `rollback_coverage_mismatch`; show it red with the product check temporarily removed.
4. Check the rest of the file at base (`-n 4`), lane the file in `recovered-suites` (or the next preferred home), timed with the step's flags, and update the lane contract tests.

## Implementation Notes

Test drift, fixed in the test helper; the product's owner check in `publication._verify_rollback` is unchanged.

- **Cause.** 544bc58e7261343d6365f7d10f018df5b077f891 (TASK-33422, 2026-09-30, landed via #2935) moved `_replacement`'s inventory from `ui.state` to `models.artifacts` and made `_encrypted_rollback` stamp `models.artifacts` on every owner, file and directory producer. Before it, the root directory's producer was `ui.state`. `test_existing_owned_directory_metadata_uses_verified_node_rollback` still plans against a `ui.state` inventory, so `producer.owner_id != item.owner_id` refused it with `rollback_coverage_mismatch`. Mode and mtime matched; only the owner differed.
- **Fix.** `_encrypted_rollback` takes an `owner` keyword (default `models.artifacts`, which `_replacement`'s callers keep). It is stamped on the owners list, the files and the directory producers. The node is parametrized: `ui.state` passes with every original assertion; `models.artifacts` asserts the refusal (AC#2).
- **Red/green.** Base: the node fails with `rollback_coverage_mismatch` (1 failed, 16 passed for the file). After: 18 passed. Negative control: with `producer.owner_id != item.owner_id` temporarily removed from `publication.py`, the `models.artifacts` case went red (`DID NOT RAISE`); reverted.
- **Rest of the file.** The other 16 nodes were green on dev at `-n 4`; nothing else to repair.
- **Lane (AC#3).** recovered-suites ("Run the recovered suites"), in the sandboxed block after the deleted-root guards. The file has no `bootstrap_profile` mark, and a `RecoveryRequired.__init__` recorder counted 0 constructions. No other PR step lists it. Serial time with the step's flags (`--storage-lease-guard --timeout=180`): 18 tests, 15.4 s at load 14-16. The lane contract's `RECOVERED_SUITES_TARGETS` lists it.

- **Runs.** The file and `test_publication_crashes.py` passed 20/20 at `-n 4` in the round's soak of touched files, at load 27-75. The recovered-suites step passed whole: 2066 passed, 1 skipped, in 1284 s at load 17-19.

Files: Tests/Backup_Recovery/test_publication_crashes.py, Tests/Backup_Recovery/test_installed_restore_validation.py, .github/workflows/derived-artifacts.yml, Tests/CI/test_ci_queue_pressure_contract.py.

