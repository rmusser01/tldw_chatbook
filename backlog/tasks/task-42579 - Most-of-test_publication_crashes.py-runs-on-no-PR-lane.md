---
id: TASK-42579
title: "Most of test_publication_crashes.py runs on no PR lane"
status: To Do
assignee: []
created_date: 2026-11-26 15:58
updated_date: 2026-11-26 15:58
labels:
- follow-up
- tests
- ci-lanes
- backup-recovery
priority: low
references:
- Tests/Backup_Recovery/test_publication_crashes.py
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Backup_Recovery/test_publication_crashes.py` holds the restore journal's publication crash, rollback-verification and journal-cache tests. Only 2 of its 49 tests run on a PR lane: `test_append_and_flush_handle_each_record_once` and `test_cached_journal_history_still_refuses_changed_records`, both in architecture-guards. The whole file runs only in the "Backup_Recovery owner suites (push to dev)" job. That job runs only after a merge: on a dev push while `vars.CI_TIERED_ENABLED != 'true'`, and on the nightly dispatch. TASK-42545 changed its `_encrypted_rollback` helper, and TASK-42545's laned `test_installed_restore_validation.py` imports that helper. A regression in the other 47 tests is therefore found only after merge.

Green on dev 6e41919880 (2026-11-26): the file passed 49/49 at `-n 4`, run together with `test_core_owners.py` (115 passed).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The whole file runs on a PR lane, with no node listed twice across PR steps (the two architecture-guards nodes move or stay, as the lane contract requires)
- [ ] #2 The file is timed serially with that lane's flags, and the time and load are recorded in the task notes
- [ ] #3 Any node red on that lane is fixed (a product fix with red/green, or a test repair that keeps every assertion)
<!-- AC:END -->
