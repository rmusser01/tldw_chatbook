---
id: TASK-42250
title: "Storage-admission hold threads are left running by sandboxed tests that never close a lease"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- tests
- backup
- console
priority: low
references:
- tldw_chatbook/Backup_Recovery/storage_admission.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consoleb stream (TASK-42156, round 19). Threads named `chatbook-storage-admission` (`tldw_chatbook/Backup_Recovery/storage_admission.py:597`, daemon threads started per hold) accumulate at 84 to 212 per xdist worker in a full `Tests/Chat` run, left by sandboxed tests that never close a lease.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The tests or fixtures that leave a lease open are named
- [ ] #2 A `Tests/Chat` run at `-n 4` ends with no `chatbook-storage-admission` thread alive past each test's teardown, counted and recorded
- [ ] #3 A fixture or test closes the lease, or each remaining source has its own task
<!-- AC:END -->
