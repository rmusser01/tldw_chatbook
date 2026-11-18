---
id: TASK-41721
title: "Three replacement publication scope tests are red on dev"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- tests
- backup
priority: medium
references:
- Tests/Backup_Recovery/test_replacement_publication_scopes.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41521 and its review (round 16). `test_actual_mapped_publication_adds_only_its_receipt_bound_selector_scope[split_pair]` fails with `FileNotFoundError` in `_descriptor` (publication.py ~904) during recovery-finish, and `test_complete_cli_replacement_admits_mapped_sources[True|False]` hit their 90 s subprocess timeouts under load (TASK-33641.6 records them on the limit). split_pair fails deterministically at dev; the CLI pair times out under load and passed at load 13-17.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each failure's cause is named, as a defect or as timing
- [ ] #2 All three pass 3 of 3 at `-n 4`, with the load recorded
<!-- AC:END -->
