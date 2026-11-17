---
id: TASK-41517
title: "test_production_settings_inspects_private_and_deleted_metadata[size0] fails locally on base on a 5 s provenance wait"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- tests
- backup
priority: medium
references:
- Tests/Backup_Recovery
dependencies:
- TASK-41122
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the pc stream (round 15). The `[size0]` parameter fails on base locally on a fixed 5 s provenance wait, so a local run of the Personal Context custody tests is red before any change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The provenance wait is a condition wait with a bounded deadline, and the node passes 10 of 10 locally at `-n 4`
- [ ] #2 The notes record the load
<!-- AC:END -->
