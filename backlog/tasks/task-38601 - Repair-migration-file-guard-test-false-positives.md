---
id: TASK-38601
title: Repair migration-file guard test false positives
status: To Do
created_date: 2026-11-02 14:29
labels:
- test
- db
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/DB/test_migration_file_sources.py is red on dev because its regex cannot see runtime-read migrations opened via with_name or f-string paths, so its signal is drowned by false positives. See bug-sweep-2026-11-02-dev.md section 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Guard regex or reader-module list covers with_name and f-string migration readers
- [ ] #2 The currently red run passes while genuine decorative files are still caught
- [ ] #3 Dreams guardian and workspace readers are recognized
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
