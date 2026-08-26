---
id: TASK-16207
title: Repair local-marks V16 migration fixture
status: Done
assignee:
  - '@codex'
created_date: '2026-08-13 23:53'
updated_date: '2026-08-13 23:53'
labels:
  - test-health
  - database
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the conversation-local-marks migration evidence by making its synthetic V16 database exclude tables introduced after V16.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The V16 fixture does not retain the V36 note-folder tables.
- [x] #2 The V16-to-current migration test reaches the expected local-marks schema.
- [x] #3 Focused, containing-file, mutation, static, and diff evidence pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this repairs a historical test fixture without changing schema or migration policy.

1. Preserve the reproduced V35-to-V36 duplicate-table failure as RED evidence.
2. Remove the two post-V16 note-folder tables before rolling the synthetic database back to version 16.
3. Prove each removal is necessary, then run focused/file/static/diff gates and document the result.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reconstructed the synthetic V16 database without `note_folders` or `note_folder_memberships`, both introduced by V35-to-V36 after the fixture's claimed version. The original test failed at V35-to-V36 because `note_folders` already existed. Removing either cleanup independently reproduced the corresponding duplicate-table failure, proving both fixture corrections are necessary. The containing file passed 19 tests; scoped Ruff and diff checks passed. ADR required: no; production schema and migration behavior are unchanged.
<!-- SECTION:NOTES:END -->
