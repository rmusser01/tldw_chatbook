---
id: TASK-16201
title: Repair historical ChaChaNotes fixture for schema v36
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 05:59'
updated_date: '2026-08-14 06:02'
labels:
  - database
  - migrations
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the historical V17-to-current migration test after the shipped V36 note-folder schema made its version-only rollback fixture invalid, without weakening production migration validation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The V17 fixture removes every post-V17 V36 table before replay.
- [x] #2 The named migration test and adjacent ChaChaNotes migration suites pass.
- [x] #3 Scoped static, diff, and mutation evidence pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this repairs a historical test fixture to match the already-shipped V36 migration; production schema and migration policy stay unchanged.

1. Reproduce the V17 replay failure and identify the post-V17 schema objects left by the current-schema template.
2. Remove only the two V36 note-folder tables from the fixture before rolling its version back.
3. Prove either table's retention fails, then run the named and adjacent migration/static gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired the V17 historical migration fixture after the shipped V36 note-folder schema made its version-only rollback invalid. The fixture now drops note_folder_memberships before note_folders, then replays the unchanged production migration chain. RED evidence: retaining either V36 table independently failed the named test at V35→V36 with the corresponding table-already-exists error. GREEN: 67 adjacent ChaChaNotes/migration tests passed; scoped Ruff check, Ruff format, and diff-check passed. ADR required: no; production schema and migration policy are unchanged.
<!-- SECTION:NOTES:END -->
