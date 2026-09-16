---
id: DRAFT-1
title: Migrate local backup and restore to selectable data groups
status: In Progress
assignee: []
created_date: '2026-09-15 14:49'
labels:
  - backup
  - recovery
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/2642'
  - backlog/decisions/126-complete-local-backup-and-recovery.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users should select the data groups they want to back up and restore, with Everything as the default and one coherent backup set. The requester approved migrating the existing Python backup system and explicitly chose replacement of selected groups with a safety copy while preserving unselected groups.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Backup defaults to Everything and supports explicit named data groups, with required dependencies and shared physical stores shown and included consistently.
- [ ] #2 Each group uses declared storage-owner capture/validation/relocation contracts in one backup set; selecting a group does not accidentally capture unrelated groups.
- [ ] #3 Restore from a full or selective backup can select groups, replaces only the reviewed selected/dependent groups after a verified safety copy, and preserves unselected stored data.
- [ ] #4 Existing full backups remain readable and existing complete backup, isolated restore, replacement, rollback, encryption and credential protections remain supported on macOS, Linux and Windows.
- [ ] #5 Preview and execution bind the exact selection and dependency closure; changed/unknown/unsafe selections are refused before publication.
- [ ] #6 Targeted native tests and first-time/power-user UAT demonstrate selective capture, selected-group restoration, unchanged unselected data, dependency handling and legacy/full recovery without Go or unrelated feature work.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
User explicitly authorized architecture migration after reviewing independently selectable data groups with a common backup coordinator/archive. Clarification: replace selected groups and preserve unselected groups; no record-level merging. Prior backup UAT evidence retained at58e9c632. Separate Models fixture proposal remains outside this migration unless explicitly authorized. Read-only mapping is in progress; no production edits yet.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
