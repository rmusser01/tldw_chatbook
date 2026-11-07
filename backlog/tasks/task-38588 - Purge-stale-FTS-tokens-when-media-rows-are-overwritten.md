---
id: TASK-38588
title: Purge stale FTS tokens when media rows are overwritten
status: Done
created_date: 2026-11-02 14:27
labels:
- bug
- db
priority: medium
updated_date: 2026-11-03 02:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The metadata overwrite path uses INSERT OR REPLACE on an external-content FTS5 table which never purges old tokens, so Library searches keep returning items for words only present in old content forever. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Overwrite path issues the FTS5 delete command before reinsert or deletes then inserts by rowid
- [x] #2 Words only present in old content are no longer searchable after overwrite
- [x] #3 Regression test proves stale token removal
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
FTS5 delete command with true old values on every title or content changing path; regression test drives the real external-content table and asserts integrity. Commit 6813f053d8.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
