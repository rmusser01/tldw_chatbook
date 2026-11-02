---
id: TASK-38588
title: Purge stale FTS tokens when media rows are overwritten
status: To Do
created_date: 2026-11-02 14:27
labels:
- bug
- db
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The metadata overwrite path uses INSERT OR REPLACE on an external-content FTS5 table which never purges old tokens, so Library searches keep returning items for words only present in old content forever. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Overwrite path issues the FTS5 delete command before reinsert or deletes then inserts by rowid
- [ ] #2 Words only present in old content are no longer searchable after overwrite
- [ ] #3 Regression test proves stale token removal
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
