---
id: TASK-41720
title: "Tree owners outside the created-root set may refuse a later rollback of a restore-created folder"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- backup
priority: low
references:
- tldw_chatbook/Backup_Recovery/later_rollback.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41521 (round 16), read from the code and not run. Folder owners not in `_CREATED_TREE_ROOT_OWNERS` (for example `rag_profiles/`) likely refuse a later rollback when a restore created their folder, the gap TASK-41113 closed for personas and TASK-41521 closed for single files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each backup-enabled folder owner is listed with whether a restore can create its folder
- [ ] #2 Each that can is covered by a created-tree rollback test, red before the fix
<!-- AC:END -->
