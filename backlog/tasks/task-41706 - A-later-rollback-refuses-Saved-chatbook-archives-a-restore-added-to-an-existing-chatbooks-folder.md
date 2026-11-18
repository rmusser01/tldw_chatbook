---
id: TASK-41706
title: "A later rollback refuses Saved chatbook archives a restore added to an existing chatbooks folder"
status: To Do
assignee: []
created_date: 2026-11-18 06:11
labels:
- follow-up
- backup
priority: medium
references:
- tldw_chatbook/Backup_Recovery/later_rollback.py
- Tests/Backup_Recovery/test_created_owner_rollback.py
dependencies:
- TASK-41521
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from TASK-41521 (review r16rs-rev2). A replace restore publishes the incoming archive files inside the target's existing `chatbooks/` folder (and retires the target's own). TASK-41521 covers a created `chatbooks/` root, but these added files are nested in a root that existed, so the later rollback refuses them with `local_snapshot_absence_unclassified`. Retiring them needs a nested-file proof of the folders they were published into; today's `_created_persona_subtree` does that only for persona folders (it checks the ChaChaNotes core and the persona config).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A later rollback of a restore that added Saved chatbook archives to an existing `chatbooks/` folder retires them and keeps the target's own archives, with a created-file rollback test that is red before the fix
<!-- AC:END -->
