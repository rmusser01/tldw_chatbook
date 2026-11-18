---
id: TASK-41707
title: "MCP store fresh copies created by a restore are not covered by a later-rollback test"
status: To Do
assignee: []
created_date: 2026-11-18 06:11
labels:
- follow-up
- backup
priority: low
references:
- tldw_chatbook/MCP/recovery.py
- tldw_chatbook/Backup_Recovery/later_rollback.py
- Tests/Backup_Recovery/test_created_owner_rollback.py
dependencies:
- TASK-41521
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from TASK-41521 (review r16rs-rev2). When the target already holds an MCP store, a restore can write the incoming store as a retained `fresh` copy (`mcp.local`, `mcp.permissions` with `fresh.bak`, `mcp.context`). TASK-41521's `[tools]` case only restores into a target without the stores, so these `fresh` copies were never created in a test, and nothing shows a later rollback retiring them with their installed IDs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A later rollback of a restore that wrote MCP `fresh` copies retires them with their installed IDs and keeps the target's own stores, with a created-file rollback test that is red before the fix
<!-- AC:END -->
