---
id: TASK-38593
title: Recover MCP permission publication fence after transient write failure
status: To Do
created_date: 2026-11-02 14:28
labels:
- bug
- mcp
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The fence uncertain flag is never cleared and begin is not paired with complete or fail around the temp-file write, so one transient disk-full during a permission save permanently fails-closed passive-hook MCP grants until restart. Note this was introduced by commit e7dc47f289 which is on local dev but not yet pushed to origin/dev. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 uncertain flag clears when a subsequent load positively re-reads the file
- [ ] #2 begin is paired with complete or fail in a try finally
- [ ] #3 A transient write failure no longer disables passive-hook grants for the whole session
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->


**Not fixable on dev (2026-11-06).** The fence this task repairs, `_PermissionPathFence` in `MCP/permission_store.py` (the `pending`/`uncertain` publication state read by passive MCP hook grants), exists only in the native-workflow work: it was added by `e7dc47f289` ("revalidate original MCP grants for passive hook context"), which is now in draft PR #3411, not on dev. PR #3361 was first opened on a local dev branch that carried that work, so its fix commit `8162e28ded` applied there. When #3361 was rebuilt on dev with only its audit commits, that commit was left out. #3411's `permission_store.py` still has the stuck-`pending` bug (`observe` returns early while `pending or uncertain`). Land `8162e28ded` (fix, `Tests/MCP/test_permission_publication_fence.py`, `Tests/Backup_Recovery/test_activation_mcp.py`) on #3411's branch, or after #3411 merges.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
