---
id: TASK-41518
title: "A reviewed way to delete a reset-<id> folder so a plugin authority reset can reclaim disk"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- plugins
- backup
priority: medium
references:
- tldw_chatbook/Plugins/backup.py
- tldw_chatbook/Backup_Recovery/plugin_authority_reset.py
dependencies:
- TASK-41107
- TASK-41108
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by plga (round 15, TASK-41107). After a post-restore plugin authority reset, package bytes move into `reset-<id>/` and keep their disk space. The reset dialog now says so, but no reviewed action deletes the folder. The folder also holds hints for kept saved-data folders, so deletion needs a review that names what is lost.

Also found by plga: marketplace sources added under a reset authority keep MACs signed by the old key, so a refresh may fail per source (unverified). TASK-41108 covers only the acquisition download records, not marketplace sources.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The user can delete a `reset-<id>/` folder from a review that names its contents, including the saved-data hints
- [ ] #2 Deleting reclaims the space, and the plugin storage figure is correct afterwards
- [ ] #3 A test covers the delete and the refusal while a restore is running
- [ ] #4 Marketplace sources added under a reset authority are shown to refresh, or are re-signed or flagged for re-adding with a named review; the unverified claim is reproduced or ruled out, with a test
<!-- AC:END -->
