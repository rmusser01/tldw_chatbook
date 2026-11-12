---
id: TASK-40309
title: "Any plugin repository download since a backup blocks restoring the whole Tools and skills group until the user deletes folders by hand"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- plugins
- backup
- recovery
priority: medium
references:
- tldw_chatbook/Plugins/backup.py
- tldw_chatbook/Backup_Recovery/restore_plan.py
- tldw_chatbook/Backup_Recovery/data_groups.py
- tldw_chatbook/Backup_Recovery/recovery_service.py
dependencies:
- TASK-40132
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (plugins stream, TASK-40132; owner ruling "Refuse early, feature next"). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** `Plugins/backup.py:83-111` `refuse_added_fetch_folders` raises `plugin_fetch_added_since_backup` (`:40`), called from `restore_plan.py:1179-1181`. Any repository download since the backup, even an update check, therefore blocks a replace restore of the Tools and skills group. The workaround is to restore without that group, or delete everything but `snapshot/` by hand.

**Open question for the owner.** Plugins sit in the `tools` data group with MCP and skills (`Backup_Recovery/data_groups.py:179-187`), so "restore without Tools and skills" (`recovery_service.py:166-175`) also drops MCP and skills. Whether Plugins should get its own data group is undecided; record the decision here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A replace restore over plugin fetch folders added since the backup completes without manual steps, and installed plugins still verify afterwards; a test covers it
- [ ] #2 plugin_fetch_added_since_backup no longer fires for that case
- [ ] #3 The owner's decision on a separate Plugins data group is recorded in the notes
<!-- AC:END -->
