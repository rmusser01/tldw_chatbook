---
id: TASK-42266
title: "A Replace that keeps a file in the config folder is refused with replacement_config_container_unverified, shown as backup_operation_failed"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- backup
- ux
priority: medium
references:
- tldw_chatbook/Backup_Recovery/replacement.py
- tldw_chatbook/Backup_Recovery/recovery_service.py
- backlog/tasks/task-42151 - A-Replace-at-the-default-config-location-without-the-Settings-group-is-refused-at-its-first-binding.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the backuprec stream (round 19; read on `fu/r19-backuprec`). A Replace without the Settings group at the default config location, when it keeps a file in the config folder, raises `replacement_config_container_unverified` (`tldw_chatbook/Backup_Recovery/replacement.py:940`, `:978`, `:1007`, `:1018`, `:1022`). The service maps it to the generic `backup_operation_failed` (`recovery_service.py:447`), whose text does not say what to do. TASK-42151 fixed the default-location Replace without the Settings group at its first binding; this is the case beside it where a config-folder file is kept.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The refusal reaches the user with copy that names the folder and what to do, not `backup_operation_failed`
- [ ] #2 A test is red before the fix and pins that copy
- [ ] #3 Either the case is completed instead of refused, or the task notes say why it is refused
<!-- AC:END -->
