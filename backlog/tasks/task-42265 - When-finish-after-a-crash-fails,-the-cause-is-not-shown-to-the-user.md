---
id: TASK-42265
title: "When finish after a crash fails, the cause is not shown to the user"
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
- tldw_chatbook/Backup_Recovery/recovery_service.py
- tldw_chatbook/UI/Screens/backup_restore_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the backuprec stream (TASK-42154, round 19; code read on `fu/r19-backuprec`). After a Replace killed after publication, `finish` can fail, and the user sees only the generic `backup_operation_failed` text (`recovery_service.py:216`: "Review the selected file and local folders, then try again. ..."), not the cause the service had.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failed `finish` shows the user the specific cause the service holds, in plain copy, with a test per distinct cause that is red before the fix
- [ ] #2 The generic copy remains only for a cause that is unknown, and the user is told what to do next
- [ ] #3 The matching `Docs/User_Guide/` page reflects the new copy
<!-- AC:END -->
