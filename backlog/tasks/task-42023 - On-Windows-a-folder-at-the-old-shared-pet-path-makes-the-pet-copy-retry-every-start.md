---
id: TASK-42023
title: "On Windows a folder at the old shared pet path makes the pet copy retry every start"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:54
labels:
- follow-up
- pets
- windows
priority: low
references:
- tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-42000 (round 17), not run. If a directory sits at the old shared pet file's path, the Windows open raises `PermissionError`, which the upgrade copy treats as a failed read, so it retries on every start instead of treating it as nothing to copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A directory at the old shared path is treated as nothing to copy on every platform, including Windows, where the open raises `PermissionError`
- [ ] #2 A regular file that cannot be read (for example locked by another process) still counts as a failed read: nothing is written, no marker, and the next start retries, as `_read_shared_pets` documents
- [ ] #3 Tests simulate the Windows `PermissionError` for both cases on every platform, the directory case red before the fix
<!-- AC:END -->
