---
id: TASK-42023
title: "On Windows a folder at the old shared pet path makes the pet copy retry every start"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
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
- [ ] #1 A non-file at the old shared path is treated as nothing to copy on every platform
- [ ] #2 A test covers it (skipped where the platform cannot create the case)
<!-- AC:END -->
