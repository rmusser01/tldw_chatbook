---
id: TASK-42000
title: "Pet settings are per-profile and are backed up and restored"
status: To Do
assignee: []
created_date: 2026-11-19 15:58
updated_date: 2026-11-19 15:58
labels:
- backup
- settings
priority: medium
references:
- tldw_chatbook/Widgets/Tamagotchi
- tldw_chatbook/Backup_Recovery/settings_file_participants.py
- tldw_chatbook/Backup_Recovery/restore_groups.py
- Docs/Backup-and-Recovery.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pet settings (`tamagotchi.config`, the pet files) live in one folder, `~/.config/tldw_chatbook`, which sits outside every profile and is shared by all of them. Because of that, a Replace restore that includes pet settings fails the whole restore with `replacement_destination_parent_required` (TASK-41719): publication will not create files in a folder outside the profile.

Owner decision, 2026-11-19, on TASK-41719's options: "Lets change that so pets are per-user/profile and are backed up". Pets move into each profile, so they are captured and restored like the profile's other settings files and no longer need a shared folder.

Coordinator's call on the upgrade path (engineering, stated to the owner): a profile that has no pet settings of its own starts from a copy of the old shared files the first time it needs them. That way no profile loses the pets it shows today. The shared files are never modified or deleted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each profile keeps its own pet settings: a change made under one profile is not seen by another profile on the same device
- [ ] #2 A backup captures the profile's pet settings, and a Replace restore puts them back into that profile; pet settings never make a restore fail
- [ ] #3 Isolated restore, Manual extraction and a later rollback treat the pet files like the profile's other settings files
- [ ] #4 On upgrade, a profile with no pet settings of its own starts from a copy of the old shared files; the shared files are never changed or deleted
- [ ] #5 TASK-41719's restore cases (pet folder missing, present, backup with only a dated copy) succeed with the pets restored, each red before the change
- [ ] #6 The User Guide and Docs/Backup-and-Recovery.md describe per-profile pets
<!-- AC:END -->
