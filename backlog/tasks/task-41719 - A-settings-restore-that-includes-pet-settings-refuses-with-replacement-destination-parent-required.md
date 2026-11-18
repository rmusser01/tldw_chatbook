---
id: TASK-41719
title: "A settings restore that includes pet settings refuses with replacement_destination_parent_required"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- backup
priority: medium
references:
- tldw_chatbook/Backup_Recovery
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41521 (round 16). A restore that includes the settings group with pet settings refuses with `replacement_destination_parent_required` whenever there is a pet file to restore, whether `~/.config/tldw_chatbook` is missing or holds a dated pet copy. `tamagotchi.config` lives at that global path, outside every profile and shared by all of them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A settings restore with pet settings either restores them or excludes them with a named, user-visible reason; it never fails the whole restore
- [ ] #2 Tests cover the folder missing and present, red before the fix
<!-- AC:END -->
