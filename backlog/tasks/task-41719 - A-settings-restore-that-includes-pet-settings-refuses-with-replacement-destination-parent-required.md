---
id: TASK-41719
title: "A settings restore that includes pet settings refuses with replacement_destination_parent_required"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-19 15:59
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

## Implementation Notes

**2026-11-19: handed to TASK-42000.** Owner decision: "Lets change that so pets are per-user/profile and are backed up." TASK-42000 moves pet settings into each profile, which also covers this task's cases with the pets restored. A round-17 attempt that left pet settings out of Replace restores was withdrawn (fu/r17-capture, 4f86eec0f0, reverted in the next commit).

Diagnosis from that attempt (probes on c176f553b3, a cross-profile Replace of Conversations, Settings and Workspaces):
- The review passes. Publication then refuses in `replacement._register_publication_parents` with `replacement_destination_parent_required`, because `~/.config/tldw_chatbook` (`%APPDATA%\tldw_chatbook` on Windows) is under no profile's registered roots. With the folder missing (traced), the nearest existing parent it would fence overlaps a registered or protected root: in the test, a HOME holding the recovery control folder; on a device, `~/.config`, which holds `~/.config/tldw_cli`. With the folder present it refuses because the destination exists (the folder-present runs failed in the same phase; that branch was read from the code). A non-private parent would refuse at the check after that (read, not run). The service reports only `backup_operation_failed`.
- A backup whose pet folder held only a dated copy records the main file as absent. `restore_groups.selected_absent_target_ids` then maps the device's `tamagotchi_pets.json` as an absence to retire, and publication rolls back (`replacement_rolled_back`). Only the dated copy's own refusal hides this today.
- One case works today: when the target already holds every pet file in the backup at the same path, the Replace succeeds and overwrites the device's pet file with the backup's.
