---
id: TASK-42022
title: "A file root a restore retired stays in the profile binding"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- backup
priority: low
references:
- tldw_chatbook/Backup_Recovery/storage_admission.py
- tldw_chatbook/Backup_Recovery/config_participants.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-42000's review (round 17). Capture registers each config-folder file as its own root; when a Replace retires the file, the root stays registered. TASK-42000 made containment and data-dir association treat an absent root as holding nothing, but a stale root still disables `_selector_evidence` reuse for the profile and reaches containment checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Publication drops a registered file root when it retires that file
- [ ] #2 A test shows the root is gone after a Replace that retired it
<!-- AC:END -->
