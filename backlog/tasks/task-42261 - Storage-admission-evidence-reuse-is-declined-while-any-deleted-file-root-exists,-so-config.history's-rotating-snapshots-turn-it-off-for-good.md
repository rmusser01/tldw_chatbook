---
id: TASK-42261
title: "Storage-admission evidence reuse is declined while any deleted file root exists, so config.history's rotating snapshots turn it off for good"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- backup
- performance
priority: low
references:
- tldw_chatbook/Backup_Recovery/storage_admission.py
- tldw_chatbook/Backup_Recovery/config_adapter.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the backupdef stream (TASK-42115, round 19). On `fu/r19-backupdef`, `_metadata_evidence` (`tldw_chatbook/Backup_Recovery/storage_admission.py:1232`) returns no evidence when any registered root or historical path is missing (`if not all(os.path.lexists(p) for p in roots | historical): return None`, about line 1277), so evidence reuse is declined while any deleted file root exists. `config.history` (`config_adapter.py:1112`) registers its rotating `config_backup_<stamp>.toml` snapshots (`_CONFIG_HISTORY`, line 15) at a first binding; once one is deleted by rotation, reuse is off from then on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Evidence reuse treats a proven deleted file root as absent, or `config.history`'s registration no longer produces a root that rotation deletes
- [ ] #2 A test registers a rotating snapshot, deletes it, and shows the next admission reuses evidence; it is red before the fix
- [ ] #3 Reuse is still declined for an unproven missing root, pinned by a test
<!-- AC:END -->
