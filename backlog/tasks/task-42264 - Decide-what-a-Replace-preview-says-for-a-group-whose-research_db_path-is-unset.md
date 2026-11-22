---
id: TASK-42264
title: "Decide what a Replace preview says for a group whose research_db_path is unset"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- backup
- product-decision
priority: medium
references:
- tldw_chatbook/Backup_Recovery/config_adapter.py
- tldw_chatbook/Backup_Recovery/profile_paths.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the backuprec stream (round 19). A Replace preview of a group whose `research_db_path` is unset reports the group as `missing_required` (statuses at `tldw_chatbook/Backup_Recovery/config_adapter.py:143` and `:925`; `research_db_path` is read in `profile_paths.py:149`). An unset path is not the same as a database that has gone missing, and the shipped default config sets one (`default_config.toml:490`), so the case arises from a profile that dropped the key. The stream did not reproduce it end to end and the exact copy was not read for this filing. What the preview should say is a product choice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision is recorded in the task notes: what the preview shows and offers for a group whose `research_db_path` is unset
- [ ] #2 The preview shows the decided state, with a test that is red before
- [ ] #3 The matching Replace preview page in `Docs/User_Guide/` says the same
<!-- AC:END -->
