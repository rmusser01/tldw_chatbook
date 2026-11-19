---
id: TASK-41514
title: "RAG and model activation tests use fixed 45, 60 and 25 second child budgets that fail at real load"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- tests
- backup
priority: medium
references:
- Tests/Backup_Recovery/test_rag_model_recovery_controls.py
- Tests/Backup_Recovery/test_activation_local_embedding.py
- Tests/UI/test_backup_known_profiles.py
dependencies:
- TASK-41117
- TASK-41301
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by bkreview (round 15). The activation children in `test_rag_model_recovery_controls.py` and `test_activation_local_embedding.py` allow fixed 45, 60 and 25 s, which fail at load 60-110. `test_actual_archive_inspection_survives_navigation` allows 1 s for its summary and is red at base under load. `test_rag_model_recovery_controls.py` is also on no lane. `_wait_while_busy` in `restart_observation.py` is the existing fix (TASK-41117).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The children wait with `_wait_while_busy` or an equivalent idle-aware wait, and each passes at load above 60
- [ ] #2 The 1 s summary wait is a condition wait
- [ ] #3 `test_rag_model_recovery_controls.py` and `test_activation_local_embedding.py` are on a PR lane or the notes name why not
<!-- AC:END -->
