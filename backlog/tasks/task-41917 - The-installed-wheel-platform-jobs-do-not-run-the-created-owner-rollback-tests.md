---
id: TASK-41917
title: "The installed-wheel platform jobs do not run the created-owner rollback tests"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- tests
- backup
priority: low
references:
- Tests/Backup_Recovery/run_platform_product.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41521's review (round 16). `run_platform_product.py` lists `test_created_persona_subtree_rollback.py` in its installed-wheel selections (`full`, `replacement`, `replacement-persona`), but not the new `test_created_owner_rollback.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The created-owner rollback tests are in the selections that cover rollback, or the notes say why not
<!-- AC:END -->
