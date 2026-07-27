---
id: TASK-927
title: >-
  Bulk DB maintenance workers carry the same wrong-path bug TASK-899 fixed
status: To Do
assignee: []
created_date: '2026-07-27 09:00'
labels:
  - settings
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-899 fixed `Tools_Settings_Window._get_database_path()` so single-database vacuum, backup, restore and integrity-check resolve through `config.py`'s profile-aware resolvers instead of hardcoded literals.

The **"all databases"** variants of those workers were left untouched and do not go through `_get_database_path()`. They carry their own copies of the same hardcoded paths, so they inherit the identical defect: wrong filenames (`tldw_evals_db.db`, `tldw_prompts_db.db`, `tldw_media_db.db` rather than `evals.db`, `tldw_chatbook_prompts.db`, `tldw_chatbook_media_v2.db`) and no profile directory segment.

The consequence is the same one TASK-899 documented. Vacuum, backup and check guard on `exists()` and therefore silently do nothing. Any restore-style path that writes unconditionally would write to a phantom location while the real database is untouched.

This is the more dangerous half in practice: "back up all databases" is exactly what a cautious user clicks before an upgrade.

The conversation/character export workers were also observed to build database paths independently and should be checked in the same pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] The bulk workers resolve every database through the same resolvers the single-database workers now use
- [ ] No hardcoded database filename or path literal remains in `Tools_Settings_Window.py`
- [ ] A bulk backup produces files for the databases that actually exist, proven by a test
- [ ] The conversation/character export workers are audited in the same pass and either fixed or explicitly cleared
<!-- AC:END -->
