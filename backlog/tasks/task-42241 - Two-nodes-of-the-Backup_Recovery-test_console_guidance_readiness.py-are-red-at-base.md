---
id: TASK-42241
title: "Two nodes of Tests/Backup_Recovery/test_console_guidance_readiness.py are red at base"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- tests
- console
- backup
priority: medium
references:
- Tests/Backup_Recovery/test_console_guidance_readiness.py
- Tests/Backup_Recovery/test_home_citation_retirement.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consolea stream (round 19). The file is `Tests/Backup_Recovery/test_console_guidance_readiness.py` (the stream's list gave it under `Tests/UI/`). On dev e5626ea043, through the scratch wrapper at `-n 4`, `test_native_guidance_is_coherent_and_refreshes_after_real_settings_save` and `test_native_display_pairs_preserve_outputs_and_later_fresh_send_guard` fail. The first fails at `Tests/Backup_Recovery/test_console_guidance_readiness.py:279`, in the child-process runner `_run` at `Tests/Backup_Recovery/test_home_citation_retirement.py:347`, whose assertion message is the child's captured log. The cause is not named.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of each red node is named and fixed
- [ ] #2 Both nodes pass through the scratch wrapper at `-n 4` on a branch carrying the fix
- [ ] #3 The file runs on a PR lane, or the reason it does not is recorded
<!-- AC:END -->
