---
id: TASK-40520
title: >-
  Console: the restored persisted scoped session scope-row test is red on dev;
  its row still reads "Scope: everything" after the initial mount
status: To Do
assignee: []
created_date: '2026-11-12 13:55'
labels:
  - console
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_scope_row.py::test_initial_mount_of_restored_persisted_scoped_session_warms_row_and_chip fails on dev 4b90eb860b. It resumes a persisted conversation that has a two-item retrieval scope, saves the screen state, mounts a restored Console from that state, opens the Inspector, and waits 3 s for the retrieval-scope row to read "Scope: 2 items". The row still reads "Scope: everything" (AssertionError at test_console_scope_row.py:987). Only the initial-mount warm (`ConsoleRetrievalController._warm_console_effective_scope_cache_if_stale`, called from `_sync_native_console_chat_ui`) can fill that row, so either the warm no longer runs or fills the cache on a restored mount, or the row is not repainted after it does.

Measured 2026-11-12 while rebasing TASK-36571.1 (host load1 40-58 from other sessions): it failed 2 of 2 runs on a detached dev 4b90eb860b worktree and 2 of 2 on the TASK-36571.1 head, so the branch did not cause it. Those runs used a session-local plugin that marks every node `bootstrap_profile`. Without it, a plain local `pytest Tests/UI/test_console_scope_row.py` errors 40 of the file's 43 tests at setup with `RecoveryRequired: raw_source_selection_changed` (the per-test config redirect, the TASK-33370 class), the same 40 on both trees, so a plain run never reaches this test. The file is not in scripts/ui_pr_gate_census.txt, so no PR lane runs it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause is identified: whether the warm skips or misses a restored mount, or the row is not repainted after the warm
- [ ] #2 On dev, a Console restored from a saved state whose active conversation has a stored retrieval scope shows that scope in the retrieval-scope row and the scope strip without a resume, switch or save
- [ ] #3 The test passes 5 of 5 runs on dev, and a mutant that skips the initial-mount warm still fails it
<!-- AC:END -->
