---
id: TASK-41115
title: "The persona later-rollback test's reopen child fails 'database is locked' on add_note about once in 50 fixture builds at load 36 or more"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- backup
- tests
- flaky
priority: low
references:
- Tests/Backup_Recovery/test_created_persona_subtree_rollback.py
- Tests/Backup_Recovery/thread_diagnostics.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The file runs whole in the push-to-dev Backup_Recovery owner job (TASK-40307), where an intermittent red hides real ones.

**Evidence.** TASK-40306's notes: one reopen child in one `-n 4` run at load 36-57 failed `sqlite3.OperationalError: database is locked` on `add_note`, about 9 s after mount, under ChaChaNotes' 15 s busy timeout, so another connection held the write lock that long. The child now records `observe_notes_write_failure` (`test_created_persona_subtree_rollback.py:178-185`), which names the holding threads on the next failure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The holder of the write lock is identified from a captured failure (local under load or CI) and recorded
- [ ] #2 The product or fixture cause is fixed, and the file passes 20 runs at -n 4 under load 40 or more
<!-- AC:END -->
