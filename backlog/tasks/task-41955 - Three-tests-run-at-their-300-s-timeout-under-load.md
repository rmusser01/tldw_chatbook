---
id: TASK-41955
title: "Three tests run at their 300 s timeout under load"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- flaky
priority: low
references:
- Tests/Architecture
- Tests/Chat/test_console_semantic_mutation_inventory.py
- Tests/Workspaces/test_native_confirmed_undo.py
dependencies:
- TASK-41641
- TASK-41655
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** `test_task_15103_review_ledger_canonical_schema_and_arithmetic` takes 291-294 s alone at load ~60 against its 300 s timeout; `test_repository_scan_results_are_cached_and_immutable` took 286 s; in `Tests/Workspaces/test_native_confirmed_undo.py`, 4-6 different nodes fail per run under load on the 8 s `_wait_for`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each test finishes well inside its bound under load (faster, a load-tolerant bound, or split), with the reason recorded
<!-- AC:END -->
