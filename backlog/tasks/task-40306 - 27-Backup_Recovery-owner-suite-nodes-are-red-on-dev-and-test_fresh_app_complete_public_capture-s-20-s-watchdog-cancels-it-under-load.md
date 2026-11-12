---
id: TASK-40306
title: "27 Backup_Recovery owner-suite nodes are red on dev, and test_fresh_app_complete_public_capture's 20 s watchdog cancels it under load"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- tests
- backup
- recovery
- dev-red
priority: medium
references:
- Tests/Backup_Recovery/test_created_persona_subtree_rollback.py
- Tests/Backup_Recovery/test_restore_plan.py
- Tests/Backup_Recovery/test_visual_identity_restore.py
- Tests/Backup_Recovery/test_inventory_unused_owners.py
- Tests/Backup_Recovery/test_builtin_later_snapshot.py
dependencies:
- TASK-39116
- TASK-40132
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (census stream review r13b-rev1; plugins stream, TASK-40132). Measured red at base ae9f9e8d21 and at the census branch head; no round-13 commit touches these files except a test that moved out of `test_restore_plan.py` (net zero). Checked by reading at 8b633d0101, 2026-11-12.

**Why.** These files run on no PR lane, so dev went red in them unseen, and every covering run has to subtract the same 27 by hand, which is how a new regression hides.

**The reds** (all under `Tests/Backup_Recovery/`):
- `test_created_persona_subtree_rollback.py`: the module fixture `native_persona_negatives` (`:421`) feeds the parametrized `test_native_persona_mapping_refuses_unproved_graph` (`:441`): 19 errors and 1 failure.
- `test_restore_plan.py:790-791`: `test_shipped76_archive_stages_installed_schema_without_changing_source[False|True]` (TASK-40132 also saw both red at its base 42fc21939d).
- `test_visual_identity_restore.py` (1), `test_inventory_unused_owners.py` (3) and `test_builtin_later_snapshot.py` (1).

**The flaky watchdog.** `test_inventory_unused_owners.py:163` (inside the `_FRESH` child script, from b5251e9a6e) sets `call_later(20, cancel.set)`. `test_fresh_app_complete_public_capture` (`:192`) then fails with `InterruptedError: cancelled` under xdist load (two TASK-40132 covering runs), and passes alone in about 22 s. A watchdog shorter than a loaded run reports a cancellation where nothing hangs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every node of the five files passes at dev head at `-n 4`, each red first given a named cause (test drift with the commit, or a product defect fixed with a test that was red)
- [ ] #2 test_fresh_app_complete_public_capture's watchdog fires only on a real hang (derived from or replaced by the outer subprocess timeout), and the file passes 3 of 3 at `-n 4` under load
<!-- AC:END -->
