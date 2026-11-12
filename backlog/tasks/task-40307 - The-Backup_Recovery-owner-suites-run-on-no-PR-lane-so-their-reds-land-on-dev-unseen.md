---
id: TASK-40307
title: "The Backup_Recovery owner suites run on no PR lane, so their reds land on dev unseen"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- tests
- backup
- ci
- lanes
priority: medium
references:
- .github/workflows/derived-artifacts.yml
- .github/workflows/backup-recovery-qualification.yml
- Tests/CI/test_ci_queue_pressure_contract.py
- Tests/Backup_Recovery/test_complete_roundtrip.py
- Tests/Backup_Recovery/test_detached_layout.py
dependencies:
- TASK-40306
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (census stream fix round). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** 341 of the 372 `Tests/Backup_Recovery/test_*.py` files run on no PR lane. `backup-recovery-qualification.yml` and `native-credential-backup.yml` are `workflow_dispatch` only, `test.yml` runs on pushes to main, and nightly-deep last ran 2026-09-27. The previous task's 27 reds are the result.

**What runs today.** 26 files whole (in `derived-artifacts.yml`, `eval-recovery-regressions.yml:94-102` and `perf-guard.yml`) and 5 by single nodes only: `test_detached_layout` (1 node, `derived-artifacts.yml:1275`), `test_chachanotes_native76_compatibility` (1), `test_canvas_policy_worker` (3), `test_response_rules_restore` (3), `test_unsaved_editors` (4). `plugin-qualification.yml:21-24` names `test_capture_service`, `test_config_sibling_capture`, `test_core_owners` and `test_operational_owners` only as path filters and never runs them.

**On no PR lane**, among others: `test_complete_roundtrip`, `test_inventory`, `test_inventory_unused_owners`, `test_restore_plan`, `test_created_persona_subtree_rollback`, `test_visual_identity_restore`, `test_builtin_later_snapshot`, `test_later_rollback_credential_ui`, the rest of `test_detached_layout`, and the four plugin path-filter files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The core owner suites named in the description run on a PR lane (never the fast PR contract), with the lane's job time recorded from three local runs and its first CI run
- [ ] #2 That lane is green when it merges, and its targets are pinned in Tests/CI
<!-- AC:END -->
