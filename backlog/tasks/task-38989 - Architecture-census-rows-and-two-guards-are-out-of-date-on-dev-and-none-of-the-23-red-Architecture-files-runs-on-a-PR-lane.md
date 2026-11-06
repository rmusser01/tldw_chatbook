---
id: TASK-38989
title: "Architecture census rows and two guards are out of date on dev and none of the 23 red Architecture files runs on a PR lane"
status: To Do
created_date: 2026-11-06 01:42
labels:
- follow-up
- tests
- architecture
- ci
priority: medium
references:
- Tests/Architecture
- tldw_chatbook/provider_registry.py
- tldw_chatbook/Agents/hook_permissions.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b on 2026-11-05: a serial `Tests/Architecture` run had 41 failures in 1044 s at load 7-10. This task covers 13 of them. Every one is a census or inventory out of date, or a guard flagging something it should not. No PR lane runs any of the 23 failing files, and the branches due to merge before this one do not touch them.

**Library export cluster (3 nodes: `test_library_cluster_membership_census.py`, and `test_library_export_wiring.py` ×2).**
- The cluster census has an unaccounted `_project_library_export_canvas`.
- Both export-wiring nodes fail on `_run_library_export_via_service`, which is neither a delegator nor class-forwarding.

**Census rows out of date, and guard false positives (7 nodes: the framework armed-clock inventory, `test_no_undefined_module_globals.py` ×2, `test_app_extracted_patch_targets.py`, `test_persona_buddy_boundary.py`, and `test_backup_owner_inventory.py` ×2).**
- The framework clock guard flags 45 `ProviderRecord(…auto_refresh=…)` rows. That is a catalog field (`tldw_chatbook/provider_registry.py:184`), not a clock.
- The undefined-globals guard flags `_bind_store_lock`. That is a class-body helper (`tldw_chatbook/Agents/hook_permissions.py:230`), called at `:284` and deleted at `:285`. The same guard keeps an exemption row for the deleted `Local_Ingestion/API_Endpoint_Sample.py`.
- `test_app_extracted_patch_targets`: `Tests/UI/test_app_instance_warning.py:165` patches `app.get_subscriptions_db_path` on its own.
- `test_persona_buddy_boundary.py:153`: a call now has 2 arguments where the test expects 1.
- The backup owner inventory (2 nodes) shows producer census drift. `chatbook_creator` writes went from 1 to 2, and `_connect_registered_sqlite` from 1 to 2. The SQLite registry also gained `personal_context.interview_drafts`, `plugins.registry`, `library.legacy_recovery`, `recovery.operations.guardian` and `recovery.operations.dreams`.

**Profile and legacy-downloader inventories (3 nodes: `test_profile_owned_path_inventory.py` ×2, and `test_task2062_3_legacy_downloader_retirement.py`).**
- The profile-path exception rules are out of sort order.
- The profile-path walker (`rglob("*.py")`, `:313`) fails to parse a tracked QA snippet, `Docs/superpowers/qa/2026-11-01-prompt-mix-ins/ci-rebase/hosted-5a8/response-rules/original-consume-visible-handoff.py`, with an IndentationError. That file came in with bffa2bffed on 2026-11-02.
- The task2062_3 inventory has new readers: `Backup_Recovery/config_adapter.py`, `inventory.py` and `detached_layout.py`.

**Not in this task.**
- The 3 timer-census nodes belong to TASK-35059 and TASK-36191.
- The RecoveryRequired node belongs to TASK-38111.
- The guarded-modal quit census is fixed by TASK-38747.

**Lane cost.** The whole directory takes 1044 s serially for 4146 tests. The dreams-boundary step has about 13 minutes free under its 20-minute cap, and the PR Fast Lane is at its cap (TASK-38716). Laning these files therefore needs per-file times and an owner decision.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 13 nodes pass on dev, each drift either recorded as a reviewed census row or reverted in product code (red before: 13 of 13 fail at 29592ecc9b)
- [ ] #2 The ProviderRecord and class-body-helper false positives are fixed in the guards, and each guard still fails on a real hand-armed auto_refresh clock and on a real undefined private helper (negative controls)
- [ ] #3 The profile-path walker skips non-runtime Python under Docs/, or that QA snippet stops being a .py file, and the walker still catches a new runtime path (negative control)
- [ ] #4 Per-file serial times for the 23 red Architecture files are recorded, and the files run on a PR lane with their added time measured, or the notes record the owner's decision to keep them off
<!-- AC:END -->
