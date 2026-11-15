---
id: TASK-40307
title: "The Backup_Recovery owner suites run on no PR lane, so their reds land on dev unseen"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-12 19:36
updated_date: '2026-11-15 02:25'
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
- [ ] #1 The core owner suites named in the description run in a push-to-dev job (never the fast PR contract), with the job's time recorded from three local runs and its first CI run
- [ ] #2 That push-to-dev job is green on the first dev push after it merges, and its targets are pinned in Tests/CI
<!-- AC:END -->

## Implementation Plan

1. Measure each core owner suite named in the description at -n 4 locally after TASK-40306 (load recorded), and check them against the lane's install set (hypothesis, Pillow).
2. Add one PR job to .github/workflows/derived-artifacts.yml for those suites (its own job, never pr-fast-lane), with a step time budget like the neighbouring steps; move test_detached_layout's single architecture-guards node into it as the whole file.
3. Wire the job into the required aggregate (needs + verdict step) and pin its targets, job shape and exclusivity in Tests/CI.
4. Time the lane's step three times locally with the lane's flags; record the first CI run when the coordinator pushes.

## Implementation Notes

- **Owner decision, 2026-11-12 (AskUserQuestion), verbatim:** "Push-to-dev job only (Recommended)". The option as put to the owner: "Same choice as the library shell tests in round 13. PRs pay nothing, and a red shows on the dev push right after the merge instead of going unseen. Uses about 40 runner-min per dev push." The acceptance criteria were amended the same day from "PR lane" to "push-to-dev job"; #1 and #2 stay open because both need a dev push after this merges (the first CI run's job time, and green on that push).
- **Job.** `backup-owner-suites-dev` ("Backup_Recovery owner suites (push to dev)") in `.github/workflows/derived-artifacts.yml`, after `library-shell-dev` and the same shape: `if: github.event_name == 'push' && github.ref == 'refs/heads/dev'`, `needs: [preflight]`, not in the required check's `needs` and no verdict step. It runs the 13 files the description names whole (`test_complete_roundtrip`, `test_inventory`, `test_inventory_unused_owners`, `test_restore_plan`, `test_created_persona_subtree_rollback`, `test_visual_identity_restore`, `test_builtin_later_snapshot`, `test_later_rollback_credential_ui`, `test_detached_layout`, `test_capture_service`, `test_config_sibling_capture`, `test_core_owners`, `test_operational_owners`) plus `test_publication_crashes.py` (TASK-40310's journal pins). `test_detached_layout`'s actor-pack node stays in `architecture-guards`, so PRs still run it; that job's "each target runs once" check now ignores push-to-dev jobs, where it never runs.
- **Round 1 (superseded):** this was first a required PR job, `backup-owner-suites` (034da9aa87); the owner chose the dev-push job instead.
- **Not serial.** `-n 4` (pytest-xdist added to the fast-lane install line, with hypothesis for `test_restore_plan`). Every file here is sandboxed and does its heavy work in child processes with their own HOME. Serially the persona and builtin rollback files alone would take up to 84 minutes at load ~45 (1263 s wall at `-n 4` times four workers), past any sensible job cap. `--dist loadfile` was tried and dropped: it puts all of `test_builtin_later_snapshot` on one worker.
- **Local job time** (the step's exact arguments, extracted from the workflow, through the wrapper; shared 18-core host): run 1, 2452 s (40.9 min) at load 26-57, 509 passed, 3 skipped; run 2, 2573 s (42.9 min) at load 29-51, 509 passed; run 3, 1836 s (30.6 min) at load 22-49, 508 passed and 1 failed: `test_operational_owners.py::test_run_log_real_bound_writer_is_excluded`, whose child failed a bare assert at its line 63, most likely `assert runtime.pause.drain(time.monotonic()+1)` in `test_runtime_startup_handoff._SCRIPT` (a 1 s drain deadline). It passed in runs 1 and 2 and in the nine-file run, so it stays in the job; proposed follow-up. The persona rollback test is the longest node (574 s call).
- **Budgets.** Step `timeout-minutes: 50`, job 55, from those runs. The first CI runs should set them. ADR-103 gets a dated amendment: no extra PR runner (its counts stay 28 concurrent test runners, 29 with perf-guard, 46 eligible jobs, which `Tests/CI/test_ui_lane_shard_count_statements.py` derives from the required check's `needs`, so it needed no change), and one more job, about 40 runner-minutes, per dev push.
- **Not done here:** the first CI run's job time and the green first dev push need this merged and pushed to dev. Both criteria stay open until then.
- **Deselected (8 nodes, pinned as `BACKUP_OWNER_SUITES_DESELECTED`):** red at 26b2e677dd outside TASK-40306's five files. Cause of each, as far as found (review r14bk-rev1 asked for one line each):
  - `test_inventory.py::test_custom_db_and_disabled_features_are_not_external_or_unused`: `assert 'included' == 'unsupported'` once at load 36-54; it passed on a rerun at load 17-37. Intermittent; cause not investigated.
  - `test_inventory.py::test_fresh_process_discovery_imports_no_bootstrap_or_services_and_changes_nothing`: test drift. Its child's import blocker forbids every `tldw_chatbook.Utils*` module, and since 8bbac5bd20 (2026-08-27, offline tiktoken assets) `tldw_chatbook/__init__.py` imports `tldw_chatbook.Utils.tiktoken_runtime` (reproduced: `AssertionError: forbidden service import: tldw_chatbook.Utils`).
  - `test_inventory.py::test_normal_config_and_pure_resolvers_agree`: reads config in-process in a plain test, and the raw config participant refuses the per-test selector (`RecoveryRequired: raw_source_selection_changed`, the known local profile trap). Still red at load 17-37.
  - `test_core_owners.py::test_factory_discovery_and_schema_policy_are_runtime_independent`: its child asserts `tldw_chatbook.DB.private_sqlite` is never imported; since 79a46b3b4e (2026-11-11) `recovery_workflow_schema.open_installed_schema` opens its in-memory reference through `connect_private_sqlite`, and `schema_policy()` reaches it (found with an import hook). A test pin or a product boundary question.
  - `test_complete_roundtrip.py::test_two_known_profiles_complete_capture_preserves_native_state_and_resumes` and `::test_two_captured_profiles_plan_shared_concrete_roots_for_isolated_restore`: test fixture. It makes each profile folder with `(profile / "custom").mkdir(parents=True, mode=0o700)`, so the selector's parent gets the umask mode (0o755), and `config_participants.companion_guard` refuses it with `config_companion_parent_unsafe`.
  - `test_complete_roundtrip.py::test_two_captured_profiles_restore_and_open_with_native_content` and `test_later_rollback_credential_ui.py::test_f9_later_rollback_requires_explicit_credential_review`: load. The shared `native_package` fixture's `pip install --target` has a fixed 60 s limit. The helper builds and installs a wheel for ten other test files, so it keeps its limit here. At load 17-37 the f9 node got past it and then failed `AssertionError: real exec expected`; cause not investigated.
- **Back in the job (review r14bk-rev1):** `test_config_sibling_capture.py::...[sidebar]` and `::...[custom]` now use the idle wait (`idle=60`, cap 280 s) and passed at load 17-37. `test_operational_owners.py::test_dormant_tamagotchi_path_constructors_have_no_installed_call_sites` was drift from 94e25526d9 (TASK-32905): the footer pet's `ConfigFileStorage()` in `Widgets/AppFooterStatus.py` is the installed `tamagotchi.config` owner's path. The census now allows exactly that one call (by file and constructor, not line), so the SQLiteStorage, JSONStorage and NotesMirror guards stay on.
- **Pins.** `ARCHITECTURE_GUARD_TARGETS` gains the two TASK-40310 journal nodes (`test_publication_crashes.py::test_append_and_flush_handle_each_record_once`, `::test_cached_journal_history_still_refuses_changed_records`), so PRs run them while the file stays whole only in the dev-push job. `Tests/CI/test_ci_queue_pressure_contract.py`: `DEV_PUSH`, `BACKUP_OWNER_SUITE_TARGETS`, `BACKUP_OWNER_SUITES_DESELECTED` and `test_backup_owner_suites_run_on_dev_pushes_only_and_are_never_required` (the dev-push `if` equal to `library-shell-dev`'s, outside the aggregate's needs and verdicts, exact command, install line, step budget inside the job cap, no file listed whole elsewhere, the detached node still in `architecture-guards`). `Tests/CI/test_derived_artifacts_workflow.py`: the job order.
- **Files.** `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_derived_artifacts_workflow.py`, `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`.

**2026-11-15 (round-14 closeout).** Both criteria wait on CI, not on code: AC#1's job time needs the first CI run of `backup-owner-suites-dev`, and AC#2 needs that first dev push after this stack merges to be green. Neither can be met before the merge. The Actor Pack later-rollback tests TASK-40305 left unlaned (`test_created_persona_subtree_rollback.py`) run whole in this job. The 8 deselected nodes are TASK-41116; the drain and builtin time-limit flakes in this job's files are TASK-41117; the persona reopen-child lock flake is TASK-41115.
