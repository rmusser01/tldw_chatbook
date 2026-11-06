---
id: TASK-38989
title: "Architecture census rows and two guards are out of date on dev, and none of the 23 red Architecture files runs on a PR lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:42
updated_date: 2026-11-06 10:17
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
- [x] #1 The 13 nodes pass on dev, each drift either recorded as a reviewed census row or reverted in product code (red before: 13 of 13 fail at 29592ecc9b). Narrowed by the filer's ruling of 2026-11-06: the 2 backup-owner-inventory nodes move to a follow-up coverage review, so 11 nodes
- [x] #2 The ProviderRecord and class-body-helper false positives are fixed in the guards, and each guard still fails on a real hand-armed auto_refresh clock and on a real undefined private helper (negative controls)
- [x] #3 The profile-path walker skips non-runtime Python under Docs/, or that QA snippet stops being a .py file, and the walker still catches a new runtime path (negative control)
- [x] #4 Per-file serial times for the 23 red Architecture files are recorded, and the files run on a PR lane with their added time measured, or the notes record the owner's decision to keep them off
<!-- AC:END -->

## Implementation Plan

1. Re-confirm the 13 reds at base e8eeae7db9 and list each drift in full: several censuses stop at the first cluster or row they find.
2. Revert product drift where the census caught a real defect. Otherwise record a reviewed census row.
3. Guards: the ProviderRecord catalog field and class-body helper scope, each with negative controls. The profile-path walker skips Docs/.
4. Record per-file serial times for the 23 files. Lane the ones that pass and need no git history on the dreams-boundary step (appended at the end), and time the addition 3 times with the step's flags.

## Implementation Notes

**Status: Done. All 11 nodes in the narrowed AC#1 pass. The 2 backup-owner-inventory nodes are excluded by the filer's ruling; see that section.**

**Library export (3 nodes): reverted in product code.**
- `LibraryScreen._run_library_export_via_service` was a stale 97-line duplicate, not a delegator. b5251e9a6e (TASK-32628) put a full body back on the screen. That body carried the lifetime span (`operation` / `run_async`) but lacked task-32232's empty-export copy and artifact-read receipt counts, which live only in the controller. Production called the screen copy, so both task-32232 features had been absent since 2026-09-16.
- The controller body now runs inside the lifetime span, and the screen method is a one-line class-forwarding delegator again. `test_library_export_roundtrip.py` (temporary `bootstrap_profile` mark): at base, `…collecting_nothing_reports_failure` (wrong message) and `…receipt_counts_come_from_the_written_manifest` (`None == 2`) fail; both pass here.
- The User Guide (`library/import-and-export.md`) already describes the restored behaviour, so it needs no change.
- Library export controller ratchet 1492 -> 1501 under the owner waiver (dated row). library_screen.py drops 79 lines.
- The cluster census also had drift that the export failure hid. Reviewed stayer rows:
  - `_project_library_export_canvas`: task-31249's deferred projection (3cee32b3d0).
  - `retain_library_collection_annotation_draft`: a screen `@on` handler that forwards to a differently named controller method (0f69e0ba96).
  - `_library_conversation_visible_link_receipt`: ef610f3a77's receipt projection.

**Guards (AC#2).**
- **Framework clock.** `NON_CLOCK_AUTO_REFRESH_CALLEES = {"ProviderRecord"}` skips the 45 catalog rows. `test_non_clock_auto_refresh_fields_stay_plain_bools` pins `ProviderRecord` as a non-widget dataclass whose `auto_refresh` is a bool. Negative controls:
  - With the exemption removed, the new test and the package node are red.
  - A real `widget.auto_refresh = 0.5` added to a package file is flagged.
  - `SomeWidget(auto_refresh=True)` stays flagged in the new synthetic test.
- **Undefined globals.** A call made directly in a class body now resolves in that class's own namespace (`_scope_level_names`, shared with module scope). Methods, lambdas and comprehensions still never see class names.
  - `test_class_body_helper_resolves_only_in_its_own_class_body` pins both sides, plus a deleted module helper.
  - Negative controls: with class scope disabled, hook_permissions and the new test are red; an over-broad "any enclosing class" check makes the new test red.
  - The stale `API_Endpoint_Sample.py` exemption is dropped; that file is gone.

**Profile paths (AC#3).**
- The walker skips `Docs/` (`NON_RUNTIME_PYTHON_ROOTS`). Negative control: a temporary `tldw_chatbook/_r10a_negctrl.py` reading `config.BASE_DATA_DIR_CLI` is caught.
- `APPROVED_EXCEPTIONS` is stable-sorted by identity, and each rule moves with its comments. It is a pure block move: the ADR-256 group, legacy_catalogs and settings_speech_tts.

**Other census rows, reviewed.**
- **Patch targets.** `test_app_instance_warning.py` now patches `get_subscriptions_db_path` on every app module through `set_app_global`.
- **Persona Buddy boundary.** TASK-32901's six controller logs interpolated the class name into free text. They are now the boundary's fixed `persona_buddy_*` categories, with the class bound as `exception_type` (bounded by a regex, as `preferences.py` does). `test_controller_failure_logging.py` reads the bound field.
- **Buddy log class, regression found by the round-10 filer, fixed.** Moving the six controller logs to fixed categories plus `logger.bind(exception_type=…)` dropped the class from what users see. The only production loguru sink, `Logging_Config._forward_loguru_to_standard`, forwarded only `record["message"]`, so dev's "Persona buddy visual graph unreadable: RuntimeError" became the bare category. My earlier test read `extra` and missed it.
  - **Fix, per the coordinator's ruling.** The forwarder appends a bound `exception_type` as `"{message}: {exception_type}"`, through `_with_bound_exception_type`. It renders only that one key, and only when the value is a plain class name (`[A-Za-z][A-Za-z0-9_]{0,63}`); no other `extra` field reaches the log. This also fixes `preferences.py`'s bound log.
  - `test_controller_failure_logging.py` now captures what users see: loguru, then the real forwarder, then std logging, formatted by `RedactingFileFormatter`. Red before the forwarder change (`assert 'RuntimeError' in 'persona_buddy_visual_graph_unreadable'`), green after.
  - New `Tests/test_persistent_diagnostic_boundary.py::test_forwarder_renders_only_a_plain_bound_exception_type` checks three things:
    - a plain class is appended;
    - a non-class value and any other bound key are not rendered;
    - `redact_log_line`, through `RedactingFileFormatter`, still masks a credential in a forwarded line that carries a class.

    It was red with the forwarder change reverted.
  - **Lanes.** Neither test file runs on a PR lane; no laned test file imports the forwarder.
  - **Regression check.** Buddy, forwarder and sanitizer tests: 20 failed and 394 passed, the same 20 as at HEAD 47dc0fcb94 (RecoveryRequired / environment). The only new node is the forwarder test.
  - **Diagnostic inventory.** It flagged the forwarder's two `std_logger.log` calls, which now pass `text`. Reviewed with `--statements`, then regenerated.
- **Legacy downloader.** `Backup_Recovery/config_adapter.py`, `detached_layout.py` and `inventory.py` name `model_download_dir` only to classify it as an inert external location. None of them reads or writes the directory.
- **Diagnostic inventory.** The Buddy and export log changes moved it, and the full branch run caught that. Every row was reviewed with `--statements` before `--write` (separate commit).

**Excluded from AC#1 by the filer's ruling (2026-11-06): backup owner inventory (2 nodes).**
- The filing saw only the first lines of a truncated diff. At e8eeae7db9 the producer census differs from `backup-recovery-owner-inventory.md` (last edited 2026-10-11) by:
  - **644** added rows;
  - **194** gone rows;
  - **12** changed counts;
  - the SQLite registry has **13** new members and **8** gone.
- New SQLite members: `chat.launch_wake`, `chat.trace_maintenance`, `db.chunking_lab`, `db.workspace_project_index`, `library.legacy_recovery`, `personal_context.interview_drafts`, `personal_context.repository`, `plugins.registry`, `recovery.operations.dreams`, `recovery.operations.guardian`, `tests.store_kernel`, `utils.db_upgrade_notice`, `workflows.local`.
- Gone SQLite members: `recovery.operations.receipts` and seven `settings.*` stores.
- Each new row needs an owner, a classification and a backup cohort. That is a backup-coverage review. Stamping 644 rows with defaults would defeat what the census is for, so the rows are left for a dedicated task.
- Review r10a-rev1 confirmed this is a coverage review that needs owner and cohort decisions. Three example rows show why:
  - `recovery.operations.receipts` left the SQLite policies while `recovery.operations.dreams` and `recovery.operations.guardian` appeared;
  - `tests.store_kernel` is in the production registry;
  - the plugin authority store (`Plugins/authority_store.py`) adds 25 producer rows.
- The coordinator will file a follow-up task to carry that review. The two nodes stay red until it lands.

**Lane (AC#4).**
- **Per-file serial times**, summed `--durations=0`, base e8eeae7db9 at load 6-15 / this branch at load 13-20:

| File | Base | Branch |
| --- | ---: | ---: |
| no_undefined_module_globals | 194.1 | 307.1 |
| profile_owned_path_inventory | 54.3 | 154.2 |
| framework_armed_clock_inventory | 25.1 | 31.7 |
| persistent_terminal_privacy_boundary | 15.9 | 30.0 |
| worker_exclusive_group_inventory | 17.2 | 21.8 |
| timer_path_static_update_inventory | 23.6 | 21.6 |
| progress_widget_clock_guard | 12.8 | 20.6 |
| recovery_entrypoints | 110.5 | 18.9 |
| backup_owner_inventory | 13.2 | 15.6 |
| app_extracted_patch_targets | 12.2 | 13.3 |
| task2062_3 | 1.9 | 2.8 |
| console_wave6_inventory | 1.9 | 2.3 |
| console_wave6_closeout | 1.8 | 2.1 |
| guarded_modal_quit_census | 1.4 | 1.6 |
| the other 9 | ~0 each | ~0 each |

  The 23 files sum to 488.5 s at base and 643.5 s on this branch. Base recovery_entrypoints included 4 × 20 s timeouts.
- **Laned in a new job, `architecture-guards` ("Architecture guard contracts"), per the coordinator's ruling on review r10a-rev1.**
  - The first cut appended the 14 files to dreams-boundary's step. The reviewer projected that AST walks run about 2x slower on CI, taking that job to about 13.5-17.5 min median and about 18.8 of 20 worst-case alongside the other round-10 additions, so the files moved into a job of their own.
  - Job shape: `needs: [preflight]`, the lanes' `if:`, `timeout-minutes: 20`, the fast-lane install line, and one serial step with `--timeout=180 --tb=short`. It sits after `dreams-tracking`, before `derived-artifacts`.
  - Wiring: the job is in the aggregate's `needs`, and a "Require successful Architecture guard contracts" verdict step was added. dreams-boundary is back to its base argv.
  - The 14 files: actor_pack_boundary, app_extracted_patch_targets, bare_chat_screen_shells, console_fleet_ownership, framework_armed_clock_inventory, library_cluster_membership_census, library_export_wiring, persona_buddy_boundary, persona_visual_runtime_boundary, profile_owned_path_inventory, progress_widget_clock_guard, recovery_entrypoints, task2062_3, worker_exclusive_group_inventory.
  - The profile inventory runs as its other 14 node ids, without `test_cli_prints_the_real_source_census_and_enforces_it`, because preflight's census step runs the same CLI. The test itself stays. The repo's lane tests reject `--deselect`, so the node ids are listed explicitly.
  - Review r10a-rev2: preflight runs that CLI without `--print-occurrences`. That test's print-and-empty-stderr check is therefore **unlaned**; it runs only in a full local `Tests/Architecture` run.
  - The lane test now pins the by-node listing as complete. The listed names must equal the file's AST `test_*` functions minus the CLI test. It also asserts there are no class-scoped tests and no parametrize ids. A dummy `test_r10a_dummy` added to the file turned it red.
- **Pins updated.**
  - `ARCHITECTURE_GUARD_TARGETS`, with its own argv pin test, and the job lists in `_assert_required_aggregation` (`test_ci_queue_pressure_contract.py`). Plus 3 continue-on-error mutation cases.
  - `VERDICTS`, the job order, and the disjoint-targets step lists (`test_derived_artifacts_workflow.py`).
  - The 14 lines were dropped from the critical gate's dreams pin.
  - ADR-103's job counts, which `test_ui_lane_shard_count_statements.py` derives from the workflow.
  - Tests/CI: 1215 passed, 1 skipped.
- **Measured.** The job's exact command (27 targets, serial, CI flags), run 3 times:
  - 195.9 s, 176.3 s and 174.3 s of pytest, 217 / 196 / 194 s wall, 152 passed each;
  - load 11-13 / 9-12 / 10-12 on 18 CPUs shared with other agents.
  
  With checkout and install, a CI job of about 5-8 min is expected, assuming the reviewer's 2x CI factor for AST walks.
- **Extra-runner cost.** This is one more runner on every ordinary PR. ADR-103's counts move from 39 to 40 eligible test jobs, and from 24 to 25 parallel runners (25 to 26 with perf-guard). No existing job moves toward its cap.
- **Kept off, owner decision needed:**
  - `test_no_undefined_module_globals.py`: 2,943 parametrized nodes, which mostly cost per-test conftest overhead. 194-307 s serial; with it in, the additions run took 754 s at load ~34. Proposed follow-up: collapse it to one test, then lane it.
  - 4 files that `git show` frozen revisions and would fail on this job's shallow checkout: console_realtime_controller_boundary, console_review_selection_controller_boundary, console_wave6_inventory, console_wave6_closeout_inventory.
  - Red files owned elsewhere: timer_path_static_update_inventory (TASK-35059/36191) and persistent_terminal_privacy_boundary (TASK-38111). backup_owner_inventory is red pending the review above.
  - guarded_modal_quit_census already runs on dreams-boundary.

**Evidence.**
- Red before: all 13 nodes failed in the serial base run (40 failed, 4106 passed, 979 s).
- Full serial Architecture run on this branch: 7 failed, 4146 passed (1414 s, load ~15):
  - the 2 backup nodes;
  - persistent_terminal_privacy (TASK-38111);
  - the 3 timer-census nodes (TASK-35059/36191);
  - the diagnostic inventory, now regenerated and verified.

AC#1 is met as narrowed by the filer's ruling. AC#4 is met: the 14 files run in the new job, and the 9 kept off are listed above as owner decisions.
