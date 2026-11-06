---
id: TASK-38111
title: >-
  Tests: several UI and Chat test files error at setup with RecoveryRequired
  raw_source_selection_changed when run on their own
status: Done
assignee:
  - '@claude'
updated_date: '2026-11-06 22:54'
created_date: '2026-10-30 05:56'
labels:
  - tests
  - test-infra
  - follow-up
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-33621.8/.9 review on origin/dev ff815f8559. Run one file per command, as agents and developers do locally, and every test in these files errors at setup: Tests/UI/test_trace_export_ui.py, Tests/UI/test_console_inspector_workflow.py, Tests/UI/test_console_inspector_focus_carriers.py, Tests/UI/test_non_obscuring_focus_contract.py and Tests/Chat/test_trajectory_capture.py. They pass when given a bootstrap_profile marker, so their tests can only be checked locally through a marked copy, and a regression in them is invisible to a per-file run.

For test_trace_export_ui.py the traceback is: Tests/UI/conftest.py _disable_model_catalog_refresh monkeypatches tldw_chatbook.app.TldwCli, which imports tldw_chatbook.app for the first time inside the per-test profile; the app module runs load_settings() at import, and config admission refuses with RecoveryRequired('raw_source_selection_changed'). This is the same class as the eager imports Tests/conftest.py already makes (Chunking, RAG config profiles, model capabilities) and the per-file list in that fixture. The other files were reported with the same error and need their own traceback read.

**Added 2026-11-02 (dev 88af9f9379).** `Tests/Wizards/test_first_run_setup_integration.py` has the same error and belongs on this list. Run on its own through the scratch-profile wrapper, with `-n 4` or serially, 17 of its tests error at setup with `RecoveryRequired: raw_source_selection_changed`. Each one errors in the file's `temp_config` fixture (:25), which redirects `TLDW_CONFIG_PATH` and then calls `load_cli_config_and_ensure_existence(force_reload=True)`. That is a different path from test_trace_export_ui.py's import-time `load_settings()`. b68d242b4d gave `TestWizardAtomicProviderHandoff` its own fixture, which brought the count down from 31 on 2026-10-31. The file's 6 assertion failures are a separate product regression, TASK-38642.

**Added 2026-11-05 (dev 29592ecc9b, recorded by TASK-38756).** `Tests/Prompt_Management/test_prompt_improvement_service.py` belongs on this list. Run on its own through the scratch-profile wrapper with `-n 4`, 49 tests fail and 3 error, every one with `RecoveryRequired: raw_source_selection_changed` (`raw_participants.py:307`); its own assertions all pass. Two paths, both read from the file's own tracebacks: the 3 setup errors come from its `scratch_cli_config` fixture (:55-67), which redirects `TLDW_CONFIG_PATH` and calls `config.load_settings(force_reload=True)` (the wizard file's shape); the 49 failures come from a config read inside the code under test, `prompt_improvement_prompts.py:86` -> `Internal_Prompts/resolver.py:147` -> `config.get_cli_setting` -> `load_cli_config_and_ensure_existence`. With a temporary module-level `bootstrap_profile` marker the 49 pass and the 3 fixture errors remain.

**Added 2026-11-06 (dev e8eeae7db9, recorded by TASK-38947).** `Tests/Chat/test_console_chat_controller.py` belongs on this list, through a third path: the code under test, as in the prompt-improvement case. A plain test's send reaches hook admission, where `hook_admission_reason` (`tldw_chatbook/Chat/console_automatic_submission.py`) calls `read_hooks_config_snapshot()`. In the per-test sandbox that raises `RecoveryRequired('raw_source_selection_changed')`; printing the swallowed exception showed it. The broad `except` turns it into the refusal "Hooks unavailable; review or disable hooks before sending." The red nodes are `test_blocked_send_persists_no_durable_record` and `test_accepted_send_persists_the_deferred_user_echo`, which TASK-38947 had filed as fake drift, and the `TestUserPromptSubmitHooks` class (5 tests: `test_block_rejects_send_without_assistant_row`, `test_stdout_injected_as_hook_origin_system_row`, `test_exit_two_shorthand_blocks_send`, `test_wake_notice_fires_nothing`, `test_hook_crash_send_proceeds`). The two TASK-485 nodes pass under a temporary `@private_profile_test` marker, even with the old fake. Under a temporary class-level `bootstrap_profile` marker, 4 of the 5 class tests pass. `test_wake_notice_fires_nothing` then fails on a separate error, `PermissionError: agent-wake sends require coordinator-issued wake authority`. Whole-file runs of this file at `-n 4` had about 200 more failures. I did not read each traceback; the ones I sampled have the same signature.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each listed file passes, or fails only on its own assertions, under the plain one-file command on a clean checkout
- [x] #2 The fix does not bind config to the developer's real profile (the conftest's documented trap)
- [x] #3 The cause of each listed file is confirmed from its own traceback, not assumed from test_trace_export_ui.py
<!-- AC:END -->

## Implementation Plan

1. Run every listed file alone through the scratch-profile wrapper on the base (dev 2684e5453f), plus the files the round-11 brief adds (Tests/RAG_Search/test_keyword_leg_*.py, Tests/Library/test_library_ingest_runner.py); read each file's own tracebacks (AC#3). Files green on the base: run them at the filing commit ff815f8559 too, and find the commit that greened them.
2. Fix each red file with the repo's existing profile seams, in test code only (no production change, no change to `RecoveryRequired` admission):
   - tests that read config but select no profile of their own: the `bootstrap_profile` marker (the shared, per-test-restored collection-time sandbox profile);
   - fixtures that redirect `TLDW_CONFIG_PATH` and force a reload: `Tests/Backup_Recovery/config_test_support.select_config_source`, binding the module namespaces that hold config imports.
3. Re-run each file alone (red-before / green-after table), a broad `-n 4` regression set on head and on fu-base-r11 (Tests/UI, Tests/Chat, Tests/Wizards, Tests/Prompt_Management, Tests/RAG_Search, Tests/Library, as practical), Tests/CI, preflight and bootbudget.
4. Prove AC#2 with a wrapper run of the fixed files whose scratch HOME stays empty of profile paths and whose real-profile guard records no refusal.

## Implementation Notes

**Approach.** The config source binds, for the whole process, to the profile its first guarded read selected (`raw_participants._register_raw_participant`); `Tests/conftest.py`'s eager imports make that the collection-time sandbox profile. A plain test's per-test `TLDW_CONFIG_PATH` redirect then makes every later config read fail closed with `raw_source_selection_changed`. That refusal is production behaviour and is unchanged. Each file now uses one of the two profile seams the suite already has, in test code only:
- `pytestmark = pytest.mark.bootstrap_profile` where the tests, or the code they drive, read config but select no profile of their own: `Tests/UI/test_console_inspector_focus_carriers.py`, `Tests/Chat/test_trajectory_capture.py`, `Tests/Chat/test_console_chat_controller.py`, `Tests/Prompt_Management/test_prompt_improvement_service.py`, `Tests/Library/test_library_ingest_runner.py` and the five `Tests/RAG_Search/test_keyword_leg_*.py` files.
- `Tests/Backup_Recovery/config_test_support.select_config_source` in the two fixtures that redirected `TLDW_CONFIG_PATH` and force-reloaded: the wizard suite's `temp_config` (binding `provider_setup_persistence`, whose module-level writer import otherwise refuses the new profile: without it 5 tests fail with `fully_applied` False, measured) and the prompt suite's `scratch_cli_config` (the resolver imports config lazily, so nothing to bind).

No production file, conftest, `Tests/private_profile.py`, app_factory or run.sh change. The eager `Chunking` import in `Tests/conftest.py` is not the cause: without it the first test's own profile would bind the source and every later test in the worker would be refused instead.

**Causes, each from the file's own traceback (AC#3).** One file per command through run.sh at `-n 4`: "Before" on the read-only base worktree (dev 2684e5453f), "After" on this branch. Both columns were run twice, before and after a machine reboot, with identical counts. A serial run (`-p no:xdist`) of the eleven edited files on this branch gave the same After counts.

| File | Before (base) | After (head) | Cause read from its traceback |
|---|---|---|---|
| UI/test_trace_export_ui.py | 20 errors at ff815f8559; 23 passed on base | unchanged file, 23 passed | `Tests/UI/conftest.py:193` lazy app import ran `app.py:693 APP_CONFIG = load_settings()`; 36e0f56c7c (TASK-36945.3) removed that read |
| UI/test_console_inspector_workflow.py | 35 errors at ff815f8559; 35 passed on base | unchanged file, 35 passed | same as above |
| UI/test_non_obscuring_focus_contract.py | 98 errors at ff815f8559; 98 passed on base | unchanged file, 98 passed | same as above |
| UI/test_console_inspector_focus_carriers.py | 14 failed, 1 passed | 15 passed | `_build_test_app` -> `app_factory.build_test_app_config` -> `load_settings()` (14/14) |
| Chat/test_trajectory_capture.py | 10 failed, 22 passed (one hit the 300 s timeout) | 31 passed, 1 failed (own assertion) | `ConsoleChatController.__init__` and `admitted` reads; hook admission's swallowed refusal ("Hooks unavailable") |
| Wizards/test_first_run_setup_integration.py | 24 passed, 17 errors | 41 passed | `temp_config:25` force reload under the redirect (17/17) |
| Prompt_Management/test_prompt_improvement_service.py | 49 failed, 53 passed, 3 errors | 105 passed | `scratch_cli_config` redirect (3); `improve()` -> Internal Prompts resolver `get_cli_setting` (49) |
| Chat/test_console_chat_controller.py | 206 failed, 159 passed | 330 passed, 35 failed (own assertions) | hook admission `read_hooks_config_snapshot` ("Hooks unavailable") and controller config reads; cascades (unknown session, StopIteration) |
| RAG_Search/test_keyword_leg_chacha.py | 13 failed | 13 passed | `RAGConfig.__post_init__` -> `default_chroma_persist_directory` -> `get_user_data_dir` |
| RAG_Search/test_keyword_leg_db_resolution.py | 11 failed | 11 passed | same |
| RAG_Search/test_keyword_leg_prompts.py | 14 failed, 2 passed | 16 passed | same |
| RAG_Search/test_keyword_leg_pushdown.py | 12 failed, 10 passed | 22 passed | same |
| RAG_Search/test_keyword_leg_tiered_merge.py | 22 failed | 22 passed | same (one via `MediaDatabase.__init__`) |
| Library/test_library_ingest_runner.py | 86 failed, 70 passed, 1 skipped | 154 passed, 2 failed (own assertion), 1 skipped | `_resolve_ingest_backend` `get_cli_setting` (75), Parakeet source service, parse-pool sizing |

The 38 remaining red nodes (35 + 1 + 2) each fail alone in their own process the same way, on stale assertions or fakes the wall had hidden (for example `got an unexpected keyword argument` in test-defined fakes, `'Accepted turn ownership changed before settlement.' == 'Session closed.'`, `error_detail` missing df22692d70's two diagnostic keys). One of them, `test_accepted_send_survives_prompt_history_maintenance_refusal`, still raises `raw_source_selection_changed`, from its own `raw._raw_participant(history)` call: since 2bce8ffa0f the guard derives the prompt-history path from the bound config's data directory, which the test's `default_prompt_history_path` stub no longer controls. That holds under any profile, so it is a stale test, not this redirect. All 38 are filed under follow-ups, not fixed here.

**AC#2 evidence.** A run.sh run of all 14 files with a fresh scratch HOME and a throwaway per-test probe (`-p`, never committed; it compared path strings and opened nothing): 955 tests, the scratch HOME still held only the five directories run.sh creates, 0 real-profile guard refusals, and for every one of the 955 tests the config source was bound either to the bootstrap profile under `TLDW_TEST_CONFIG_ROOT` (921) or to a fresh source under the run's basetemp (34). 0 resolved or bound under the real home's profile roots. The run itself: 38 failed (exactly the residual nodes above), 916 passed, 1 skipped.

**Regression (head vs base, `-n 4`, same command on both trees, run concurrently at load 40-150).**
- Tests/Wizards, Tests/Prompt_Management, Tests/RAG_Search and Tests/Library: 7106 nodes on both trees. Base 6676 passed / 394 failed / 22 errors / 14 skipped; head 6903 / 187 / 2 / 14. Files whose counts changed: the eight edited files in these directories (all better), plus two unedited files with flaky nodes (test_embeddings_performance 2 -> 1 failed, test_first_run_setup_wizard 6 -> 5 failed). One node was worse on head, `test_first_run_setup_wizard.py::test_ctrl_n_recovers_hidden_widget_focus_and_stops_at_summary` (unedited file); alone it failed 2 of 9 runs on head and 2 of 9 on base.
- Every Tests/UI, Tests/Chat, Tests/integration and Tests/Backup_Recovery file that imports from an edited module, plus the edited and base-green listed files there (23 files): 1718 nodes on both trees. Base 676 passed / 1032 failed / 10 errors; head 872 / 836 / 10. Changed files: test_console_chat_controller, test_trajectory_capture and test_console_inspector_focus_carriers (all better), plus two unedited files that each had one fewer failure on head (test_console_fleet_wake_safety, test_console_runtime_shutdown). One node was worse on head, `test_console_durable_progress.py::test_durable_discard_contention_keeps_mounted_reads_and_heartbeat_responsive[dispose]` (unedited file); alone it passed 3 of 3 runs on both trees. The 836 reds that remain are in files outside this task's list and are equally red on base, mostly `test_library_shell.py` (779 `RecoveryRequired`).
- Tests/CI at `-n 4`: 1215 passed, 1 skipped. Boot budget ratchets (bootbudget.sh): 32 passed. Preflight passed.

No new test was added. The red evidence is each file's Before column: every edited file was red on base with `RecoveryRequired` and is green after, apart from its own assertions.
