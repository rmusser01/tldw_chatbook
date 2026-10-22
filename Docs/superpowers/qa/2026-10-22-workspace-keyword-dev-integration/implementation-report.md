# Latest-dev keyword integration fix report

**Final writer handoff:** product source `19d38fd32ed9009d9fd7c9ffb04e38fd989d370d`; clean worktree; no task-owned test/query process running. Source changes are complete. Parent retains Backlog closure, final scoped review, CI/threads/queue and merge ownership. The sections below preserve each earlier checkpoint and its later disposition.

Final evidence: focused10 passed/1 warning363.34s; e161 covering2 failed/20 passed/2 warnings617.23s; affected correction5 passed/1 warning137.63s supersedes only the two setup failures and verifies the new ownership race; final query1 passed/1 warning244.90s. These are distinct source-qualified runs, **not an aggregate all-green suite**. All earlier RED, fixture refusal and child-pass/outer-timeout outputs remain retained.

Final canonical lexical p95 values are164.260-273.264ms, all below500ms. Expanded admitted misses remain644.722ms nested common,549.382ms nested rare and617.648ms wide common; no expanded-target PASS, whole-tool SLA or improvement claim. All42 native Capture receipts and40 public registry retirement returns pass; final service has0 searches/0 operations. Whole public p95 including probe retirement bookkeeping:1948.863ms nested/2443.248ms wide. Historical full-build/platform/headless limits remain.

Final key source SHA256 values: service `aa764965e2a0d87737d47e1f9f5411d123af64e06e471eb0c0e27d346c6cd648`; index DB `2f5516cc05746296d563e92afb6fa9d6eb324e991f9b742455c0c6081292ea1a`. All24 source hashes, exact query/probe identities, command/full output and raw-summary hashes are retained in integration-idle-source.json/integration-idle-summary.json. Scoped source and test lint/format/AST/whitespace checks pass, with3 inherited shared controller-test lint findings and0 introduced. Parent's three corrected guards passed separately.

## Source and ownership

Commit: `e16100e2b3aea1e1da3d6d73ecbcbefebb135a80`, branch `codex/workspace-project-keyword-index`. Rebased predecessor: `99857a64e7f59ecb7cdc1a1a008a208855d1dd23`, dev base: `2a5afa03b31f60e00bdcd10a161e3e351fad023b`.

The exact 30 authorized paths became 29 Git files because of the ADR rename. `integration-owned-paths.json` and `integration-source-manifest.json` retain the path list and SHA256 values, validated against every dirty tracked/untracked path and strict UTF-8 reads before staging. The worktree was clean after commit. The original checkout, fixed corpora/profile, closed SDD ledger and historical raw receipts remain untouched.

Changes:

- History eviction now removes a natively retired old run only when its supervisor is absent or physically dead. The existing sixteen-record diagnostic tail remains; live tails retain shutdown join records beyond it. Existing `aclose` joining and repeated-cancellation shielding are unchanged. The actual-thread regression holds the first supervisor after native retirement, joins twenty later supervisors, verifies retention, cancels close twice and requires physical exit before cancellation propagates.
- The unmerged project decision and active links/task governance move to ADR-241. Landed MCP/Settings ADR-240 is unchanged. Historical raw receipts, hashes, chronological rulings and source-era ADR-240 labels are preserved with renumbering context. Parent reopened TASK-36006 and amended AC12/13 plus plan22 before source changes; this writer has not marked it Done.
- Three DB timestamp writers use the shared canonical millisecond UTC `Z` helper, `utc_now_iso`, under existing ADR-173. No schema/migration or planner change. Old offset timestamps remain readable. The new real-store test reaches publication, `clear_revoked` and `clear_owned_namespace`.
- The existing `directories_eligible` EXPLAIN test now explicitly asserts `sqlite_stat1` is absent. One plan-pinned census row points to that exact test, which also rejects a temporary B-tree for the production frontier query.
- The card worker W002 finding is a reviewed static false positive: after its sole await, it rechecks mounted state and the visit/request/workspace/binding token. All remaining lookups and helpers execute synchronously without suspension or subtree replacement. Its worker already sets `exit_on_error=False`. One reviewed four-site census row records this reasoning; no UI code or broad census rewrite.
- Parent inspected the diagnostic inventory delta: only controller 69->71 and registry 12->13 fixed-literal debug statements/digests changed. No interpolated content, path, secret, endpoint or sink change. Included the exact authorized inventory; parent receipt: `derived-guards/diagnostic-refresh-verified.json`.

## Verification and adverse evidence

Original M1 RED: `m1-red.txt`, **1 failed in 231.75s** (child 221.32s), because the actual first supervisor record was pruned after twenty later runs. Child log/XML and hashes are retained in `m1-red-child/` and `m1-red-child-retention.json`. Never rerun or rewrite this RED.

The first fixed-source invocation hit the outer Windows pytest hard timeout: exit1, elapsed311.37s, so four later controls never ran. The actual regression child completed all assertions and passed **1 case, 1 warning in295.06s** (call286.89s/setup7.55s). Its log/XML and hashes remain in `m1-green-timeout-child/` and its retention receipt. Concurrent root stdlib scans qualify load, without proving the exclusive timeout cause. Exact task process inspection found no remaining matching Python child. No assertion or deadline changed. The outer timeout is not an aggregate PASS.

After all root guards retired, the final focused invocation passed **10 cases, 1 warning in363.34s**, exit0, elapsed368.20s. The M1 outer node took148.61s; normal history137.61s; incremental refresh32.48s; native DB closure/repeated cancellation17.69s; failed supervisor dispatch16.08s. The actual DB plan, three timestamp writer cases and revoke-before-cleanup control also passed. Exact command, hashes, cwd/start/exit are in `integration-focused-command.json`; full output is included below and in `integration-focused.txt`. Private child log/XML copies and SHA256 receipts are retained under `integration-focused-children/` and `integration-focused-child-retention.json`.

Timestamp first RED (`timestamp-red.txt`) has two meaningful shape failures and one legacy-owner fixture refusal before cleanup's write. After correcting only the fixture to issue a native-v1 binding token through existing `claim_job`, `timestamp-red-corrected.txt` has **three meaningful stored-shape failures in7.46s**. All failed on the actual noncanonical microsecond/+00:00 value before the three helper substitutions. Both adverse outputs remain.

Scoped Ruff check on five owned Python files, formatting on four non-docstring-only Python files, AST/in-memory compilation and `git diff --check` passed; exact commands/full logs are in `integration-final-static.json` and its log files. Initial `python -m ruff` failures (missing worktree-local binary) remain in `integration-static-1/-2.txt`; the existing shared executable needed no installation. Initial assertion-wrapping formatting failure remains in `integration-ruff-2.txt`; formatting after test retirement preserves the regression AST (`integration-test-format-equivalence.json`).

Parent reran only the three corrected source guards: index plan7.411s, Textual71.464s and timestamp39.941s all exit0, with raw bytes/commands in `derived-guards/corrected-guards-summary.json`. The earlier capture encoding failure and other guard qualifications remain separate evidence. No blanket inventory regeneration or all-green whole-suite claim.

## Pending covering and query qualification

The single authorized 22-case covering invocation is running on exact commit e16100e2b3, session41346. Exact command/hashes: `integration-cover-command.json`; full output: `integration-cover.txt`. Its result remains pending.

Prepared, not executed: `test_integration_idle_latency.py`, copied from the preserved final-fix probe with only its output JSON redirected into the current integration directory. It reuses the fixed nested/wide20k caches and stable profile, preserving five warmups plus fifty common/rare/miss samples per shape, twenty public native Capture marker calls per shape, code-issued authority and actual registry/process/pipe retirement assertions, exact counts, the canonical500ms gate and existing900s test bound. Expanded admitted timing remains separate, including its historical accepted miss. `integration-query-proposal.json` records both probe hashes and function AST mapping: from original3b616994e6 to currente16100e2b3, only service `_retire` and the three DB timestamp writer functions differ; all query functions are identical. Execute once only after cover/tests/renderers/guards physically retire and parent gives the handshake. No corpus build, copy, relocation or old receipt overwrite. Historical build durations do not become current-source build evidence.

Parent owns PR3204, fresh review, Backlog closure, all issues, CI/current-head Qodo/threads/queue and merge. Qodo billing currently blocks review; parent is obtaining human restoration or an explicit waiver. No automatic gate bypass. Semantic implementation starts only after actual keyword merge. No push/PR/merge/network/provider/embedding/install/full-suite/corpus-build action by this writer. Historical timing/adverse/warning/Windows/headless qualifications remain.

## Exact focused command metadata

```json
{
  "command": [
    "C:\\Users\\GDesktop-1\\.codex\\worktrees\\workspace-project-guidance\\tldw_tui\\.venv\\Scripts\\python.exe",
    "-m",
    "pytest",
    "Tests/Workspaces/test_project_index_service.py::test_history_keeps_live_retired_supervisor_until_cancelled_close_joins",
    "Tests/Workspaces/test_project_index_service.py::test_completed_refresh_history_is_bounded",
    "Tests/Workspaces/test_project_index_service.py::test_shutdown_retains_lifetime_through_native_db_closure_and_repeated_cancel",
    "Tests/Workspaces/test_project_index_service.py::test_failed_supervisor_dispatch_waits_for_started_writer_retirement",
    "Tests/Workspaces/test_project_index_service.py::test_incremental_refresh_retires_deleted_and_renamed_membership",
    "Tests/DB/test_workspace_project_index_db.py::test_scan_unit_atomic_receipts_replay_and_indexed_attempt_frontier",
    "Tests/DB/test_workspace_project_index_db.py::test_index_timestamp_writers_store_canonical_utc",
    "Tests/DB/test_workspace_project_index_db.py::test_revoke_is_visible_before_cleanup_and_old_cleanup_cannot_erase_enable",
    "-q",
    "--tb=short",
    "--show-capture=no",
    "--no-showlocals"
  ],
  "source_sha256": {
    "tldw_chatbook/Workspaces/project_index/service.py": "d0943ea7236310fd7ab6d585118d43f769f578198dcdd9e777941fb92ed7ed82",
    "Tests/Workspaces/test_project_index_service.py": "4089b955152ddf45b718e0df302d4e919b4890712c0f6faabc280b0f5f488e6c",
    "tldw_chatbook/DB/Workspace_Project_Index_DB.py": "2f5516cc05746296d563e92afb6fa9d6eb324e991f9b742455c0c6081292ea1a",
    "Tests/DB/test_workspace_project_index_db.py": "1bcdc71b2bc92839a8ff711d7e9aae873c168ad1e8e5d87c0cd941c1b7921260",
    "tldw_chatbook/DB/Workspace_DB.py": "ee1ffab98e0026f7137dbb2a7cc2b51562bf15c998c725b55a9d257ee2480485"
  },
  "started_unix": 1792647678.9082391,
  "cwd": "C:\\Users\\GDesktop-1\\.codex\\worktrees\\workspace-project-guidance\\tldw_tui",
  "expected_cases": 10,
  "isolated_from_root_guards": true,
  "exit_code": 0,
  "elapsed_seconds": 368.19511437416077
}
```

## Full focused output

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Lib\site-packages\pytest_asyncio\plugin.py:208: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture caching scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future. Valid fixture loop scopes are: "function", "class", "module", "package", "session"

  warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET))
..........                                                               [100%]
============================== warnings summary ===============================
Tests/Workspaces/test_project_index_service.py::test_history_keeps_live_retired_supervisor_until_cancelled_close_joins
  C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Lib\site-packages\pydantic\fields.py:1076: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'exclude_if'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.11/migration/
    warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================ slowest 25 durations =============================
148.61s call     Tests/Workspaces/test_project_index_service.py::test_history_keeps_live_retired_supervisor_until_cancelled_close_joins
137.61s call     Tests/Workspaces/test_project_index_service.py::test_completed_refresh_history_is_bounded
32.48s call     Tests/Workspaces/test_project_index_service.py::test_incremental_refresh_retires_deleted_and_renamed_membership
17.69s call     Tests/Workspaces/test_project_index_service.py::test_shutdown_retains_lifetime_through_native_db_closure_and_repeated_cancel
16.08s call     Tests/Workspaces/test_project_index_service.py::test_failed_supervisor_dispatch_waits_for_started_writer_retirement
2.48s call     Tests/DB/test_workspace_project_index_db.py::test_scan_unit_atomic_receipts_replay_and_indexed_attempt_frontier
2.28s call     Tests/DB/test_workspace_project_index_db.py::test_revoke_is_visible_before_cleanup_and_old_cleanup_cannot_erase_enable
1.65s call     Tests/DB/test_workspace_project_index_db.py::test_index_timestamp_writers_store_canonical_utc[clear_revoked]
1.54s call     Tests/DB/test_workspace_project_index_db.py::test_index_timestamp_writers_store_canonical_utc[clear_owned_namespace]
1.18s call     Tests/DB/test_workspace_project_index_db.py::test_index_timestamp_writers_store_canonical_utc[publish]

(15 durations < 1s hidden.  Use -vv to show these durations.)
10 passed, 1 warning in 363.34s (0:06:03)

```

## Cover result and reviewed ownership correction

The single broader invocation on e16100e2b3 retired with **2 failed, 20 passed, 2 warnings in617.23s**, exit1, elapsed625.28s. Both retry params stopped during controller construction before assertions: real gateway complete_auxiliary activates compaction config reads, while isolate_test_environment redirects TLDW_CONFIG_PATH after the module-level config participant was installed. raw_participants correctly refuses that re-selection (raw_source_selection_changed). The twenty other cases passed, including all Library combinations, mounted themes/status, app shutdown and other dev seams. Full command/output remain integration-cover-command.json/integration-cover.txt; thirty copied child receipts and hashes are retained. This is not an aggregate all-green run.

Parent approved only bootstrap_profile on the two-param retry wrapper, with a causal comment, preserving pytest's isolated collection-time profile. No shared fixture/config/controller change or recovery-gate bypass. The later affected run reaches and passes both complete retry behaviors.

Fresh full review pr-review.md established one Important ownership race and no Critical/Minor. Parent recorded AC14/plan23 before correction. A new actual-runtime-pin regression pauses the real failed-start history iterator; a test thread removes an actual physically retired run from the actual history dictionary under the production lock. This is scoped scheduling instrumentation, **not an invocation of the production _retire collector**. It does not fabricate RuntimeError. Existing twenty-completion/history tests cover actual collection. RED review-i1-red.txt: **1 failed, 1 warning in25.21s** (child20.34s); the real dictionary iterator replaces the sentinel ValueError with RuntimeError, while the real pin remains open and close count is0.

The minimal fix computes the exact-pin ownership boolean under self._lock, releases the lock and then closes an unowned non-None pin. Original failure and already-owned writer behavior remain. GREEN review-i1-green.txt: **5 passed, 1 warning in137.63s**, exit0, elapsed142.24s. It proves the original exception, exactly one pin close outside the lock, operation retirement, actual service close/storage-pause drain, held production writer pin retention, failed-supervisor dispatch and both repaired retry params. Raw RED/GREEN children and hashes are retained in review-i1-child-retention.json. Scoped Ruff/format/AST/diff pass; shared controller test has3 inherited lint diagnostics on both e161 and current,0 introduced.

Correction committed **19d38fd32ed9009d9fd7c9ffb04e38fd989d370d**; exact four paths/hashes in review-i1-commit.json/review-i1-owned-source.json. Working tree clean after commit; no Backlog Done. The twenty prior covering passes remain qualified to e161, with only affected checks repeated. No whole-cover rerun. Final query proposal now points to this corrected commit and hashes24 source paths; service AST changes versus original3b616994e6 are _retire and _start_owned, while all service/DB query methods remain identical. Probe unchanged apart from output redirection. Query execution still awaits parent handshake and physical retirement of all task-owned checks/review work.

## Exact Cover command metadata

```json
{
  "command": [
    "C:\\Users\\GDesktop-1\\.codex\\worktrees\\workspace-project-guidance\\tldw_tui\\.venv\\Scripts\\python.exe",
    "-m",
    "pytest",
    "Tests/Chat/test_console_workspace_guidance.py",
    "Tests/Chat/test_console_library_runtime_policy.py::test_project_turn_preserves_all_library_policy_combinations",
    "Tests/UI/test_workspace_project_index_settings.py::test_actual_settings_card_enable_build_clear_preserves_library",
    "Tests/UI/test_workspace_project_index_settings.py::test_saved_policy_and_stale_actions",
    "Tests/UI/test_console_project_index_status.py::test_console_status_binding_fence_and_safe_settings_route",
    "Tests/Workspaces/test_project_index_service.py::test_real_app_shutdown_without_console_drains_index_writer",
    "Tests/Chat/test_console_chat_controller.py::test_controller_retry_retains_reservation_after_repreparation_failure",
    "Tests/Chat/test_console_chat_controller.py::test_dispatch_recovery_preflight_preserves_prior_generation_token",
    "Tests/UI/test_settings_configuration_hub.py::test_settings_ownership_records_cover_categories_and_runtime_boundaries",
    "Tests/UI/test_settings_configuration_hub.py::test_settings_overview_ownership_rows_are_sourced_from_record",
    "-q",
    "--tb=short",
    "--show-capture=no",
    "--no-showlocals"
  ],
  "commit": "e16100e2b3aea1e1da3d6d73ecbcbefebb135a80",
  "source_sha256": {
    "tldw_chatbook/Workspaces/project_index/service.py": "d0943ea7236310fd7ab6d585118d43f769f578198dcdd9e777941fb92ed7ed82",
    "Tests/Workspaces/test_project_index_service.py": "4089b955152ddf45b718e0df302d4e919b4890712c0f6faabc280b0f5f488e6c",
    "tldw_chatbook/DB/Workspace_Project_Index_DB.py": "2f5516cc05746296d563e92afb6fa9d6eb324e991f9b742455c0c6081292ea1a",
    "Tests/DB/test_workspace_project_index_db.py": "1bcdc71b2bc92839a8ff711d7e9aae873c168ad1e8e5d87c0cd941c1b7921260",
    "tldw_chatbook/DB/Workspace_DB.py": "ee1ffab98e0026f7137dbb2a7cc2b51562bf15c998c725b55a9d257ee2480485"
  },
  "started_unix": 1792648117.7343435,
  "cwd": "C:\\Users\\GDesktop-1\\.codex\\worktrees\\workspace-project-guidance\\tldw_tui",
  "expected_cases": 22,
  "scope": "single authorized broader covering invocation",
  "exit_code": 1,
  "elapsed_seconds": 625.2794768810272
}
```

## Full Cover output

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Lib\site-packages\pytest_asyncio\plugin.py:208: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture caching scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future. Valid fixture loop scopes are: "function", "class", "module", "package", "session"

  warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET))
................FF....                                                   [100%]
================================== FAILURES ===================================
_ test_controller_retry_retains_reservation_after_repreparation_failure[fresh] _
Tests\Chat\test_console_chat_controller.py:8693: in test_controller_retry_retains_reservation_after_repreparation_failure
    await test_two_saved_turns_keep_history_references_through_production_trace(
Tests\Chat\test_console_chat_controller.py:8159: in test_two_saved_turns_keep_history_references_through_production_trace
    controller = ConsoleChatController(
tldw_chatbook\Chat\console_chat_controller.py:3897: in __init__
    auxiliary_timeout_seconds=get_cli_setting(
tldw_chatbook\config.py:9812: in get_cli_setting
    config = load_cli_config_and_ensure_existence()  # Ensures config is loaded
tldw_chatbook\config.py:7287: in load_cli_config_and_ensure_existence
    return _load_cli_config_bootstrap(force_reload=force_reload).config
tldw_chatbook\Backup_Recovery\config_participants.py:505: in wrapped
    with operation(source, target=target):
C:\Python312\Lib\contextlib.py:137: in __enter__
    return next(self.gen)
tldw_chatbook\Backup_Recovery\config_participants.py:436: in operation
    with raw._scope(source, route, writing=True, selected_read=target) as active:
C:\Python312\Lib\contextlib.py:137: in __enter__
    return next(self.gen)
tldw_chatbook\Backup_Recovery\raw_participants.py:945: in _scope
    participant = _raw_participant(source)
tldw_chatbook\Backup_Recovery\raw_participants.py:357: in _raw_participant
    participant = _register_raw_participant(source, owner, selected)
tldw_chatbook\Backup_Recovery\raw_participants.py:307: in _register_raw_participant
    raise bootstrap.RecoveryRequired("raw_source_selection_changed")
E   tldw_chatbook.Backup_Recovery.bootstrap.RecoveryRequired: raw_source_selection_changed
_ test_controller_retry_retains_reservation_after_repreparation_failure[agent] _
Tests\Chat\test_console_chat_controller.py:8693: in test_controller_retry_retains_reservation_after_repreparation_failure
    await test_two_saved_turns_keep_history_references_through_production_trace(
Tests\Chat\test_console_chat_controller.py:8159: in test_two_saved_turns_keep_history_references_through_production_trace
    controller = ConsoleChatController(
tldw_chatbook\Chat\console_chat_controller.py:3897: in __init__
    auxiliary_timeout_seconds=get_cli_setting(
tldw_chatbook\config.py:9812: in get_cli_setting
    config = load_cli_config_and_ensure_existence()  # Ensures config is loaded
tldw_chatbook\config.py:7287: in load_cli_config_and_ensure_existence
    return _load_cli_config_bootstrap(force_reload=force_reload).config
tldw_chatbook\Backup_Recovery\config_participants.py:505: in wrapped
    with operation(source, target=target):
C:\Python312\Lib\contextlib.py:137: in __enter__
    return next(self.gen)
tldw_chatbook\Backup_Recovery\config_participants.py:436: in operation
    with raw._scope(source, route, writing=True, selected_read=target) as active:
C:\Python312\Lib\contextlib.py:137: in __enter__
    return next(self.gen)
tldw_chatbook\Backup_Recovery\raw_participants.py:945: in _scope
    participant = _raw_participant(source)
tldw_chatbook\Backup_Recovery\raw_participants.py:357: in _raw_participant
    participant = _register_raw_participant(source, owner, selected)
tldw_chatbook\Backup_Recovery\raw_participants.py:307: in _register_raw_participant
    raise bootstrap.RecoveryRequired("raw_source_selection_changed")
E   tldw_chatbook.Backup_Recovery.bootstrap.RecoveryRequired: raw_source_selection_changed
============================== warnings summary ===============================
..\..\..\..\Working\Github\tldw_tui\.venv\Lib\site-packages\pydantic\fields.py:1076
  C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Lib\site-packages\pydantic\fields.py:1076: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'exclude_if'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.11/migration/
    warn(

..\..\..\..\Working\Github\tldw_tui\.venv\Lib\site-packages\pydub\utils.py:14
  C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Lib\site-packages\pydub\utils.py:14: DeprecationWarning: 'audioop' is deprecated and slated for removal in Python 3.13
    import audioop

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================ slowest 25 durations =============================
96.67s call     Tests/UI/test_workspace_project_index_settings.py::test_actual_settings_card_enable_build_clear_preserves_library[textual-dark-size0]
89.28s call     Tests/UI/test_workspace_project_index_settings.py::test_actual_settings_card_enable_build_clear_preserves_library[textual-light-size1]
50.78s call     Tests/Chat/test_console_workspace_guidance.py::test_pending_project_turn_stays_with_owning_tab[initial-fenced]
43.35s call     Tests/Chat/test_console_workspace_guidance.py::test_pending_project_turn_stays_with_owning_tab[retry-native]
42.02s call     Tests/Workspaces/test_project_index_service.py::test_real_app_shutdown_without_console_drains_index_writer
38.95s call     Tests/Chat/test_console_workspace_guidance.py::test_pending_project_turn_stays_with_owning_tab[initial-native]
38.67s call     Tests/Chat/test_console_workspace_guidance.py::test_pending_project_turn_stays_with_owning_tab[retry-fenced]
30.02s call     Tests/Chat/test_console_library_runtime_policy.py::test_project_turn_preserves_all_library_policy_combinations[allowed-automatic]
29.78s call     Tests/Chat/test_console_library_runtime_policy.py::test_project_turn_preserves_all_library_policy_combinations[blocked-never]
29.62s call     Tests/Chat/test_console_library_runtime_policy.py::test_project_turn_preserves_all_library_policy_combinations[blocked-automatic]
28.42s call     Tests/Chat/test_console_library_runtime_policy.py::test_project_turn_preserves_all_library_policy_combinations[allowed-never]
17.64s call     Tests/UI/test_settings_configuration_hub.py::test_settings_overview_ownership_rows_are_sourced_from_record
16.61s call     Tests/UI/test_workspace_project_index_settings.py::test_saved_policy_and_stale_actions
11.50s call     Tests/UI/test_console_project_index_status.py::test_console_status_binding_fence_and_safe_settings_route[textual-dark-size0]
9.22s call     Tests/UI/test_console_project_index_status.py::test_console_status_binding_fence_and_safe_settings_route[textual-light-size1]
6.39s call     Tests/UI/test_settings_configuration_hub.py::test_settings_ownership_records_cover_categories_and_runtime_boundaries
3.93s call     Tests/Chat/test_console_workspace_guidance.py::test_dispatch_retry_provider_payload_allows_workspace_grounding
2.75s call     Tests/Chat/test_console_chat_controller.py::test_dispatch_recovery_preflight_preserves_prior_generation_token[stopped]
2.58s call     Tests/Chat/test_console_chat_controller.py::test_dispatch_recovery_preflight_preserves_prior_generation_token[failed]
2.06s call     Tests/Chat/test_console_workspace_guidance.py::test_automatic_evidence_provider_payload_allows_workspace_grounding
1.76s call     Tests/Chat/test_console_chat_controller.py::test_controller_retry_retains_reservation_after_repreparation_failure[agent]
1.41s call     Tests/Chat/test_console_chat_controller.py::test_controller_retry_retains_reservation_after_repreparation_failure[fresh]

(3 durations < 1s hidden.  Use -vv to show these durations.)
=========================== short test summary info ===========================
FAILED Tests/Chat/test_console_chat_controller.py::test_controller_retry_retains_reservation_after_repreparation_failure[fresh]
FAILED Tests/Chat/test_console_chat_controller.py::test_controller_retry_retains_reservation_after_repreparation_failure[agent]
2 failed, 20 passed, 2 warnings in 617.23s (0:10:17)

```

## Exact Affected correction command metadata

```json
{
  "command": [
    "C:\\Users\\GDesktop-1\\.codex\\worktrees\\workspace-project-guidance\\tldw_tui\\.venv\\Scripts\\python.exe",
    "-m",
    "pytest",
    "Tests/Workspaces/test_project_index_service.py::test_failed_start_history_race_retires_unrecorded_production_pin",
    "Tests/Workspaces/test_project_index_service.py::test_production_runtime_pin_drains_idle_and_spans_writer_retirement",
    "Tests/Workspaces/test_project_index_service.py::test_failed_supervisor_dispatch_waits_for_started_writer_retirement",
    "Tests/Chat/test_console_chat_controller.py::test_controller_retry_retains_reservation_after_repreparation_failure",
    "-q",
    "--tb=short",
    "--show-capture=no",
    "--no-showlocals"
  ],
  "source_sha256": {
    "tldw_chatbook/Workspaces/project_index/service.py": "aa764965e2a0d87737d47e1f9f5411d123af64e06e471eb0c0e27d346c6cd648",
    "Tests/Workspaces/test_project_index_service.py": "ac0b4dcb0849213a01be69c21a8424990fb2516718512699247691eb8c339653",
    "Tests/Chat/test_console_chat_controller.py": "e6d89b430e020d7f7af6fd8bdbc6d0fdc21cea0bbe1940d9c5557e76cd2ed078"
  },
  "started_unix": 1792649078.6732411,
  "cwd": "C:\\Users\\GDesktop-1\\.codex\\worktrees\\workspace-project-guidance\\tldw_tui",
  "expected_cases": 5,
  "mutation_qualification": "Test thread removes actual physically retired run from actual history under production lock; no production collector invocation or fabricated iterator exception.",
  "exit_code": 0,
  "elapsed_seconds": 142.24106740951538
}
```

## Full Affected correction output

```text
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Lib\site-packages\pytest_asyncio\plugin.py:208: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture caching scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future. Valid fixture loop scopes are: "function", "class", "module", "package", "session"

  warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET))
.....                                                                    [100%]
============================== warnings summary ===============================
Tests/Workspaces/test_project_index_service.py::test_failed_start_history_race_retires_unrecorded_production_pin
  C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Lib\site-packages\pydantic\fields.py:1076: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'exclude_if'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.11/migration/
    warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================ slowest 25 durations =============================
45.75s call     Tests/Chat/test_console_chat_controller.py::test_controller_retry_retains_reservation_after_repreparation_failure[agent]
32.42s call     Tests/Chat/test_console_chat_controller.py::test_controller_retry_retains_reservation_after_repreparation_failure[fresh]
23.96s call     Tests/Workspaces/test_project_index_service.py::test_failed_start_history_race_retires_unrecorded_production_pin
21.63s call     Tests/Workspaces/test_project_index_service.py::test_production_runtime_pin_drains_idle_and_spans_writer_retirement
11.83s call     Tests/Workspaces/test_project_index_service.py::test_failed_supervisor_dispatch_waits_for_started_writer_retirement

(10 durations < 1s hidden.  Use -vv to show these durations.)
5 passed, 1 warning in 137.63s (0:02:17)

```

## Final query invocation and external gate update

After commit19d38fd32ed9009d9fd7c9ffb04e38fd989d370d, source/probe hash verification, clean status and exact task-owned process retirement check, parent gave the final timing handshake. The single prepared query-only invocation started in session64797; exact command/profile/load/start/source/probe hashes are in integration-idle-source.json. CLI uses -p Tests.conftest because the old probe directory had a local conftest with that exact plugin, while this directory does not. Existing bootstrap_profile semantics and900s bound are preserved. No additional test-body change, corpus rebuild/relocation or overwrite of old JSON. Result pending; no performance PASS claimed.

Human explicitly waived Qodo for PR3204 only. Parent records that exception; independent review, required CI, resolved threads, queue and actual merge before semantic implementation remain mandatory. This supersedes the earlier billing-blocked gate status, not the historical receipt.

## Final same-cache query result

The single run retired with **1 passed, 1 warning in244.90s**, exit0, elapsed248.96s, on exact source19d38fd32ed9009d9fd7c9ffb04e38fd989d370d. All24 source hashes remained unchanged. No timing rerun, corpus/cache rebuild or old receipt overwrite. Light read-only coordination and uncontrolled OS background remain qualified. Exact recalculated summary/raw artifact hashes: integration-idle-summary.json.

| Shape | Workload | Canonical lexical p95 ms | Expanded admitted p95 ms |
|---|---|---:|---:|
| nested | common | 273.264 | 644.722 |
| nested | rare | 205.671 | 549.382 |
| nested | miss | 169.099 | 454.642 |
| wide | common | 244.297 | 617.648 |
| wide | rare | 168.701 | 474.645 |
| wide | miss | 164.260 | 435.502 |

All six canonical cells satisfy the original500ms gate; admitted misses remain explicit in the summary and table, not relabeled as canonical or an authority optimization. Exact100/1/0 counts across50 measured samples after5 warmups per cell passed. All42 actual Capture retirement proofs are positive (worker gone, retired, pipes closed, return code observed);40 public calls entered without borrowed registry and returned with actual registry retirement. Final service close leaves0 searches and0 operations. Instrumented public whole-tool p95 is {'nested': 1948.863400000846, 'wide': 2443.247999995947}ms; includes probe retirement bookkeeping and is not the canonical gate or a universal user-latency promise.

## Exact final query command/source metadata

```json
{
  "commit": "19d38fd32ed9009d9fd7c9ffb04e38fd989d370d",
  "command": [
    "C:\\Users\\GDesktop-1\\.codex\\worktrees\\workspace-project-guidance\\tldw_tui\\.venv\\Scripts\\python.exe",
    "-m",
    "pytest",
    "-p",
    "Tests.conftest",
    ".superpowers\\sdd\\2026-10-22-keyword-dev-integration\\test_integration_idle_latency.py",
    "-q",
    "--tb=short",
    "--show-capture=no",
    "--no-showlocals",
    "--timeout=900"
  ],
  "cwd": "C:\\Users\\GDesktop-1\\.codex\\worktrees\\workspace-project-guidance\\tldw_tui",
  "started_unix": 1792649375.2345402,
  "source_sha256": {
    "tldw_chatbook/Workspaces/project_index/service.py": "aa764965e2a0d87737d47e1f9f5411d123af64e06e471eb0c0e27d346c6cd648",
    "tldw_chatbook/Workspaces/project_index/runtime.py": "5bce003a30f2a0dc5538d948270b820feae21004790cc9d8261b889c8edca755",
    "tldw_chatbook/Workspaces/project_index/lifetime.py": "f84d62b1574fecf715a76e9c10949de84be287c6028816545c14888d7d7c8672",
    "tldw_chatbook/Workspaces/project_index/scanner.py": "5e34731b35153728779d59ad7c3566b61dfde8486055caa45e494b431ef5fd81",
    "tldw_chatbook/Workspaces/project_index/models.py": "b76e8cab113b10131de502fa42277e224269d14bb0060a5c459298c433ccb76d",
    "tldw_chatbook/DB/Workspace_Project_Index_DB.py": "2f5516cc05746296d563e92afb6fa9d6eb324e991f9b742455c0c6081292ea1a",
    "tldw_chatbook/Chat/console_chat_controller.py": "c213c2659248d6d75ef776f31e88104adb7e72b03c39820621c6bd7fab766d5e",
    "tldw_chatbook/Agents/local_tool_provider.py": "f54493f97a97714fe58be15673b7ad9a32c02402d1f5e31c00879c298c7f0310",
    "tldw_chatbook/UI/Console_Modules/workspace.py": "c68f8dc9f1c55d28d8d3d18dcc8aa76212112b92c0be2b8323962b96ffb696d8",
    "tldw_chatbook/UI/Screens/settings_screen.py": "f7bae7273c060f126212fd23c98a070ad6f5accf5564735396990955cfcd81f1",
    "tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py": "1dc9240797879be7734bf1960763c88fdc3423c193745e959ef4ca082379f6cc",
    "tldw_chatbook/Workspaces/registry_service.py": "4bd5b8d3065ffd4caeb8e8ec69b0cece6eea5ff8defba3a94ba816303019ec04",
    "Tests/Agents/test_project_index_tool.py": "8050fb112e1c937eca31b062b03efd939ba7e2b737eb5154b591cfb6aac570ad",
    "Tests/Chat/test_console_workspace_folder_note_contract.py": "f50a9519e883214b9a34874f886a1a256810c0b91be35e515bf043e0685c39a1",
    "Tests/DB/test_workspace_project_index_db.py": "1bcdc71b2bc92839a8ff711d7e9aae873c168ad1e8e5d87c0cd941c1b7921260",
    "Tests/UI/test_workspace_project_index_settings.py": "b03fe667216f8d553a65dfc7aae8713a0c5e30c6757c5714582395621b6a7c66",
    "Tests/Workspaces/test_project_index_detached_cleanup.py": "da3ee32891c0f152b24ddb7e4faa73b9f04ae37cb6114cd3d412a611645ca31c",
    "Tests/Workspaces/test_project_index_service.py": "ac0b4dcb0849213a01be69c21a8424990fb2516718512699247691eb8c339653",
    "tldw_chatbook/Tools/project_index_scan_worker.py": "1cc7f40ab95aba039b102197d07e86247fb4e6641f1a2c5195946e5f5d99b25e",
    "tldw_chatbook/Tools/project_index_scan_protocol.py": "05de5a229e37da04556dfa244ab5991dbbf301b921343a936beb8cd5034a93b3",
    "tldw_chatbook/DB/migrations/workspace_project_index_003.sql": "3d24306f98ee0a2041193677db65a2e9d585480b4f179191bb65f502991bdbad",
    "tldw_chatbook/Backup_Recovery/participants.py": "14aef638569a1c453b37843cb4748658a157d17e296fbe53687b1e9edec0f5a6",
    "tldw_chatbook/Backup_Recovery/storage_admission.py": "d9ec07f682a027aebee1da3d06fa1fbaa170e3ed6650392cd7f3b5b3b4b1253a",
    "tldw_chatbook/DB/store_kernel.py": "825daabb5547e2b1650dec218ba03ca9434610b5c715489edb60d97df8fe7077"
  },
  "probe_sha256": "66d1a9b78a21cde38e520f1bc3640846851a5b3ac69d4189c8e9ea3724372b81",
  "profile": "C:\\Users\\GDesktop-1\\.codex\\worktrees\\workspace-project-guidance\\tldw_tui\\.superpowers\\k5-scale\\stable-profile",
  "load": "All task-owned tests, renderers and root guards physically retired. Parent handshake received; light read-only review/coordination and uncontrolled OS background may overlap.",
  "gate": "Canonical decorated lexical p95<=500ms. Expanded admitted timing remains separately measured with prior accepted misses, no retry for favorable timing.",
  "pytest_bootstrap": "-p Tests.conftest replaces old local conftest.py pytest_plugins=[Tests.conftest]; same bootstrap_profile semantics.",
  "exit_code": 0,
  "elapsed_seconds": 248.96487045288086,
  "source_hashes_unchanged_after_run": true
}
```

## Full final query output

```text
[32m2026-10-21 23:09:36.221[0m | [34m[1mDEBUG   [0m | [36mtldw_chatbook.config[0m:[36m<module>[0m:[36m150[0m - [34m[1mCRITICAL DEBUG: config.py module is being imported/executed NOW.[0m
[32m2026-10-21 23:09:36.462[0m | [1mINFO    [0m | [36mtldw_chatbook.config[0m:[36m_load_cli_config_bootstrap_unlocked[0m:[36m7087[0m - [1mAttempting to load CLI config from: C:\Users\GDesktop-1\.codex\worktrees\workspace-project-guidance\tldw_tui\.superpowers\k5-scale\stable-profile\config\config.toml[0m
[32m2026-10-21 23:09:36.487[0m | [1mINFO    [0m | [36mtldw_chatbook.config[0m:[36m_load_cli_config_bootstrap_unlocked[0m:[36m7108[0m - [1mSuccessfully loaded and merged CLI config from C:\Users\GDesktop-1\.codex\worktrees\workspace-project-guidance\tldw_tui\.superpowers\k5-scale\stable-profile\config\config.toml[0m
[32m2026-10-21 23:09:36.488[0m | [34m[1mDEBUG   [0m | [36mtldw_chatbook.config[0m:[36m_load_cli_config_bootstrap_unlocked[0m:[36m7192[0m - [34m[1mload_cli_config_and_ensure_existence returning config with top-level keys: ['config_schema_version', 'general', 'console', 'console_ssh', 'hooks', 'skills', 'tamagotchi', 'appearance', 'acp', 'tldw_api', 'library', 'caching', 'agents', 'splash_screen', 'session_summary', 'logging', 'metrics', 'database', 'webhooks', 'scheduling', 'media_cleanup', 'api_endpoints', 'providers', 'model_catalog', 'api_settings', 'chat_defaults', 'chat', 'web_security', 'network', 'image_generation', 'character_defaults', 'analysis_defaults', 'permission_summary', 'llm_management', 'llamacpp_snapshots', 'notes', 'Prompts', 'prompts', 'embedding_config', 'rag_citations', 'rag', 'rag_search', 'chunking', 'sync', 'model_capabilities', 'tools', 'SearchSettings', 'webfetch', 'SearchEngines', 'media_processing', 'meetings', 'dictation', 'transcription', 'diarization', 'local_ingestion', 'mcp', 'subscriptions', 'github', 'briefings_feed_server', 'canvas', 'web_server'][0m
[32m2026-10-21 23:09:36.488[0m | [34m[1mDEBUG   [0m | [36mtldw_chatbook.config[0m:[36m_load_cli_config_bootstrap_unlocked[0m:[36m7196[0m - [34m[1m  'api_settings' found with keys: ['openai', 'anthropic', 'cohere', 'deepseek', 'groq', 'google', 'huggingface', 'mistralai', 'openrouter', 'moonshot', 'qwencloud', 'zai', 'databricks', 'together', 'fireworks', 'cerebras', 'sambanova', 'nvidia', 'deepinfra', 'nebius', 'novita', 'minimax', 'mimo', 'tokenhub', 'byteplus', 'stepfun', 'vercel', 'zenmux', 'kilo', 'siliconflow', 'baseten', 'gmi', 'ollama_cloud', 'upstage', 'arcee', 'qianfan', 'nous', 'venice', 'meta', 'azure', 'wandb', 'cloudflare', 'opencode_zen', 'commandcode', 'llama_cpp', 'oobabooga', 'koboldcpp', 'ollama', 'vllm', 'aphrodite', 'tabbyapi', 'custom', 'custom_2', 'local-llm', 'local_llamafile', 'local_llamacpp', 'local_vllm', 'local_ollama', 'local_onnx', 'local_transformers', 'local_mlx_lm'][0m
[32m2026-10-21 23:09:36.640[0m | [1mINFO    [0m | [36mtldw_chatbook.config[0m:[36m_load_settings_uncached[0m:[36m2321[0m - [1mDetermined ACTUAL_PROJECT_ROOT for general paths: C:\Users\GDesktop-1\.codex\worktrees\workspace-project-guidance\tldw_tui\tldw_chatbook[0m
[32m2026-10-21 23:09:36.641[0m | [1mINFO    [0m | [36mtldw_chatbook.config[0m:[36m_load_settings_uncached[0m:[36m2324[0m - [1mDetermined APP_COMPONENT_ROOT for config files: C:\Users\GDesktop-1\.codex\worktrees\workspace-project-guidance\tldw_tui\tldw_chatbook[0m
[32m2026-10-21 23:09:36.644[0m | [34m[1mDEBUG   [0m | [36mtldw_chatbook.config[0m:[36m_load_settings_uncached[0m:[36m2347[0m - [34m[1mload_settings: Configuration loaded from disk (cache miss or forced reload)[0m
[32m2026-10-21 23:09:37.053[0m | [34m[1mDEBUG   [0m | [36mtldw_chatbook.config[0m:[36m_load_settings_uncached[0m:[36m3910[0m - [34m[1mEnsured chat dictionaries folder exists: C:\Users\GDesktop-1\.codex\worktrees\workspace-project-guidance\tldw_tui\.superpowers\k5-scale\stable-profile\home\.local\share\tldw_cli\default_user\chat_dicts[0m
[32m2026-10-21 23:09:37.053[0m | [34m[1mDEBUG   [0m | [36mtldw_chatbook.config[0m:[36m_load_settings_uncached[0m:[36m3941[0m - [34m[1mload_settings: Configuration cached for future use[0m
[32m2026-10-21 23:09:37.054[0m | [1mINFO    [0m | [36mtldw_chatbook.config[0m:[36m<module>[0m:[36m11956[0m - [1mDefault API Endpoint (from config.py global scope): openai[0m
[32m2026-10-21 23:09:37.054[0m | [34m[1mDEBUG   [0m | [36mtldw_chatbook.config[0m:[36m<module>[0m:[36m11979[0m - [34m[1mCRITICAL DEBUG: Database initialization is now lazy - will initialize on first access[0m
[32m2026-10-21 23:09:37.054[0m | [1mINFO    [0m | [36mtldw_chatbook.Utils.optional_deps[0m:[36m<module>[0m:[36m1554[0m - [1mLazy dependency checking enabled (default). Dependencies will be checked on first use.[0m
C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Lib\site-packages\pytest_asyncio\plugin.py:208: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture caching scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future. Valid fixture loop scopes are: "function", "class", "module", "package", "session"

  warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET))
.                                                                        [100%]
============================== warnings summary ===============================
.superpowers/sdd/2026-10-22-keyword-dev-integration/test_integration_idle_latency.py::test_final_fix_idle_latency
  C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Lib\site-packages\pydantic\fields.py:1076: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'exclude_if'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.11/migration/
    warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================ slowest 25 durations =============================
244.10s call     .superpowers/sdd/2026-10-22-keyword-dev-integration/test_integration_idle_latency.py::test_final_fix_idle_latency

(2 durations < 1s hidden.  Use -vv to show these durations.)
1 passed, 1 warning in 244.90s (0:04:04)

```
