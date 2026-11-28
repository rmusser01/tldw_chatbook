---
id: TASK-41116
title: "Eight Backup_Recovery owner-suite nodes are deselected in the push-to-dev job: two are cheap test fixes, one needs a private_sqlite boundary decision"
status: In Progress
assignee:
- '@claude'
updated_date: 2026-11-28 17:38
created_date: 2026-11-15 02:20
labels:
- follow-up
- backup
- tests
- dev-red
priority: medium
references:
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
- Tests/Backup_Recovery/test_complete_roundtrip.py
- Tests/Backup_Recovery/test_inventory.py
- Tests/Backup_Recovery/test_core_owners.py
- Tests/Backup_Recovery/test_later_rollback_credential_ui.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** TASK-40307's dev-push job leaves out 8 nodes (`BACKUP_OWNER_SUITES_DESELECTED`, `derived-artifacts.yml` the `--deselect` lines of "Run the Backup_Recovery owner suites"). Until each is fixed or retired, those behaviours are guarded by nothing. TASK-40307's notes give each cause:
- `test_complete_roundtrip.py::test_two_known_profiles_complete_capture_preserves_native_state_and_resumes` and `::test_two_captured_profiles_plan_shared_concrete_roots_for_isolated_restore`: fixture. `(profile / "custom").mkdir(parents=True, mode=0o700)` leaves the parent at the umask mode, and `config_participants.companion_guard` refuses `config_companion_parent_unsafe`. Cheap.
- `test_inventory.py::test_fresh_process_discovery_imports_no_bootstrap_or_services_and_changes_nothing`: drift. The child forbids every `tldw_chatbook.Utils*` import, and since 8bbac5bd20 `tldw_chatbook/__init__.py` imports `Utils.tiktoken_runtime`. Cheap (allow that one module).
- `test_core_owners.py::test_factory_discovery_and_schema_policy_are_runtime_independent`: since 79a46b3b4e `recovery_workflow_schema.open_installed_schema` reaches `DB.private_sqlite`. A test pin or a product boundary decision.
- `test_inventory.py::test_normal_config_and_pure_resolvers_agree`: reads config in a plain test (`RecoveryRequired: raw_source_selection_changed`).
- `test_inventory.py::test_custom_db_and_disabled_features_are_not_external_or_unused`: intermittent (`'included' == 'unsupported'`), cause not investigated.
- `test_complete_roundtrip.py::test_two_captured_profiles_restore_and_open_with_native_content` and `test_later_rollback_credential_ui.py::test_f9_later_rollback_requires_explicit_credential_review`: the shared `native_package` fixture's fixed 60 s `pip install --target`; past it, f9 failed `AssertionError: real exec expected`, cause not investigated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the 8 nodes passes and its --deselect line and BACKUP_OWNER_SUITES_DESELECTED entry are removed, or it is retired with the commit that changed what it pinned named
- [x] #2 The private_sqlite boundary in schema_policy() is decided (test pin or product change) and recorded in the notes
- [x] #3 The cause of 'real exec expected' in the f9 node is recorded
<!-- AC:END -->

## Implementation Plan

1. Reproduce each of the 8 deselected nodes at -n 4.
2. Fix the cheap test causes: the roundtrip fixture's parent directory mode, the order-dependent owner registry in the inventory node, the f9 drive's stale hotkey.
3. Record the cause of 'real exec expected'.
4. For the private_sqlite boundary (core owners, fresh-process discovery) and the plain-config node, report NEEDS_CONTEXT with options instead of choosing.
5. Remove each fixed node's --deselect line and BACKUP_OWNER_SUITES_DESELECTED entry; update ADR-103's dated paragraph count.

## Implementation Notes

Commits 6673ea6661, 146f5156af, 7d9c3988b1, a7816ce8b5, 829b7d3047. Net, five nodes re-admitted (8 -> 3 deselected): the custom-DB node, the two private_sqlite nodes and the two roundtrip capture nodes (re-admitted, deselected again, then re-admitted after the bind fix in the last section). Not Done: AC#1 is open for the three.

**Re-admitted (green, --deselect line and pin entry removed):**
- `test_inventory.py::test_custom_db_and_disabled_features_are_not_external_or_unused`: cause found, order dependence. `install_adapters()` registers process-wide; any earlier test in the worker that ran it (test_restore_plan, test_data_groups, ...) made the media core report the custom file `included`. Red: `pytest -p no:xdist test_restore_plan.py::test_selected_shared_files_remain_one_physical_candidate test_inventory.py::<node>` -> `'included' == 'unsupported'`; deterministic. Fix: an autouse fixture gives every test_inventory test an empty registry. The same pair also failed `test_multiple_profile_configs_preserve_shared_physical_alias` (`undeclared_alias`), a latent flake in the job that this fixes too; with the fixture, the file after install_adapters: 32 passed (only the private_sqlite node below fails).
- `test_complete_roundtrip.py::test_two_known_profiles_...` and `::test_two_captured_profiles_plan_shared_...`: three successive causes. (1) `mkdir(parents=True, mode=0o700)` left the profile dir at the umask mode -> `config_companion_parent_unsafe`; now created at 0o700. (2) Then capture refused the seed's 0o644 `ui_state.toml` (`capture_config_source_unsafe`); children now run under `os.umask(0o077)`. (3) Then fixed budgets killed slow children at load ~100: `_run_profile_child`'s 60 s and the capture's 55 s watchdog. Children now fail on CPU idleness (`_wait_while_busy(idle=60, cap 10x)`), the watchdog sits inside that cap. Green at -n 4, load 72-98: 257 s and 265 s of call, close to the job's 300 s per-test limit on this host.

**Still deselected (the two private_sqlite nodes below were re-admitted later, in 829b7d3047):**
- `::test_two_captured_profiles_restore_and_open_with_native_content` and `test_later_rollback_credential_ui.py::test_f9_...`: both use the `native_package` fixture (wheel build + `pip install --target`, about 140-150 s of setup at load 80-100). Its fixed 60 s pip timeout fired at load ~90; it now fails only on CPU idleness. Remaining failures at load 72-98: the native-content node exceeds pytest's 300 s (setup 141 s + call 158 s), and once in the bind child `close_unenrolled_clients_and_restart` (`control_records._enrollment` waits 0.1 s for the unbound namespace; load-sensitive product timeout).
- **AC#3, cause of "real exec expected" (recorded):** 04233174e6 (2026-10-18) bound F9 to Kanban. The f9 drive pressed F9 (until then unbound, a no-op; `default_tab="settings"` kept it on Settings), so it now left Settings, `query_one('#settings-backup-restore')` raised NoMatches, `app.run()` exited and `main_cli_runner()` returned, reaching `raise AssertionError('real exec expected')`. The drive now presses `SHELL_DESTINATION_SHORTCUTS['settings']`; the node then gets through the restart into the recovery app (FRESH_ARCHIVE_INSPECTED) and fails on the restart driver's 20 s `asyncio.timeout` at load ~100.
- `test_inventory.py::test_normal_config_and_pure_resolvers_agree`: `RecoveryRequired: raw_source_selection_changed` from an in-process config read. Left alone: the round-15 brief gives that trap to TASK-41127 only.
- **AC#2, the private_sqlite boundary (options as reported; option (a) was ruled, below).** Two nodes hinge on it:
  - `test_core_owners.py::test_factory_discovery_and_schema_policy_are_runtime_independent` asserts `tldw_chatbook.DB.private_sqlite` is not imported. It is imported at `DB/recovery_operations.py` import, by the module-level `_AGENT_RUNS_SCHEMA += ... workflow_admission_catalog(catalog)`, which replays installed DDL in an in-memory SQLite via `connect_private_sqlite("recovery.validation_schema", ":memory:")`. That line came with 175ea27008 (2026-11-01); 79a46b3b4e only moved the opener from `sqlite_validation` to `recovery_workflow_schema` (the attribution in the filing is off). Measured side effects of all core `schema_policy()` + `discover()` calls: 60 `sqlite3.connect` audit events (in-memory), no subprocess, no thread, no config import, HOME not created.
  - `test_inventory.py::test_fresh_process_discovery_...` forbids every `tldw_chatbook.DB*` and `Utils*` import. Discovery imports `Utils.tiktoken_runtime` (package init), `Utils.platform_files` (`inventory.py` top), and via `recovered_media` `DB.private_sqlite`, `DB.recovery_sqlite`, `Utils.private_paths`, `Utils.path_validation`, `Utils.file_durability`. Checked on the archived tree of the merge that added the test (24094f23d5): the same set, so this node was red from birth; allowing tiktoken_runtime alone would not make it pass.
  - Options: (a) test pin: allow the low-level `Utils` modules and `DB.private_sqlite*`/`DB.recovery_sqlite`/`DB.sqlite_datetime_fix` in both children, keep config, the service DBs (`ChaChaNotes_DB`, `base_db`, ...), keyring, chromadb, textual forbidden and the no-write assertions. Consequence: discovery and schema policy may load the private SQLite layer and open in-memory connections. (b) product: keep discovery DB-free: precompute or lazily derive the workflow admission catalog (a frozen tuple needs a drift test), and defer `recovered_media`'s private_sqlite/recovery_sqlite imports into the functions that open files. Consequence: a Backup_Recovery/DB change with a wider blast radius. (c) retire both nodes, naming 175ea27008 and b5251e9a6e.

**AC#2 decided (coordinator ruling 2026-11-16: option (a), test pin), commit 829b7d3047.**
- `test_inventory.py::test_fresh_process_discovery_...`: the child's import guard allows exactly `tldw_chatbook.Utils`, `.Utils.tiktoken_runtime`, `.Utils.platform_files`, `.Utils.private_paths`, `.Utils.path_validation`, `.Utils.file_durability`, `tldw_chatbook.DB`, `.DB.sqlite_datetime_fix`, `.DB.private_sqlite`, `.DB.private_sqlite_process`, `.DB.private_sqlite_protocol` and `.DB.recovery_sqlite` (the set discovery has imported since b5251e9a6e). `tldw_chatbook.config`, every other DB/Utils module, keyring, chromadb and textual stay forbidden; the before/after tree comparison is unchanged.
- `test_core_owners.py::test_factory_discovery_...`: private_sqlite may load (175ea27008's in-memory workflow catalog); an audit hook requires every `sqlite3.connect` to be `":memory:"`, and config, ChaChaNotes_DB, base_db and the HOME directory stay absent.
- Red, then green, with one command (`pytest <both nodes> -n 2`): 2 failed -> 2 passed. Negative controls: dropping `DB.recovery_sqlite` from the allowlist fails the inventory node (`forbidden service import`); a file `sqlite3.connect` in the child fails the core node (the path appears in the assertion). Both files whole at -n 4: 98 passed.
- Both `--deselect` lines and pin entries removed; ADR-103's dated paragraph says 3.

**Review r15bt-rev1 (a944bd1755, 4905d25fc4).**
- The two re-admitted roundtrip capture nodes hit the job's 300 s per-test limit at load 88-109, so the child cap and watchdog never applied. A complete run with the limit lifted (-n 4, load 67-110) took 393 s and 403 s, and both passed. Each now carries `@pytest.mark.timeout(810)`, 2x the 403 s, and ADR-103 records it.
- Both purity children share one prelude from `test_inventory.py` (`pure_child_guard`, `PURE_CHILD_CHECK`); `test_core_owners` adds its six `DB.recovery_*` modules.
  - Each child refuses config, keyring, chromadb, textual and every other DB/Utils module, both on import and by a `sys.modules` check.
  - Each records `sqlite3.connect` and process audit events, allows only `":memory:"` (a controlled connect proves the hook fires) and no process.
  - Negative controls failed as they should: removing `recovery_goal_schema` -> `forbidden service import`; a subprocess in the inventory child -> `['subprocess.Popen']`.

**Roundtrip capture nodes deselected again (5 deselected).** A run with the job's flags (`-n 4 --timeout=300`, load 69-87) showed the 810 s marks take effect: `[remove-include]` 417 s, `[link-include]` 358 s and the known-profiles node 332 s all passed. The plan-shared node failed at 111 s in its bind child: `RecoveryRequired: close_unenrolled_clients_and_restart`. `control_records._enrollment` gives `authority.maintenance((UNBOUND_NAMESPACE,), 0.1)` only 0.1 s when no session is passed, and under load something still holds the unbound namespace at that moment. Both nodes share that bind child, and it refused once before (round 15, load ~100). They are green when the bind wins, but not stably, so both are deselected again with that cause; their marks stay for re-admission. Follow-up: identify the holder (probably the preview's own private-SQLite helpers still exiting), then make the bind wait for it or pass the session.

**Review r15bt-rev2: the bind fixture fixed, both roundtrip nodes re-admitted.** The refusal came from the fixture, not the product: production's `replacement.py` binds inside `authority.maintenance(names, 30)` and passes `session=`, while the `_BIND` child called `bind_profile` with no session, and with no session `_enrollment` waits only 0.1 s. The child now binds inside `authority.maintenance((UNBOUND_NAMESPACE, *names), 30)` with `session=`. With the job's flags (`-n 4 --timeout=300`, marks 840 s), both nodes passed 3 of 3 runs at load 46-62 (calls 178-190 s). Their deselects, Tests/CI pin entries and the workflow comment are gone, and ADR-103 says 3. Still deselected: the native-content node (same fixture, but `native_package` setup takes 140-150 s at load ~100 and the node has no measured mark yet), the f9 node, and the plain-config node.

**`test_inventory.py::test_normal_config_and_pure_resolvers_agree`, traced (round 29, 2026-11-28, during TASK-42650).** TASK-41127 did not take this node, so it is still this task's.
- **What raises:** the test calls the guarded `config.get_user_data_dir()` directly. Under the per-test `TLDW_CONFIG_PATH` redirect that read raises `RecoveryRequired('raw_source_selection_changed')`, because conftest's import-time config reads recorded the bootstrap path as `tldw_chatbook.config`'s raw selection.
- **Same failure as TASK-42650,** but this node has no injectable reader to substitute.
- **A `bootstrap_profile` marker alone is not enough** (tried, reverted): the test then fails with `config_directory_selection_changed`, because it patches `get_cli_setting` to move `data_dir`.
- **What it needs:** a selected config source (`select_config_source` with a real config file) that holds that `data_dir`.

