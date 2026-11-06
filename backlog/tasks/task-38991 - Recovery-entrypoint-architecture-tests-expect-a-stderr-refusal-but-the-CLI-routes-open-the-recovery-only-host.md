---
id: TASK-38991
title: "Recovery-entrypoint architecture tests expect a stderr refusal, but the CLI routes open the recovery-only host"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:42
updated_date: 2026-11-06 04:43
labels:
- follow-up
- backup
- recovery
- tests
priority: medium
references:
- tldw_chatbook/cli.py
- tldw_chatbook/Backup_Recovery/launcher.py
- tldw_chatbook/Backup_Recovery/bootstrap.py
- Tests/Architecture/test_recovery_entrypoints.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. The cause was traced in review of the filing.

All 7 failing nodes are in `Tests/Architecture/test_recovery_entrypoints.py`. The file had 7 failed and 25 passed (108 s), whether XDG_* was set or unset. TASK-34100.4 (In Progress) recorded the same 7 as pre-existing on its base but does not own them. No lane runs the file, and none of the branches due to merge before this one touches it.

**The route tests (5 nodes): the expectation is stale; the product is safe.**
- `test_real_route_refuses_before_side_effect_imports` fails for the `main_cli_runner()` route and the `runpy tldw_chatbook __main__` route, each with the parameter True and False. That is 4 nodes.
  - The test expects the child to print "Recovery required: recovery_pending" and exit.
  - Instead the child paints a Textual screen and is killed at the 20 s timeout.
- That paint is the recovery-only host, not the app:
  - `cli.py:84` calls `launcher.startup_unlock()`;
  - `startup_preflight()` returns `recovery_pending` (`tldw_chatbook/Backup_Recovery/bootstrap.py:514-517`);
  - `startup_unlock` then returns `minimal_recovery(reason)` (`tldw_chatbook/Backup_Recovery/launcher.py:448-449`), which runs `recovery_app(reason).run()` (`:620-623`).
- `RecoveryApp` never imports the normal app. On `recovery_pending`, its `on_mount` opens Backup & Restore. A review probe at base confirmed that the route calls `minimal_recovery('recovery_pending')` and that `tldw_chatbook.app` is never imported.
- The routing predates the test (b5251e9a6e, 2026-09-16). TASK-34100.4's 87aa6d1d03 (2026-10-03) added `startup_unlock` to this route.
- The 5th node, `test_console_enrolls_…`, fails because `ready.select(15)` returns `[]`. Its fixture writes `config.write_text("preferences")` (`:151`), which is not valid TOML, so startup routes to `configuration_invalid` instead of the state the test means to set up.

**The inventories are out of date (2 nodes).**
- New bootstrap leaf imports: `effective_roots`, `Utils.platform_files` and `collections.abc`.
- About 20 package-main routes have no classification, for example `Chat/console_trace_regex_worker.py`, `Chunking/lab_runner.py` and `Backup_Recovery/detached_worker.py`.

Making `tldw-cli` exit with a stderr refusal instead of opening the recovery host would be a product change and needs an owner decision. This task updates the tests to match the shipped behaviour.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The route test asserts the shipped behaviour for both main routes and both parameter values: on a pending recovery, the recovery-only host opens (with minimal_recovery stubbed so the child exits) and tldw_chatbook.app is never imported
- [x] #2 The enrollment test's fixture writes a valid TOML config, so the test exercises the state it means to, and the node passes
- [x] #3 The two inventory nodes pass, with each new bootstrap import and package-main route classified in the notes
- [x] #4 All 7 nodes pass on dev (red before: 7 of 7 at 29592ecc9b), and the route test still fails when a main route imports tldw_chatbook.app before the recovery fence (negative control)
<!-- AC:END -->

## Implementation Plan

1. Re-confirm the 7 reds at base e8eeae7db9 (serial Tests/Architecture run).
2. Split the two main routes (`main_cli_runner()`, `runpy tldw_chatbook __main__`) out of the stderr-refusal parametrize into a route test that stubs `launcher.minimal_recovery` in the child, asserts it is called with `recovery_pending` (exit 0, as the real host returns), and fails if `tldw_chatbook.app` is ever imported. Keep the other routes' refusal test unchanged.
3. Enrollment fixture: write valid TOML so startup reaches enrollment instead of `configuration_invalid`.
4. Inventories: classify the bootstrap leaf imports and the package-main routes found at base; drop rows whose route is gone.
5. Negative control: a temporary `import tldw_chatbook.app` before the fence in `cli.py` turns the route test red; revert.

## Implementation Notes

Test-only change in `Tests/Architecture/test_recovery_entrypoints.py`; the product routing is unchanged.

- **Route test (AC#1).** The two main routes left the stderr-refusal parametrize for `test_main_route_opens_recovery_only_host_without_app_import[True/False × main_cli_runner / runpy tldw_chatbook]`. The child stubs `launcher.minimal_recovery`: it records the reason and returns 0, as the real host does after `run()`. A meta-path finder prints `app_import_before_fence` if `tldw_chatbook.app` is ever looked up. The test asserts the marker is absent, `recovery_only_host:recovery_pending` is present and the exit is 0. The old node ids (`test_real_route_refuses…[*-main routes]`) are gone; the other 10 routes keep the refusal test. The shared setup and the side-effect, config and `.local` assertions moved into `_run_pending_route`, unchanged.
- **Enrollment fixture (AC#2).** It wrote `"preferences"` (invalid TOML → `configuration_invalid` → the real recovery host → `ready.select(15)` empty). It now writes `"[preferences]\n"`, so startup reaches enrollment.
- **Inventories (AC#3), reviewed at e8eeae7db9.**
  - Bootstrap ImportFrom leaves, all present since b5251e9a6e, the commit that created the test (so this node was red at birth):
    - `collections.abc`: stdlib.
    - `tldw_chatbook.Utils.platform_files`: an os/fcntl shim that imports only `os`, `fcntl` or (Windows) `windows_files` (stdlib + ctypes).
    - `effective_roots`: a lazy sibling import inside `bootstrap.effective_roots`, reached only when an enrolled root is absent. It pulls `native_files`/`qualification` (pydantic), but no config, DB or app.
  - Package-main routes. Rows removed because the route is gone: `Utils/paths.py` (76b49e4fcd), `TTS/backends/higgs_voice_manager.py` (demo CLI dropped, f5212dd57e) and `Chunking/templates/example_usage.py` (deleted, 224e331af5). The higgs runpy refusal route still passes and stays.
  - New rows:
    - `recovery_host`: `Backup_Recovery/__main__.py`, which runs `recovery_main`, the same recovery CLI as `tldw-cli recovery`.
    - `app_spawned_pipe_worker`: a child that an admitted app or recovery process spawns. Its inputs are argv, pipes or staged paths; it loads no config and does not enroll (grep confirmed no config import in any of them). Members and their spawners:
      - `Audio/diarizer_worker.py`
      - `Audio/voice_process_entry.py` (`Chat/console_voice_process.py`)
      - `Backup_Recovery/age_worker.py` (`python -I`)
      - `Backup_Recovery/detached_worker.py` (`portable_relocation`)
      - `Backup_Recovery/rag_projection_validation.py`
      - `Chat/console_trace_regex_worker.py`
      - `Chunking/lab_runner.py`
      - `DB/private_sqlite_helper_entry.py` (`private_sqlite_process`)
      - `Terminal/posix_launcher.py` (`posix_backend`)
      - `Tools/workspace_tool_worker.py`
      - `Web_Server/artifact_share_server.py` (`artifact_share`)
    - `external_read_only_worker`: `Tools/project_index_scan_worker.py`, a stdlib-only scanner like `_grep_worker`.
    - `development_tool`: `Tools/build_remote_worker_bundle.py`.
    - `remote_host_worker`: `Tools/remote_worker_bundle.py`, the generated bundle, which runs on the SSH host.
- **Evidence (AC#4).**
  - Red before: all 7 nodes failed in the serial `Tests/Architecture` run at base e8eeae7db9 (40 failed / 4106 passed, 979 s).
  - After: the file passes, 32 passed, run serially through run.sh.
  - Negative control: a temporary `import tldw_chatbook.app` before the `startup_unlock` import in `cli.py` turned all 4 new route nodes red on `assert 'app_import_before_fence' not in …`. Reverted.

