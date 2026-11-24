---
id: TASK-42479
title: "The three-turn profiler's child environment drops PYTHON_KEYRING_BACKEND, so the app it runs gets the OS keyring backend"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 11:13
updated_date: 2026-11-24 02:15
labels:
- follow-up
- tests
- performance
- hardening
priority: low
references:
- Tests/Performance/run_console_three_turn_profile.py
- Tests/Performance/test_console_three_turn_profile.py
- tldw_chatbook/app_service_wiring.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), while TASK-42477 surveyed how child processes inherit test isolation. Checked on `fu/r22-keychain` (62531d037a plus TASK-42477).

**Why.** `Tests/conftest.py` pins `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` before any import (TASK-19570 A), so tests never reach the OS credential store, and children built from `os.environ` inherit it. The three-turn profiler's child does not. `build_child_environment` (`Tests/Performance/run_console_three_turn_profile.py:4092`) builds the environment from `_SAFE_INHERITED_ENVIRONMENT` (PATH, LANG, LC_ALL, LC_CTYPE, TERM, TZ), plus the sample-owned HOME, XDG, TMPDIR and config paths and three flags. `assert_child_environment` (:4118) refuses any other key.

That child builds a real `TldwCli` and mounts it through `ConsoleHarness` (:5550). `TldwCli.__init__` wires a credential-store factory; on first use it calls `build_default_server_credential_store()`, which calls `keyring.get_keyring()` (`app_service_wiring.py::_resolve_server_credential_store`, lazy since TASK-21111(b)). Any keyring use in that child would therefore get the real OS backend, the macOS Keychain on a developer machine. Whether the profiled flow reaches that first use, or any other `keyring` caller, was not measured.
**AC amendment (2026-11-24, review of `fu/r24-misc`).** ACs #4-#6 were added before any further code. The review found the same drop outside the three-turn profiler: `run_console_mount_profile.py::_configure_isolated_profile` moves HOME and the config paths but leaves the host's backend, and three more sites move HOME and build or launch the app without the pin. Keychain safety is the priority, so the coordinator decided to close the whole class in this task rather than file a follow-up: pin every such site, guard the class with a census, and lane the pin tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The child environment carries `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`, and `assert_child_environment` accepts that key only with that value
- [x] #2 The allowlist tests (`test_build_child_environment_is_strictly_allowlisted`, `test_assert_child_environment_requires_sample_owned_paths`) pin the null backend, and a child environment without it, or with another backend, is refused
- [x] #3 `Tests/Performance/test_console_three_turn_profile.py` passes (except `test_scripted_mounted_sample_uses_real_composer_queue_and_fs_write`, red at base and already tracked by TASK-33372)
- [x] #4 Every other site in `Tests/`, `Helper_Scripts/` and `scripts/` that moves HOME and builds or launches the app pins `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` (at least `run_console_mount_profile.py`, `Tests/ProductionApp/test_config_profile_isolation.py`, `Helper_Scripts/verify_notes_files_sync_tui.py` and `scripts/update_boot_budget_snapshots.py`, before any app import), and each env builder that can be called on its own is proven by an env-capture or `mock.patch.dict` test, never by a real spawn without the pin
- [x] #5 A census guard in the Architecture guards step scans `Tests/`, `Helper_Scripts/` and `scripts/` and fails when an env dict literal, `os.environ.update(...)` or `env[...] =` sets HOME together with `TLDW_CONFIG_PATH`, `TLDW_TEST_MODE` or an XDG path but not `PYTHON_KEYRING_BACKEND`. It is red before the pins and names the sites, green after, has a negative control and a floor so an empty scan cannot pass, and has a matching contract row
- [x] #6 The profiler builders' keyring-pin tests run in the Architecture guards step next to `Tests/test_keyring_isolation.py`, with matching `ARCHITECTURE_GUARD_TARGETS` rows
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red first, without a real child: in `Tests/Performance/test_console_three_turn_profile.py`, pin the null backend in the two allowlist tests (built env carries it even when the host value differs; `assert_child_environment` refuses an env without it or with another backend), and add a spawn-capture test that runs `_run_parent_child` with `run_child_with_watchdog` monkeypatched to record the `environment` it is handed. Run only those nodes at base (no real child): red.
2. Fix where the env is built: `build_child_environment` sets `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`; `assert_child_environment` accepts the key only with that value (with the other fixed flags).
3. Survey the other child-env builders in Tests/Performance and the profiling helpers; `run_workspace_tool_executor_profile.py::_isolated_environment` builds its child env from an allowlist too and drops the variable, so fix it the same way with a red-first assertion in its existing environment test. Builders that copy `os.environ` inherit the conftest pin and need nothing.
4. Green: the touched tests, then the whole `test_console_three_turn_profile.py` and `test_workspace_tool_executor_profile.py` files at `-n 4`; soak the new nodes.
5. Do not run the real profiler (its parent mode drives a real provider endpoint); notes, ACs, Done.

Review amendment (2026-11-24, ACs #4-#6):
6. Write the census guard first, in `Tests/test_keyring_isolation.py` (already laned). It is an AST scan over `Tests/`, `Helper_Scripts/` and `scripts/` with a negative control on synthetic sources, a floor, and an exact list of justified exceptions. Run it red on the current tree, which names the unpinned sites.
7. Write the env-capture/`mock.patch.dict` tests for each callable builder (mount profiler, boot-budget script, Notes sync verifier, citation benchmark, live-API helpers, MCP and backup-helper environments) and run them red; no real child runs.
8. Pin every reported site; green; re-run the guard green.
9. Lane the four profiler pin nodes beside `Tests/test_keyring_isolation.py`, with contract rows; soak the touched tests 20 times at `-n 4`; re-run the brief's check list, including the Architecture guards step and serial Tests/Architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause.** `build_child_environment` built the child's environment from an allowlist (PATH, LANG, LC_*, TERM, TZ, the sample-owned paths and three flags) that left out `PYTHON_KEYRING_BACKEND`, and `assert_child_environment` refused any other key. The child builds a real `TldwCli`, so any keyring use there would resolve the OS backend (the login Keychain on macOS).

**Fix (where the environment is built).**
- `Tests/Performance/run_console_three_turn_profile.py`: the builder sets `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` whatever the host sets; `assert_child_environment` allows the key and checks it with the other fixed flags, so a child environment without it, or with another backend, fails `child_environment_mismatch:flags`.
- Survey of the other child-environment builders in `Tests/Performance` and the profiling helpers: `run_workspace_tool_executor_profile.py::_isolated_environment` is the only other allowlist builder and dropped the variable the same way, so it now sets it too (its child imports the app). The rest copy `os.environ` and inherit the conftest pin, and `test_ui_ready_module_census.py`'s first-use probe already pins it. **Corrected after review:** this survey covered only child-environment builders and missed in-process moves. `run_console_mount_profile.py::_configure_isolated_profile` moves HOME and the config paths in `os.environ` but kept the host's backend, and a repo-wide scan found 18 more sites. They are fixed under ACs #4-#6 below, not filed as a follow-up.

**Tests.** `test_build_child_environment_is_strictly_allowlisted` now passes a host macOS backend and pins the null one. `test_assert_child_environment_requires_sample_owned_paths` refuses an environment without the key and one with another backend, then pins the built value. The new `test_parent_spawns_each_child_with_the_null_keyring_backend` captures the environment `_run_parent_child` hands `run_child_with_watchdog` (a fake spawn; no child runs). `test_windows_isolated_environment_uses_only_temporary_home` pins the workspace profiler's key.

**Red/green (no real child at base; only these four nodes ran there).** Red at b31e3e9241: `assert None == 'keyring.backends.null.Keyring'`, `Failed: DID NOT RAISE`, `assert [None] == ['keyring.backends.null.Keyring']`, and the workspace key-set mismatch. Green after the fix: 4 passed.

**Whole files** (`-n 4`, load 41-46): `test_console_three_turn_profile.py` + `test_workspace_tool_executor_profile.py` gave 663 passed, 3 failed. Two passed on rerun (load). The third, `test_scripted_mounted_sample_uses_real_composer_queue_and_fs_write`, is red at base under the same load (serial, load 57-97: base failed 3 runs and a fourth printed no summary; this branch failed 5 of 5). TASK-33372 (To Do) already tracks it: its ScriptedGateway fake lacks `cached_context_window`; an AST walk of `run_scripted_mounted_sample` shows it never reaches the changed builders. The real profiler was not run (its parent mode drives a provider endpoint).

Files (first pass): `Tests/Performance/run_console_three_turn_profile.py`, `Tests/Performance/run_workspace_tool_executor_profile.py`, `Tests/Performance/test_console_three_turn_profile.py`, `Tests/Performance/test_workspace_tool_executor_profile.py`.
**Review amendment (ACs #4-#6, 2026-11-24): the class is closed.**
- **Census guard** (`Tests/test_keyring_isolation.py`, already a target of the Architecture guards step, so its existing contract row covers it). It is an AST scan of `Tests/`, `Helper_Scripts/` and `scripts/`, taking each function and each module top level as one scope.
  - **What it tracks (re-review rework).** Each mapping a scope fills is tracked on its own: a name (`env = {...}`, then `env.update(...)`, `env["K"] = ...`, `dict(env, K=...)`), `os.environ` itself, or an unnamed literal. A mapping moves the profile when it gets HOME together with `TLDW_CONFIG_PATH`, `TLDW_TEST_MODE` or an XDG path. A table copied in by a `.items()` comprehension or loop (`x[key] = ...`, `monkeypatch.setenv`) counts for the mapping it fills.
  - **What pins it.** The mapping must set `PYTHON_KEYRING_BACKEND` (dict key, keyword or subscript) to the constant `keyring.backends.null.Keyring`, or to a module constant holding it. Another value does not count, and a `pop`/`del` of the key undoes it. The one inherited case is a pytest file (`test_*.py`, `conftest.py`) whose mapping is a copy of the running environment (`{**os.environ}`, `dict(os.environ)`, `os.environ.copy()`) or `os.environ` itself, because the conftest has already pinned it there.
  - **Corrected after re-review.** The first version also counted `os.environ.items()` and any `setenv` in the scope as a copy, which excused allowlists and denylists built from `os.environ.items()`. Those are fresh environments, and three were unpinned (below). It also accepted the key's bare name, a non-null value and a `pop`.
  - **Limits** (in the census docstring): it reads one scope at a time, so HOME and the profile keys set in different helpers are not seen together (`test_installed_distribution.py::_sanitized_build_env` sets HOME while `_installed_env` adds the XDG paths; it is pinned by hand), and a mapping handed to another function to be filled in is not followed.
  - **Exceptions.** There are 11 exact `(path, scope)` pairs, each with a reason. Eight were there before: a Node VM probe, six Git-only or ambient-Git environments, and `Tests/private_profile.py`, where forcing the key would override the `TLDW_TEST_REAL_KEYRING` opt-out. Three are new: the portable launcher's host-environment input, which that test asserts the launcher pins to null; the benchmark restore test's in-process host seed, which uses the `fail` backend; and the closeout runner's expected-values table. A stale pair fails the test.
  - **Negative control and floor.** Fourteen synthetic sources cover scratch unpinned, pinned, pinned by a module constant, a macOS value, a `pop` from a copy, a copy under pytest, the same copy in a script, an `os.environ.items()` allowlist, a scope-level `setenv`, a module-level keyword, a keyword pin, a pinned path table, an unpinned path table, and no move. Against the previous scanner, four of them were red: the macOS value, the `pop`, the allowlist and the `setenv`. A floor of 120 profile moves stops an empty scan from passing (174 are found today); with `_CENSUS_TREES` emptied by a temporary edit (reverted), the test failed with `the census saw only 0 profile moves`.
- **Red before the pins.** The census named 19 sites. They are listed with the fixes below. After the re-review rework it named three more (`test_native_multivolume.py`, `test_recovery_profile_terminal.py`, `test_logging_pipeline_single_pass.py`).
- **Re-review pins.** Each of the three sites' environment moved into a helper with no behaviour change (`_image_case_environment`, `_terminal_child_environment`, `_sandboxed_wiring_environment`), and each helper now pins the backend. The terminal test is macOS-only and its child mounts `TldwCli`, so on a Mac it reached the login Keychain. `_sanitized_build_env` is pinned too, with the key added to `_BUILD_ENV_KEYS`. Each has a red-first builder-capture test, and no real child ran: `test_terminal_child_environment_pins_the_null_keyring_backend`, `test_image_case_environment_pins_the_null_keyring_backend`, `test_sandboxed_wiring_environment_pins_the_null_keyring_backend` (host backend set to `fail`), and an assertion in `test_sdist_build_env_excludes_host_state`. Red: `assert None == 'keyring.backends.null.Keyring'` (or the `fail` value), 4 nodes; green after. Following the re-review rule, the three files' spawning tests were not run; collection is clean (30 tests).
- **Pins** (`PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` in the same env literal or update that moves HOME):
  - in-process, before any app import: `Tests/Performance/run_console_mount_profile.py::_configure_isolated_profile`, `scripts/update_boot_budget_snapshots.py::_isolate_config`, `Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py::isolated_benchmark_host_state` (restored on exit like the other overrides);
  - child environments: `Helper_Scripts/verify_notes_files_sync_tui.py::build_isolated_environment` (edited only, never run), `Tests/ProductionApp/test_config_profile_isolation.py`, `test_service_composition_lifecycle.py::_run_lifecycle_case`, `Tests/Backup_Recovery/test_complete_roundtrip.py` (`_capture_two_profiles`, `_isolated_environment`), `test_recovered_media_crashes.py`, `test_tts_profile_lock_inventory.py`, the four `Tests/Chat/test_live_*_api.py::_live_subprocess_environment`, `Tests/Chat/workflow_wave_archive.py::restore_and_rebackup`, `Tests/Live/persona_buddy_terminal_probe.py::_run_child`, `Tests/MCP/test_mcp_unified_stdio.py::_isolated_subprocess_environment`, `Tests/Packaging/test_backup_helper_distribution.py::_isolated_runtime_environment`, `Tests/Packaging/test_settings_preimport_import_closure.py`.
- **Env-capture proofs, each red first, no real child.** They use `mock.patch.dict(os.environ, {backend: fail})` or call the builder directly:
  - the mount profiler, the boot-budget script and the Notes sync verifier's builder (loaded by path; only its environment function runs), all in `Tests/test_keyring_isolation.py`;
  - the four live-API isolation tests, `test_restore_child_environment_pins_the_null_keyring_backend` and `test_isolated_runtime_environment_pins_the_null_keyring_backend`;
  - the citation benchmark's restore test, which now sets the backend to the `fail` backend, expects the null one inside the context, and expects the `fail` one back afterwards.
  - Red: `assert None == 'keyring.backends.null.Keyring'` (or the `fail` value), 10 nodes. Green after the pins.
- **Covered by the census only.** The MCP builder's test module is skipped here (the `mcp-unified` extra is not installed), so its proof is the census. The envs built inline in a test body that also spawns (config isolation, lifecycle, recovered media, TTS lock, settings preimport, roundtrip capture, workflow wave, persona probe) are also covered by the census alone: capturing them would need a spawn, and before the pin that spawn would lack it.
- **Lane (AC #6).** The four profiler pin nodes now follow `Tests/test_keyring_isolation.py` in "Run Architecture guard contracts", and `ARCHITECTURE_GUARD_TARGETS` has the same rows. Timed with the step's flags (serial, `--timeout=180`): 18 passed in 19.6 s at load ~50; the census is about 10 s of that.
- **Touched spawning files, run once after the pins** (`-n 4`, load 50-70): 164 passed, 8 skipped (live, opt-in), 9 failed, none from the pin.
  - The four `test_workflow_native_journeys.py` failures and `test_runtime_backend_transition_detaches_and_rebinds_notes_organization` (`RecoveryRequired` in-process) fail the same way at base. Their child environments copy the pinned pytest environment, so running them at base was safe.
  - `test_production_app_composes_one_stable_dependency_graph` fails on `AttributeError: 'TldwCli' object has no attribute 'notes_organization_sync_service'`: the test is stale against the app.
  - The three `test_backup_helper_distribution.py` install tests fail at the offline `uv pip install` step (`setuptools` not in the cache), before the pinned environment is used.
  - These last two were not run at base, because their base children would lack the pin.
- **Soak.** The 25 new and changed tests (the keyring file, the 4 profiler nodes, the 7 pin assertions) passed 20 of 20 runs at `-n 4` (500 of 500), at a 1-minute load of 59-110.
- **Checks.**
  - Architecture guards step via `r19-step.sh` (`-p no:xdist --timeout=180`, 53 targets): 824 passed, 1 skipped, 9:54 at load 14-52. An earlier run at load 80-93 timed out `test_timer_path_static_update_inventory.py`'s census (>180 s; it took 146 s on a rerun at load 76-102) and hit one Personas notice flake that ran before any of these targets. Both files passed on that rerun and in the clean run.
  - `Tests/Architecture` serial: 1492 passed, 5 skipped.
  - `Tests/CI` `-n 4`: 1823 passed, 2 skipped.
  - `preflight.sh`: rc 0. `ruff --select F`: clean. `ruff_gate.py`: rc 0.

- **Re-review checks** (after the census rework):
  - 25 touched tests (the keyring file plus the 4 new builder captures) passed 20 of 20 runs at `-n 4` (500 of 500), at a 1-minute load of 14-25.
  - `Tests/test_keyring_isolation.py` with the Architecture guards flags (serial, `--timeout=180`): 21 passed in 8.7 s at load 14-20, the census 4.2 s.
  - `Tests/CI` `-n 4`: 1823 passed, 2 skipped.
  - `preflight.sh`: rc 0. `ruff --select F`: clean. `ruff_gate.py`: rc 0.
  - No real child, profiler or live helper was run.

Files (re-review): `Tests/test_keyring_isolation.py`, `Tests/Backup_Recovery/test_recovery_profile_terminal.py`, `Tests/Backup_Recovery/test_native_multivolume.py`, `Tests/test_logging_pipeline_single_pass.py`, `Tests/Packaging/test_installed_distribution.py`, `.github/workflows/derived-artifacts.yml` (timing comment).

Files (amendment): `Tests/test_keyring_isolation.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, the 19 pinned sites above, `Tests/Backup_Recovery/test_complete_roundtrip.py`, `Tests/Packaging/test_backup_helper_distribution.py`, `Tests/Performance/test_rag_citation_provenance_benchmark.py`, the four `Tests/Chat/test_live_*_api.py`.
<!-- SECTION:NOTES:END -->
