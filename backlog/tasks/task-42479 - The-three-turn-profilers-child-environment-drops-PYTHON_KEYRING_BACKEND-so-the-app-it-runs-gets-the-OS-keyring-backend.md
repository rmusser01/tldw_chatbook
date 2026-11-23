---
id: TASK-42479
title: "The three-turn profiler's child environment drops PYTHON_KEYRING_BACKEND, so the app it runs gets the OS keyring backend"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 11:13
updated_date: 2026-11-23 23:10
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
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The child environment carries `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`, and `assert_child_environment` accepts that key only with that value
- [x] #2 The allowlist tests (`test_build_child_environment_is_strictly_allowlisted`, `test_assert_child_environment_requires_sample_owned_paths`) pin the null backend, and a child environment without it, or with another backend, is refused
- [x] #3 `Tests/Performance/test_console_three_turn_profile.py` passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red first, without a real child: in `Tests/Performance/test_console_three_turn_profile.py`, pin the null backend in the two allowlist tests (built env carries it even when the host value differs; `assert_child_environment` refuses an env without it or with another backend), and add a spawn-capture test that runs `_run_parent_child` with `run_child_with_watchdog` monkeypatched to record the `environment` it is handed. Run only those nodes at base (no real child): red.
2. Fix where the env is built: `build_child_environment` sets `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`; `assert_child_environment` accepts the key only with that value (with the other fixed flags).
3. Survey the other child-env builders in Tests/Performance and the profiling helpers; `run_workspace_tool_executor_profile.py::_isolated_environment` builds its child env from an allowlist too and drops the variable, so fix it the same way with a red-first assertion in its existing environment test. Builders that copy `os.environ` inherit the conftest pin and need nothing.
4. Green: the touched tests, then the whole `test_console_three_turn_profile.py` and `test_workspace_tool_executor_profile.py` files at `-n 4`; soak the new nodes.
5. Do not run the real profiler (its parent mode drives a real provider endpoint); notes, ACs, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause.** `build_child_environment` built the child's environment from an allowlist (PATH, LANG, LC_*, TERM, TZ, the sample-owned paths and three flags) that left out `PYTHON_KEYRING_BACKEND`, and `assert_child_environment` refused any other key. The child builds a real `TldwCli`, so any keyring use there would resolve the OS backend (the login Keychain on macOS).

**Fix (where the environment is built).**
- `Tests/Performance/run_console_three_turn_profile.py`: the builder sets `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` whatever the host sets; `assert_child_environment` allows the key and checks it with the other fixed flags, so a child environment without it, or with another backend, fails `child_environment_mismatch:flags`.
- Survey of the other child-environment builders in `Tests/Performance` and the profiling helpers: `run_workspace_tool_executor_profile.py::_isolated_environment` is the only other allowlist builder and dropped the variable the same way, so it now sets it too (its child imports the app). The rest copy `os.environ` and inherit the conftest pin, and `test_ui_ready_module_census.py`'s first-use probe already pins it. No follow-up task needed.

**Tests.** `test_build_child_environment_is_strictly_allowlisted` now passes a host macOS backend and pins the null one. `test_assert_child_environment_requires_sample_owned_paths` refuses an environment without the key and one with another backend, then pins the built value. The new `test_parent_spawns_each_child_with_the_null_keyring_backend` captures the environment `_run_parent_child` hands `run_child_with_watchdog` (a fake spawn; no child runs). `test_windows_isolated_environment_uses_only_temporary_home` pins the workspace profiler's key.

**Red/green (no real child at base; only these four nodes ran there).** Red at b31e3e9241: `assert None == 'keyring.backends.null.Keyring'`, `Failed: DID NOT RAISE`, `assert [None] == ['keyring.backends.null.Keyring']`, and the workspace key-set mismatch. Green after the fix: 4 passed.

**Whole files** (`-n 4`, load 41-46): `test_console_three_turn_profile.py` + `test_workspace_tool_executor_profile.py` gave 663 passed, 3 failed. Two passed on rerun (load). The third, `test_scripted_mounted_sample_uses_real_composer_queue_and_fs_write`, is red at base under the same load (serial, load 57-97: base failed 3 runs and a fourth printed no summary; this branch failed 5 of 5); an AST walk of `run_scripted_mounted_sample` shows it never reaches the changed builders. The real profiler was not run (its parent mode drives a provider endpoint).

Files: `Tests/Performance/run_console_three_turn_profile.py`, `Tests/Performance/run_workspace_tool_executor_profile.py`, `Tests/Performance/test_console_three_turn_profile.py`, `Tests/Performance/test_workspace_tool_executor_profile.py`.
<!-- SECTION:NOTES:END -->
