---
id: TASK-42479
title: "The three-turn profiler's child environment drops PYTHON_KEYRING_BACKEND, so the app it runs gets the OS keyring backend"
status: To Do
assignee: []
created_date: 2026-11-23 11:13
updated_date: 2026-11-23 11:13
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
- [ ] #1 The child environment carries `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`, and `assert_child_environment` accepts that key only with that value
- [ ] #2 The allowlist tests (`test_build_child_environment_is_strictly_allowlisted`, `test_assert_child_environment_requires_sample_owned_paths`) pin the null backend, and a child environment without it, or with another backend, is refused
- [ ] #3 `Tests/Performance/test_console_three_turn_profile.py` passes
<!-- AC:END -->
