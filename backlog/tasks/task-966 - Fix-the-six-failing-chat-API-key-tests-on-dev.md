---
id: TASK-966
title: Fix the six failing chat API key tests on dev
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 18:06'
updated_date: '2026-07-27 18:42'
labels:
  - ui
  - tests
  - dev-baseline
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_tools_settings_window.py's six test_chat_api_key_* tests fail on pristine origin/dev with KeyError: 'openai'. They have been the standing baseline noise for every PR in the path-naming audit series and were verified pre-existing by stash bisection and by running the file on a pristine dev worktree. Standing baseline failures are corrosive: they train reviewers to skim red output, which is exactly how a real regression slips through.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The six tests pass on a clean checkout,The KeyError root cause is fixed rather than the assertion relaxed,The file has no remaining expected failures
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the six test_chat_api_key_* failures and read full tracebacks + captured logs, not just the KeyError.
2. Trace where the config path actually resolves at runtime (get_provider_readiness / save_setting_to_cli_config / load_cli_config_and_ensure_existence all go through _get_effective_config_path()) versus what the test's mock_config_path fixture patches (DEFAULT_CONFIG_PATH).
3. Check whether Tests/conftest.py's autouse environment-isolation fixture sets an environment variable that outranks the module-attribute patch.
4. Fix at the correct layer once the precedence is confirmed.
5. Re-run the six tests plus the whole file for regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause confirmed via full traceback + captured stderr logs, not the KeyError alone: the six tests use mount_settings_window(), which monkeypatches tldw_chatbook.config.DEFAULT_CONFIG_PATH to temp_config_path. But every code path these tests exercise (get_provider_readiness, save_setting_to_cli_config -> apply_settings_mutation_to_cli_config, load_cli_config_and_ensure_existence) resolves its path via _get_effective_config_path(), which checks the TLDW_CONFIG_PATH environment variable FIRST and only falls back to DEFAULT_CONFIG_PATH when that env var is unset. Tests/conftest.py's autouse isolate_test_environment fixture (applies to every test in the whole suite) always sets TLDW_CONFIG_PATH to a sandbox path under a DIFFERENT tmp_path-derived directory -- so the DEFAULT_CONFIG_PATH patch was silently a no-op for this module's newer async-pilot tests: they were reading/writing a config file the test never touched, while asserting against temp_config_path, which never got written. That produced both symptoms: '' where a configured key was expected (never loaded), and KeyError: 'openai' after save (temp_config_path's on-disk api_settings stayed exactly as the test wrote it -- {} -- because save also went to the sandbox path, not temp_config_path).

This is a test bug, not a production bug: the TLDW_CONFIG_PATH-over-DEFAULT_CONFIG_PATH precedence is intentional and load-bearing elsewhere (it's exactly the mechanism Tests/conftest.py's own sandboxing relies on to keep every test off the user's real config). The older AppTest-based tests in this same file that exercise the raw config-text-area save/reload path are all skipped in this environment (Textual's AppTest is unavailable), so they never surfaced this.

Fixed by making the file's own autouse mock_config_path fixture also monkeypatch.setenv('TLDW_CONFIG_PATH', str(temp_config_path)), so it actually controls the path every real code path resolves, instead of a module attribute nothing reads once the env var is set. No production code changed.

Before: 6 failed / 40 passed / 16 skipped (Tests/UI/test_tools_settings_window.py). After: 46 passed / 16 skipped (unrelated AppTest-unavailable skips, unchanged), 0 failed.
<!-- SECTION:NOTES:END -->
