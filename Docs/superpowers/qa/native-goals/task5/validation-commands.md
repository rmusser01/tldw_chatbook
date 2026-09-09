# Reproduce targeted checks

Run in the native-goal-runs worktree. All application tests use the isolated
`Tests/conftest.py` configuration and real SQLite. Set:

```sh
PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
```

Amended98-pass scope (one opt-in live skip):

```sh
TLDW_GOAL_QUALIFICATION_ARTIFACT=/private/tmp/task5-deterministic-final-result.json "$PY" -m pytest Tests/UI/test_console_goal_setup.py Tests/UI/test_console_goal_controls.py Tests/UI/test_console_goal_navigation.py Tests/UI/test_console_goal_settings.py Tests/Chat/test_console_goal_dispatch.py Tests/Chat/test_console_goal_scheduling.py Tests/Chat/test_console_runtime_lifetime.py Tests/Chat/test_goal_cli_verification.py Tests/Architecture/test_screen_size_ratchet.py -q --tb=short --show-capture=no
```

Final selection identity/removal (10 passed) and double-Start (6 passed) followups:

```sh
"$PY" -m pytest Tests/UI/test_console_goal_controls.py Tests/UI/test_console_goal_navigation.py -q --tb=short --show-capture=no
"$PY" -m pytest Tests/UI/test_console_goal_setup.py -q --tb=short --show-capture=no
```

Existing canonical save/revert (3 passed,355 deselected):

```sh
"$PY" -m pytest Tests/UI/test_console_goal_settings.py Tests/UI/test_settings_configuration_hub.py -k 'mounted_goal_policy or console_behavior_uses_batched_save_adapter or console_behavior_revert_restores_global_defaults' -q --tb=short --show-capture=no
```

Architecture/privacy/migration (97 passed,3 proved baseline failures,1 historical skip):

```sh
"$PY" -m pytest Tests/DB/test_private_sqlite_inventory.py Tests/Architecture/test_persistent_diagnostic_inventory.py Tests/Architecture/test_console_fleet_ownership.py Tests/DB/test_goal_runs_migration.py Tests/Agents/test_goal_run_service.py -q --tb=short --show-capture=no
```

The live command below was run finitely twice against the authorized loopback
endpoint, once before the final-report wording correction and once after it. Both
are quality failures; no further live calls are part of final verification.

```sh
TLDW_GOAL_LIVE_LOCAL=1 TLDW_GOAL_LIVE_ARTIFACT=/private/tmp/task5-live-final-result.json "$PY" -m pytest Tests/Chat/test_goal_cli_verification.py -k configured_local_goal_live -q --tb=short --show-capture=no
```

Rendered capture:

```sh
TLDW_GOAL_SCREENSHOT_DIR=/private/tmp/task5-screens-final "$PY" -m pytest Tests/UI/test_console_goal_navigation.py -k production -q --tb=short --show-capture=no
/usr/bin/qlmanage -t -s 2000 -o /private/tmp/task5-screens-final /private/tmp/task5-screens-final/goal-review-80x24.svg /private/tmp/task5-screens-final/goal-review-160x44.svg /private/tmp/task5-screens-final/goal-setup-80x24.svg /private/tmp/task5-screens-final/goal-setup-160x44.svg
```

Final static checks (the same path list applies to `ruff format --check`):

```sh
"$PY" -m ruff check Tests/UI/test_console_goal_*.py Tests/Chat/test_console_goal_dispatch.py Tests/Chat/test_console_goal_scheduling.py Tests/Chat/test_goal_cli_verification.py tldw_chatbook/UI/Console_Modules/goals.py tldw_chatbook/UI/Screens/settings_goal_policy.py tldw_chatbook/Widgets/Console/console_goal_*.py tldw_chatbook/Agents/goal_iteration.py tldw_chatbook/Agents/goal_run_service.py tldw_chatbook/Chat/console_goal_runs.py
git diff --check
```

Legacy owners use the documented baseline diagnostic comparison and scoped
AST-preserving formatting; these commands do not claim the whole repository is
lint-clean. The test counts overlap and are not additive.
