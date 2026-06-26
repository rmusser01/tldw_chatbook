# Latest Dev Core App Usability Smoke Evidence

Date: 2026-06-26
Task: TASK-139
Branch: codex/next-app-usability-scan
Base HEAD: 2fef598b (Merge pull request #562 from rmusser01/codex/retire-legacy-entrypoints)

## Scope

- Covered clean first-run Home launch and non-Sync/non-Persona first-use routes: Home, Console, Library, Skills, MCP, Settings.
- Covered Home model-setup recovery into Settings > Providers & Models.
- Covered deterministic Library-to-Console staged context.
- Explicitly excluded Sync and Persona implementation ownership.

## Readiness Gate

TASK-88 remains blocked. The current Unified MCP client/service/schema surfaces expose runtime, governance, tools, external server, credential, ACP, path, and workspace operations, but not a confirmed server-first persisted MCP defaults contract for Settings to own safely.

## Red Findings

- Settings Overview leaked the full local config path in first-run rendered content.
- Home "Set up Console model" opened the legacy model route instead of taking the user to the Settings/provider configuration recovery path.

## Fixes

- Settings Overview now shows a non-sensitive config file label while leaving explicit Storage/Diagnostics detail surfaces unchanged.
- Home model setup now routes to Settings and passes existing `providers-models` navigation context.
- Added focused smoke coverage and tightened Home/Settings regression tests around these behaviors.

## Verification

- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_latest_dev_core_app_usability_smoke.py --tb=short`
  - Result: 3 passed, 1 warning.
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Home/test_dashboard_state.py Tests/UI/test_home_screen.py::test_home_selected_action_uses_user_facing_route_labels Tests/UI/test_settings_configuration_hub.py::test_settings_overview_config_path_label_hides_local_directory Tests/UI/test_settings_configuration_hub.py::test_settings_first_slice_categories_have_real_content --tb=short`
  - Result: 20 passed, 8 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_home_screen.py Tests/UI/test_product_maturity_phase1_first_run.py --tb=short`
  - Result: 43 passed, 1 warning.
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase1_empty_setup_states.py --tb=short`
  - Result: 6 passed, 1 warning.
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_latest_dev_core_app_usability_smoke.py Tests/UI/test_product_maturity_phase1_navigation_smoke.py Tests/UI/test_product_maturity_phase1_core_loop.py --tb=short`
  - Result: 7 passed, 1 warning.
- `git diff --check`
  - Result: passed.

Warnings were existing dependency/import warnings and did not affect the smoke outcome.

## Follow-Ups

No new backlog defects were opened from this slice.
