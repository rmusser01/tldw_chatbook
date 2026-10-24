---
id: TASK-36526
title: Bootstrap-profile Settings hub tests assign into the shared cached config
status: To Do
created_date: 2026-10-24 00:26
labels:
- follow-up
- tests
priority: low
references:
- Tests/UI/test_settings_configuration_hub.py
- Tests/UI/app_factory.py
- Tests/Sync_Interop/test_sync_crypto.py
- Tests/Prompts_DB/test_prompts_db_retained_history.py
- Tests/Agents/test_run_log_service_wiring.py
- Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py
- backlog/docs/lessons-testing-evidence.md
dependencies:
- TASK-36381
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36381's implementation and review (branch fu/h-settingsreds, 2026-10-23). Every test named here passes on dev today. Each one is a trap for the next change near it.

**Shared cached config.** `build_test_app_config()` returns `load_settings()`'s cached dict by reference, on purpose (app_factory.py:99-104). On the shared bootstrap profile, a test that assigns into `app.app_config` therefore changes the config the next test in that worker reads. TASK-36381 hit this: putting 40 sync hub tests on `bootstrap_profile` made the untouched `test_settings_console_behavior_saves_display_name_exactly` fail. Its branch's `app_factory.py` docstring now limits `bootstrap_profile` to tests that neither save config nor assign into `app.app_config`. Three bootstrap-profile tests in `test_settings_configuration_hub.py` do assign into it, on dev and on that branch:
- `test_settings_console_behavior_saves_display_name_exactly` (dev :7547)
- `test_settings_provider_category_saves_provider_defaults_without_sampling` (:9453)
- `test_settings_provider_switch_does_not_save_stale_endpoint` (:11067)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No bootstrap-profile test in test_settings_configuration_hub.py saves config or assigns into app.app_config; each one that did runs on a private profile or works on a copy
<!-- AC:END -->
