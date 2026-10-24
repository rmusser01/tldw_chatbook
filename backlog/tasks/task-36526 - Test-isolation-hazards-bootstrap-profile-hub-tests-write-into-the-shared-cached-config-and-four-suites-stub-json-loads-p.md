---
id: TASK-36526
title: 'Test isolation hazards: bootstrap-profile hub tests write into the shared cached config, and four suites stub json.loads process-wide'
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

**Process-wide json.loads stubs.** `monkeypatch.setattr(module.json, "loads", stub)` replaces the stdlib function for the whole process. TASK-36381's red came from exactly this: the probe's storage admission called `json.loads(data, object_pairs_hook=...)` and got a `TypeError` (lesson in lessons-testing-evidence.md). These stubs accept only one positional argument, so any guarded read that runs inside their window fails the same way:
- `test_sync_crypto.py:58-61`
- `test_prompts_db_retained_history.py:433-455`
- `test_run_log_service_wiring.py:688-704`
- `test_model_discovery_disk_cache.py:157-162`

The four files pass on dev (178 passed together with the discovery file, whose one failure is the TASK-36381 red).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No bootstrap-profile test in test_settings_configuration_hub.py saves config or assigns into app.app_config; each one that did runs on a private profile or works on a copy
- [ ] #2 The four named suites scope their json.loads stubs to the module under test, and each still fails when the behaviour it pins is removed
<!-- AC:END -->
