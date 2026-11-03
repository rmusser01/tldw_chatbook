---
id: TASK-36526
title: Bootstrap-profile Settings hub tests assign into the shared cached config
status: Won't Do
updated_date: 2026-11-02 21:52
created_date: 2026-10-24 00:26
labels:
- follow-up
- tests
priority: low
references:
- Tests/UI/test_settings_configuration_hub.py
- Tests/UI/app_factory.py
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Won't Do: TASK-36478 removed the trap (2026-11-02, dev 88af9f9379).
- TASK-36478 (#3254, merged 2026-10-26) handles each bootstrap-profile test in `Tests/conftest.py` (`_restore_bootstrap_profile_config`). It writes the shared `config.toml` back to the bytes the test started with, then drops config.py's caches. The next test's first read therefore rebuilds the dict from the restored file. That also undoes an in-place edit of `app.app_config`, the case this task was about; that function's docstring names it.
- As a smoke run only, the three tests named above, run in one process with `-p no:randomly` through the scratch-profile wrapper, gave 3 passed. They already passed on dev before TASK-36478, so the decision rests on the code above.
- `Tests/UI/app_factory.py`'s docstring still limits `bootstrap_profile` to tests that neither save config nor assign into `app.app_config`. That rule is now stricter than needed, but it is harmless, so it stays.
<!-- SECTION:NOTES:END -->
