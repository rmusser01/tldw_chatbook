---
id: TASK-41962
title: "Console session settings: 18 nodes red and 8 teardown errors ('Native app workers remain uncertain') under the shared profile"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- console
priority: medium
references:
- Tests/UI/test_console_session_settings.py
- Tests/conftest.py
dependencies:
- TASK-41633
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** After TASK-41633's module marker, 18 nodes stay red (TASK-41633 notes F6) and 8 error at teardown with "Native app workers remain uncertain; config retirement refused" (F11). One test redirects `TLDW_CONFIG_PATH` itself, so the marker cannot fix it.

**Failing nodes (dev d0486978dd, TASK-41633 runs).**
- `Tests/UI/test_console_session_settings.py::test_console_global_name_refresh_failure_notifies_once`
- `Tests/UI/test_console_session_settings.py::test_console_inspector_hosts_staged_context_above_source_readiness`
- `Tests/UI/test_console_session_settings.py::test_console_readiness_unblocks_after_provider_save_without_restart`
- `Tests/UI/test_console_session_settings.py::test_console_settings_modal_core_first_hierarchy_and_title`
- `Tests/UI/test_console_session_settings.py::test_console_settings_result_applies_name_override_without_losing_prompt_source`
- `Tests/UI/test_console_session_settings.py::test_covered_cancelled_source_reopen_transfers_exact_draft_to_modal`
- `Tests/UI/test_console_session_settings.py::test_endpoint_command_aborts_when_settings_dismissed_before_template`
- `Tests/UI/test_console_session_settings.py::test_endpoint_command_layers_template_on_exact_settings_modal`
- `Tests/UI/test_console_session_settings.py::test_mounted_console_unmount_times_out_hung_refresh_and_repairs_on_resume`
- `Tests/UI/test_console_session_settings.py::test_open_console_settings_propagates_mount_cancellation`
- `Tests/UI/test_console_session_settings.py::test_open_console_settings_real_callback_stages_typed_credential_route`
- `Tests/UI/test_console_session_settings.py::test_open_console_settings_unwinds_exact_modal_after_mutating_failed_mount`
- `Tests/UI/test_console_session_settings.py::test_roleplay_writer_cleanup_waits_for_owner_acceptance`
- `Tests/UI/test_console_session_settings.py::test_roleplay_writer_startup_failure_releases_fork_transition[constructor]`
- `Tests/UI/test_console_session_settings.py::test_roleplay_writer_startup_failure_releases_fork_transition[start]`
- `Tests/UI/test_console_session_settings.py::test_source_reopen_revalidates_exact_owner_after_model_resolution`
- `Tests/UI/test_console_session_settings.py::test_suspended_open_uses_the_draft_pair_for_initial_discovery`
- `Tests/UI/test_console_session_settings.py::test_system_prompt_editor_clears_character_template_source`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The teardown errors have a recorded cause and are fixed
- [ ] #2 Each red node has a recorded cause and passes
<!-- AC:END -->
