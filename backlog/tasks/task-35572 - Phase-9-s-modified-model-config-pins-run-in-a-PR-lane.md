---
id: TASK-35572
title: Phase 9's modified model-config pins run in a PR lane
status: To Do
created_date: 2026-10-19 04:30
dependencies:
- TASK-35570
labels:
- model-config-redesign
- follow-up
- ci
priority: low
references:
- scripts/ui_pr_gate_census.txt
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
- backlog/docs/lessons-testing-evidence.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in Phase 9's fix-round review (2026-10-19). Phase 9 placed every NEW model-config evidence test in a PR lane, but 31 test functions it MODIFIED still run in no PR lane: the census, the PR Fast Lane's steps and the P0 gate leave them to nightly-deep and the main-push Core Tests. Some are pins the plan says were rewritten on purpose (for example `test_console_settings_modal_enumerated_inputs_list_accepted_values` and `test_mounted_no_resume_keeps_blocking_modal_focus_transfer`), so a later PR can undo what they pin and still merge green.

The list was taken with `Tests/CI/test_console_p0_regression_gate.py`'s `_lanes()` / `_selected_by`, over the test functions present at both ends of Phase 9's range whose source differs, and marked with Tests/CI's `_uses_bootstrap_profile`. 7 are bootstrap_profile and 24 are not:

- `Tests/LLM_Calls/test_kobold_tabby_config.py::test_configuration_reaches_the_public_analyze_boundary` (TASK-33001.14, bootstrap_profile)
- `Tests/LLM_Calls/test_llama_summarizer_config.py::test_budget_and_diagnostic_reach_the_public_analyze_boundary` (TASK-33001.14, bootstrap_profile)
- `Tests/UI/test_console_cost_chip_screen.py::test_idle_draft_edit_burst_coalesces_one_cost_refresh` (TASK-33006.20, bootstrap_profile)
- `Tests/UI/test_console_roleplay_resume_navigation.py::test_mounted_no_resume_keeps_blocking_modal_focus_transfer` (TASK-33001.12, bootstrap_profile)
- `Tests/UI/test_console_settings_geometry.py::test_new_focus_supersedes_pending_suspended_anchor_and_is_revealed` (TASK-33006.24, bootstrap_profile)
- `Tests/UI/test_console_workbench_contract.py::test_console_ready_empty_transcript_exposes_activation_panel_copy` (TASK-33005.7, bootstrap_profile)
- `Tests/UI/test_console_workbench_contract.py::test_console_ready_empty_transcript_omits_setup_action_row` (TASK-33005.7, bootstrap_profile)
- `Tests/Chat/test_console_onboarding_state.py::test_setup_complete_collapses_to_ready_line` (TASK-33005.7)
- `Tests/Chat/test_encrypted_api_key_never_sent.py::test_summarizers_never_send_ciphertext` (TASK-33001.14)
- `Tests/Internal_Prompts/test_summarization_migration.py::test_local_template_fallback_reaches_the_core_user_content` (TASK-33001.14)
- `Tests/UI/test_console_internals_decomposition.py::test_console_empty_ready_transcript_shows_first_run_guidance` (TASK-33005.7)
- `Tests/UI/test_console_internals_decomposition.py::test_console_empty_transcript_uses_compact_ready_state` (TASK-33005.7)
- `Tests/UI/test_console_internals_decomposition.py::test_console_native_transcript_is_visible_transcript_surface` (TASK-33005.7)
- `Tests/UI/test_console_internals_decomposition.py::test_console_on_screen_resume_resyncs_stale_setup_modal_block` (TASK-33005.7)
- `Tests/UI/test_console_internals_decomposition.py::test_console_setup_modal_dismisses_when_readiness_and_model_satisfied` (TASK-33005.7)
- `Tests/UI/test_console_narrow_layout.py::test_console_narrow_60x18_renders_ready_empty_state` (TASK-33005.7)
- `Tests/UI/test_console_native_chat_flow.py::test_console_native_tab_switch_restores_transcript_messages` (TASK-33005.7)
- `Tests/UI/test_console_native_chat_flow.py::test_console_workspace_conversation_switch_restores_transcript_messages` (TASK-33005.7)
- `Tests/UI/test_console_session_settings.py::test_console_settings_delayed_initial_focus_is_safe_after_unmount` (TASK-33006.10)
- `Tests/UI/test_console_session_settings.py::test_console_settings_modal_enumerated_inputs_list_accepted_values` (TASK-33002.7)
- `Tests/UI/test_console_setup_card_switcher.py::test_detected_stub_server_second_model_is_saved_by_the_real_writer` (TASK-35190)
- `Tests/UI/test_first_run_wizard_live_contract.py::test_full_track_skip_everything_leaves_app_usable` (TASK-33005.9)
- `Tests/UI/test_non_obscuring_focus_contract.py::test_bundled_native_choice_and_tree_states_match_source_contracts` (TASK-33003.17)
- `Tests/UI/test_non_obscuring_focus_contract.py::test_native_choice_and_tree_states_follow_shared_contracts` (TASK-33003.17)
- `Tests/UI/test_settings_configuration_hub.py::test_settings_provider_switch_resets_staged_model_for_each_provider_transition` (TASK-33005.7)
- `Tests/UI/test_settings_configuration_hub.py::test_settings_provider_switch_updates_inspector_readiness` (TASK-33005.7)
- `Tests/UI/test_settings_custom_endpoint_default.py::test_home_console_readiness_seam_handles_a_registry_default` (TASK-33005.9)
- `Tests/UI/test_settings_custom_endpoint_default.py::test_home_readiness_computes_a_malformed_default` (TASK-33005.9)
- `Tests/UI/test_settings_provider_key_check.py::test_settings_t_lists_kimi_and_zai_at_the_alias_a_send_pins` (TASK-33005.11)
- `Tests/UI/test_theme_contrast.py::test_user_theme_with_any_kept_variable_form_still_loads` (TASK-33003.16)
- `Tests/UI/test_theme_contrast.py::test_user_theme_with_unmeasurable_surfaces_keeps_visible_guard_colours` (TASK-33003.16)

Two constraints shape the placement. The admission-sensitive job (`pr-fast-lane-admission`) has little room left: Phase 9's own targets made it about 29 minutes serial locally, estimated at 23-30 CI minutes under its 35-minute cap. And a test's solo time is not its cost in a step (the lessons entry on it), so each placement is timed in its step's order. Where the bootstrap_profile ones go depends on the rule TASK-35570 decides.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed test runs in exactly one PR lane, chosen by its marker under the rule TASK-35570 decides, and Tests/CI's lane contracts list each placement
- [ ] #2 Each placement is timed in its step's order, and every PR job it joins still ends at least 5 minutes under its cap in its slowest measured run, or the owner has ruled on a timeout raise or a new job
- [ ] #3 A lane probe over Phase 9's range finds no modified model-config test function that runs in no PR lane
<!-- AC:END -->
