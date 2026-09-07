# Speech & TTS Settings Ownership Release Evidence

Date: 2026-08-01
Task: TASK-1988
Decision: [ADR-039](../../../../backlog/decisions/039-global-and-studio-tts-settings-ownership.md)

## Evidence boundary

- Headless complete-WAV and playback-handoff proof: Passing
- Human audible playback proof: Passed in TASK-1989 with explicit user
  confirmation; deterministic WAV and playback-handoff evidence remains
  separately identified.
- No provider process, provider network, model download, or audio hardware is
  used by the automated gate.
- Automated evidence validates complete responses and playback handoff. It
  does not claim audible output or incremental streaming.
- Live acceptance uses the user-supplied external server and model only in
  TASK-1989. It must not download, launch, supervise, or stop audio.cpp.

## Requirement evidence

Each stable requirement ID from the approved PRD appears exactly once. The
companion test mechanically checks this table and verifies every referenced
automated test node still exists.

| Requirement | Evidence kind | Test or UAT journey | Result |
| --- | --- | --- | --- |
| OWN-001 | Automated | `Tests/UI/test_speech_settings_contracts.py::test_every_adr_039_scope_partition_is_exact` | Passing |
| OWN-002 | Automated | `Tests/TTS/test_speech_tts_settings_ownership_hardening.py::test_studio_save_reset_and_preview_preserve_other_owners` | Passing |
| OWN-003 | Automated | `Tests/TTS/test_speech_tts_settings_ownership_hardening.py::test_studio_save_reset_and_preview_preserve_other_owners` | Passing |
| OWN-004 | Automated | `Tests/UI/test_speech_settings_contracts.py::test_every_control_retains_its_exact_provider_owner` | Passing |
| OWN-005 | Automated | `Tests/UI/test_speech_settings_contracts.py::test_every_current_control_is_classified_exactly_once` | Passing |
| OWN-006 | Automated | `Tests/UI/test_speech_settings_contracts.py::test_default_provider_and_configure_provider_have_distinct_ids` | Passing |
| IA-001 | Automated | `Tests/UI/test_speech_tts_settings_ownership_closeout.py::test_first_time_audio_cpp_setup_lab_generation_and_console_handoff` | Passing |
| IA-002 | Automated | `Tests/UI/test_speech_tts_settings_ownership_closeout.py::test_first_time_audio_cpp_setup_lab_generation_and_console_handoff` | Passing |
| IA-003 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_global_panel_states_scope_and_mounts_only_selected_provider` | Passing |
| IA-004 | Automated | `Tests/UI/test_studio_tts_preferences.py::test_studio_surface_states_scope_and_excludes_every_global_owner` | Passing |
| IA-005 | Automated | `Tests/UI/test_speech_runtime_status.py::test_bounded_navigation_round_trips_every_allowed_intent`; `Tests/UI/test_speech_tts_settings_ownership_closeout.py::test_first_time_audio_cpp_setup_lab_generation_and_console_handoff` | Passing |
| CFG-001 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_global_panel_states_scope_and_mounts_only_selected_provider` | Passing |
| CFG-002 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_audio_cpp_cached_choices_are_revisioned_and_model_scoped` | Passing |
| CFG-003 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_each_provider_form_mounts_its_complete_bounded_inventory` | Passing |
| CFG-004 | Automated + UAT | `Tests/UI/test_settings_speech_tts_panel.py::test_panel_exposes_mode_specific_managed_audio_cpp_setup_controls`; UAT-01 (original External scope) | Automated passing; live UAT passed for the original External scope; the later Managed amendment is tracked separately by TASK-3795 |
| CFG-005 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_normal_panel_actions_do_not_contact_or_initialize_tts` | Passing |
| CFG-006 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_normal_panel_actions_do_not_contact_or_initialize_tts` | Passing |
| CFG-007 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_saved_but_unavailable_and_transient_reconfiguration_are_not_ready` | Passing |
| CFG-008 | Automated + UAT | `Tests/UI/test_settings_speech_tts_panel.py::test_credential_operations_are_separate_from_ordinary_save`; UAT-08 | Automated passing; live UAT passed after P1 fix |
| CFG-009 | Automated | `Tests/TTS/test_stts_settings_reconfiguration.py::test_changed_audio_cpp_config_retires_only_audio_cpp` | Passing |
| CFG-010 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_panel_tracks_dirty_state_and_revert_restores_the_saved_snapshot` | Passing |
| CFG-011 | Automated | `Tests/UI/test_settings_audio_cpp_experience_model.py::test_fresh_authoritative_catalog_keeps_missing_exact_values_visible` | Passing |
| CFG-012 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_dirty_global_link_cancel_preserves_draft_focus_and_navigation` | Passing |
| CAT-001 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_normal_panel_actions_do_not_contact_or_initialize_tts` | Passing |
| CAT-002 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_audio_cpp_first_run_offers_only_dynamic_default_policies` | Passing |
| CAT-003 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_audio_cpp_authoritative_missing_model_stays_visible` | Passing |
| CAT-004 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_audio_cpp_cached_choices_are_revisioned_and_model_scoped` | Passing |
| CAT-005 | Automated | `Tests/UI/test_speech_runtime_status.py::test_status_store_rejects_older_revisions_and_times` | Passing |
| CAT-006 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_audio_cpp_remote_http_warning_and_dirty_draft_attribution_update` | Passing |
| CFG-020 | Automated | `Tests/TTS/test_speech_tts_settings_ownership_hardening.py::test_studio_save_reset_and_preview_preserve_other_owners` | Passing |
| CFG-021 | Automated + UAT | `Tests/TTS/test_speech_tts_settings_ownership_hardening.py::test_studio_save_reset_and_preview_preserve_other_owners`; UAT-05 | Automated passing; live UAT passed after P1 fix |
| CFG-022 | Automated | `Tests/TTS/test_studio_preferences.py::test_provider_options_survive_switching_without_cross_provider_leakage` | Passing |
| CFG-023 | Automated | `Tests/UI/test_studio_tts_preferences.py::test_revert_and_reset_restore_only_the_studio_scope` | Passing |
| CFG-024 | Automated | `Tests/UI/test_studio_tts_preferences.py::test_unsaved_request_tuning_reaches_generation_without_persistence` | Passing |
| CFG-025 | Automated + UAT | `Tests/UI/test_studio_tts_preferences.py::test_character_profile_is_a_preview_until_explicit_adoption`; UAT-07 | Automated passing; live UAT passed |
| CFG-026 | Automated | `Tests/UI/test_studio_tts_preferences.py::test_studio_surface_states_scope_and_excludes_every_global_owner` | Passing |
| STATE-001 | Automated + UAT | `Tests/TTS/test_speech_tts_settings_ownership_hardening.py::test_studio_save_reset_and_preview_preserve_other_owners`; UAT-06 | Automated passing; live UAT passed after two P1 fixes |
| STATE-002 | Automated | `Tests/TTS/test_effective_settings.py::test_studio_resolution_uses_draft_saved_global_then_fallback` | Passing |
| STATE-003 | Automated | `Tests/TTS/test_effective_settings.py::test_effective_snapshot_is_immutable_and_contains_no_sensitive_payload` | Passing |
| STATE-004 | Automated | `Tests/TTS/test_effective_settings.py::test_first_available_reads_one_catalog_and_freezes_its_revision` | Passing |
| STATE-005 | Automated | `Tests/TTS/test_effective_settings.py::test_missing_exact_model_blocks_instead_of_using_dynamic_fallback` | Passing |
| STATE-010 | Automated | `Tests/UI/test_speech_runtime_status.py::test_every_configuration_state_uses_the_canonical_vocabulary` | Passing |
| STATE-011 | Automated | `Tests/UI/test_speech_runtime_status.py::test_every_runtime_state_uses_the_canonical_vocabulary` | Passing |
| STATE-012 | Automated + UAT | `Tests/UI/test_speech_runtime_status.py::test_external_audio_cpp_readiness_is_independent_of_every_local_dependency`; UAT-10 | Automated passing; live UAT passed |
| STATE-013 | Automated | `Tests/UI/test_speech_settings_contracts.py::test_safe_status_is_revisioned_frozen_and_has_no_free_form_payload` | Passing |
| STATE-014 | Automated | `Tests/UI/test_speech_profile_navigation.py::test_exact_preset_preserves_existing_playground_audio`; `Tests/TTS/test_stts_audio_cpp_generation.py::test_retiring_only_generation_preserves_completed_artifact` | Passing |
| MIG-001 | Automated | `Tests/TTS/test_studio_preferences.py::test_sparse_round_trip_and_provider_isolation` | Passing |
| MIG-002 | Automated | `Tests/TTS/test_speech_tts_settings_ownership_hardening.py::test_migration_and_disabled_studio_reader_are_additive_and_idempotent` | Passing |
| MIG-003 | Automated | `Tests/TTS/test_speech_tts_settings_ownership_hardening.py::test_migration_and_disabled_studio_reader_are_additive_and_idempotent` | Passing |
| MIG-004 | Automated | `Tests/TTS/test_studio_preferences.py::test_unrecoverable_studio_record_can_reset_without_touching_other_scopes` | Passing |
| MIG-005 | Automated | `Tests/TTS/test_studio_preferences.py::test_concurrent_writers_publish_one_snapshot_and_conflict_the_other` | Passing |
| MIG-006 | Automated | `Tests/TTS/test_speech_tts_settings_ownership_hardening.py::test_migration_and_disabled_studio_reader_are_additive_and_idempotent` | Passing |
| SEC-001 | Automated | `Tests/TTS/test_speech_tts_settings_ownership_hardening.py::test_privacy_sentinels_do_not_cross_owned_output_boundaries` | Passing |
| SEC-002 | Automated + UAT | `Tests/UI/test_settings_speech_tts_panel.py::test_environment_credential_is_read_only_and_editor_starts_empty`; UAT-08 | Automated passing; live UAT passed after P1 fix |
| SEC-003 | Automated | `Tests/TTS/test_speech_tts_settings_ownership_hardening.py::test_privacy_sentinels_do_not_cross_owned_output_boundaries`; `Tests/TTS/test_tts_logging_privacy.py::test_audio_cpp_service_boundary_never_exposes_private_http_or_request_values`; `Tests/TTS/test_tts_logging_privacy.py::test_console_tts_metrics_use_only_the_safe_slice_one_allowlist`; `Tests/UI/test_speech_profile_navigation.py::test_speech_screen_state_keeps_only_bounded_playground_axes` | Passing |
| SEC-004 | Automated | `Tests/TTS/test_speech_tts_settings_ownership_hardening.py::test_privacy_sentinels_do_not_cross_owned_output_boundaries` | Passing |
| SEC-005 | Manual UAT | UAT-01 | Passed with explicit human audible confirmation and privacy-reviewed synthetic evidence |
| STATE-020 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_invalid_save_is_field_specific_and_posts_no_event` | Passing |
| STATE-021 | Automated + UAT | `Tests/UI/test_settings_speech_tts_panel.py::test_saved_but_unavailable_and_transient_reconfiguration_are_not_ready`; UAT-02 | Automated passing; live UAT passed after P1 fixes |
| STATE-022 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_cache_reload_failure_keeps_persistence_and_runtime_results_distinct` | Passing |
| STATE-023 | Automated | `Tests/UI/test_settings_speech_tts_panel.py::test_audio_cpp_authoritative_missing_model_stays_visible` | Passing |
| STATE-024 | Automated | `Tests/UI/test_studio_tts_preferences.py::test_corrupt_record_offers_a_studio_only_reset` | Passing |
| A11Y-001 | Automated + UAT | `Tests/UI/test_speech_tts_settings_ownership_closeout.py::test_global_and_studio_controls_have_programmatic_labels_and_text_states`; UAT-01 | Automated passing; live UAT passed after P0 result-action fix |
| A11Y-002 | Automated + UAT | `Tests/UI/test_speech_tts_settings_ownership_closeout.py::test_keyboard_order_reaches_primary_actions_and_status_does_not_steal_focus`; `Tests/UI/test_speech_tts_settings_ownership_closeout.py::test_keyboard_invalid_save_and_dirty_leave_cancel_restore_focus`; UAT-01 | Automated passing; live UAT passed after P0 result-action fix |
| A11Y-003 | Automated | `Tests/UI/test_speech_axis_row.py::test_the_override_is_not_signalled_by_colour_alone` | Passing |
| A11Y-004 | Automated | `Tests/UI/test_speech_tts_settings_ownership_closeout.py::test_global_and_studio_controls_have_programmatic_labels_and_text_states` | Passing |
| A11Y-005 | Automated + UAT | `Tests/UI/test_speech_tts_settings_ownership_closeout.py::test_supported_narrow_layout_uses_vertical_scroll_without_clipping_actions`; UAT-01 | Automated passing; live UAT passed after P0 result-action fix |
| A11Y-006 | Automated + UAT | `Tests/UI/test_speech_tts_settings_ownership_closeout.py::test_keyboard_order_reaches_primary_actions_and_status_does_not_steal_focus`; `Tests/UI/test_speech_tts_settings_ownership_closeout.py::test_keyboard_invalid_save_and_dirty_leave_cancel_restore_focus`; UAT-02 | Automated passing; live UAT passed |

## Manual journey ownership

TASK-1989 owns live UAT-01 through UAT-10 from the approved PRD. TASK-1988
provides deterministic fake coverage and the exact live script boundary; it
does not convert headless playback-control handoff into an audible claim.
TASK-1989 has passed UAT-01 through UAT-10. The final UAT-03 retest used the
user-supplied listener's two advertised TTS models, `pocket-tts-en` and
`supertonic-3`, and generated then entered playback for one distinct WAV from
each model in the same Chatbook session without changing or restarting the
external server. The record does not mislabel these playback-state observations
as additional human audible confirmation.

## End-to-end fake gate

`Tests/UI/test_speech_tts_settings_ownership_closeout.py::test_first_time_audio_cpp_setup_lab_generation_and_console_handoff`
drives Settings search, local persistence, exact scoped navigation, distinct
Test and Refresh actions, Studio generation, a structurally valid complete WAV,
and the existing Console playback event. Its injected Settings service fails
any provider operation, while its Lab services are in-process fakes.

## Cross-slice race gate

The closeout race gate intentionally reuses the focused deterministic barrier
tests rather than introducing a second concurrency model:

- global persistence and targeted reconfiguration serialization:
  `Tests/TTS/test_stts_settings_reconfiguration.py::test_concurrent_settings_saves_are_serialized`;
- Studio compare-before-publish:
  `Tests/TTS/test_studio_preferences.py::test_concurrent_writers_publish_one_snapshot_and_conflict_the_other`;
- stale configuration, catalog, and voice publication:
  `Tests/UI/test_stts_playground_audio_cpp.py::test_catalog_result_is_discarded_when_configuration_revision_changes`,
  `Tests/UI/test_stts_playground_audio_cpp.py::test_superseded_catalog_success_cannot_invalidate_newer_success`, and
  `Tests/UI/test_stts_playground_audio_cpp.py::test_superseded_same_model_voice_result_cannot_overwrite_newer_success`;
- navigation and late generation fencing:
  `Tests/UI/test_speech_profile_navigation.py::test_exact_preset_rejects_late_prior_generation_completion` and
  `Tests/TTS/test_stts_audio_cpp_generation.py::test_retiring_playground_context_fences_in_flight_completion`;
- completed-artifact independence:
  `Tests/UI/test_speech_profile_navigation.py::test_exact_preset_preserves_existing_playground_audio` and
  `Tests/TTS/test_stts_audio_cpp_generation.py::test_retiring_only_generation_preserves_completed_artifact`;
- active-playback result replacement and transition ownership:
  `Tests/UI/test_speech_playground_pane.py::test_new_result_stops_active_playback_before_replacing_controls`,
  `Tests/UI/test_speech_playground_pane.py::test_auto_play_new_result_cancels_prior_start_worker_before_takeover`,
  `Tests/UI/test_speech_playground_pane.py::test_profile_navigation_fences_result_waiting_for_playback_stop`, and
  `Tests/UI/test_speech_playground_pane.py::test_play_is_blocked_while_new_result_waits_for_playback_stop`.

## Compatibility gate

`Tests/TTS/test_speech_tts_settings_ownership_hardening.py::test_every_legacy_provider_retains_its_accepted_request_shape`
pins OpenAI, ElevenLabs, Kokoro, Chatterbox, Higgs, and AllTalk behind the
temporary legacy bridge. The same file proves additive rollback behavior,
cross-owner isolation, privacy boundaries, and the non-authoritative semantics
of approximate catalogs.
