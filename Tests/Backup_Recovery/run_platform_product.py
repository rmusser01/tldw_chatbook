"""Run the finite installed backup product qualification on a native runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import struct
import subprocess  # nosec B404 - fixed local commands and arguments only
import sys
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from defusedxml import ElementTree as ET

_NATIVE_TESTS = (
    "Tests/Utils/test_windows_files.py",
    "Tests/Utils/test_windows_security_decode.py",
)
_PRODUCT_TESTS = (
    "Tests/DB/test_private_sqlite_windows_descriptor.py",
    (
        "Tests/ProductionApp/test_backup_restore_end_to_end.py::"
        "test_f9_created_archive_restores_and_opens_through_actual_controls"
    ),
    (
        "Tests/Backup_Recovery/test_complete_roundtrip.py::"
        "test_two_captured_profiles_restore_and_open_with_native_content"
    ),
    (
        "Tests/Backup_Recovery/test_f9_replacement_workflow.py::"
        "test_full_f9_replacement_after_explicit_safety_and_credential_review"
    ),
    (
        "Tests/Backup_Recovery/test_later_rollback_credential_ui.py::"
        "test_f9_later_rollback_requires_explicit_credential_review"
    ),
    (
        "Tests/Backup_Recovery/test_projection_dependency_lock.py::"
        "test_dependency_union_retains_same_private_regular_lock"
    ),
    (
        "Tests/Backup_Recovery/test_projection_dependency_lock.py::"
        "test_independent_process_refuses_contended_lock_before_publication"
    ),
    (
        "Tests/Backup_Recovery/test_projection_dependency_lock.py::"
        "test_unsafe_existing_lock_is_refused_without_publication"
    ),
    (
        "Tests/Backup_Recovery/test_projection_dependency_lock.py::"
        "test_failed_publication_releases_lock_without_removing_stable_name"
    ),
    (
        "Tests/ProductionApp/test_backup_restore_composition.py::"
        "test_actual_mounted_backup_publishes_verified_archive_after_navigation"
    ),
    "Tests/Backup_Recovery/test_mcp_recovery_native_files.py",
    "Tests/Backup_Recovery/test_skills_recovery_native_files.py",
    "Tests/Backup_Recovery/test_provider_recovery_native_files.py",
    "Tests/Backup_Recovery/test_admission_closed_gate.py",
    "Tests/Backup_Recovery/test_runtime_native_poll.py",
    "Tests/Backup_Recovery/test_runtime_cache_revisit.py",
    "Tests/Backup_Recovery/test_initial_screen_observation.py",
    "Tests/Backup_Recovery/test_caller_cache_retirement.py",
    "Tests/Backup_Recovery/test_console_config_sync_lifetime.py",
    "Tests/Backup_Recovery/test_console_guidance_readiness.py",
    "Tests/Backup_Recovery/test_console_tray_layout.py",
    "Tests/Backup_Recovery/test_console_bounded_layout.py",
    "Tests/Backup_Recovery/test_sqlite_inside_config_scope.py",
    "Tests/Backup_Recovery/test_persona_observation_budget.py",
    "Tests/Backup_Recovery/test_db_status_maintenance.py",
    "Tests/Backup_Recovery/test_canvas_policy_maintenance.py",
    "Tests/Backup_Recovery/test_canvas_policy_worker.py",
    "Tests/Backup_Recovery/test_canvas_view_binding_lifetime.py",
    "Tests/Backup_Recovery/test_context_policy_config_lifetime.py",
    "Tests/Backup_Recovery/test_console_projection_config_lifetime.py",
    "Tests/Backup_Recovery/test_console_canvas_config_order.py",
    "Tests/Backup_Recovery/test_config_native_lock_order.py",
    "Tests/Backup_Recovery/test_core_dependency_discovery.py",
    "Tests/Backup_Recovery/test_prompt_count_admission.py",
    "Tests/Backup_Recovery/test_capture_root_order.py",
    "Tests/Backup_Recovery/test_admission_diagnostics.py",
    "Tests/Backup_Recovery/test_loop_diagnostics.py",
    "Tests/Backup_Recovery/test_startup_readmission_continuity.py",
    "Tests/Backup_Recovery/test_startup_pending_readmission.py::test_initial_pending_fence_waits_for_live_scheduler_and_native_pause",
    "Tests/Backup_Recovery/test_recovery_restart_windows.py",
    "Tests/Backup_Recovery/test_windows_acl_fixture.py",
    "Tests/Backup_Recovery/test_merged_service_producer_maintenance.py",
    "Tests/Backup_Recovery/test_file_notes_maintenance.py::test_repository_probe_preserves_maintenance_and_error_lifetimes",
    "Tests/Backup_Recovery/test_notes_recovery_review.py::test_hidden_replica_tombstone_requires_fresh_pairing_before_sweep",
    "Tests/Backup_Recovery/test_thread_diagnostics.py",
    "Tests/Backup_Recovery/test_later_failure_diagnostics.py",
    "Tests/Canvas/test_profiles.py::test_native_crlf_checkout_preserves_verified_canvas_profiles",
    "Tests/Backup_Recovery/test_later_rollback_handoff.py",
    "Tests/Backup_Recovery/test_recovery_restart.py::test_actual_handoff_execs_fresh_recovery_ui",
    "Tests/Backup_Recovery/test_recovery_restart.py::test_actual_restart_keeps_launching_package_ahead_of_shadow_cwd",
    "Tests/Backup_Recovery/test_database_default_spelling.py",
    "Tests/Backup_Recovery/test_restart_observation.py",
    "Tests/Backup_Recovery/test_console_resume_character_lifetime.py",
    "Tests/TTS/test_profile_migration_native_maintenance.py::test_native_exact_cleanup_preserves_failure_and_exclusion",
    "Tests/TTS/test_profile_native_close_contract.py::test_native_repository_close_preserves_proven_state_and_exclusion",
    "Tests/Backup_Recovery/test_tts_profile_lock_inventory.py",
    "Tests/Backup_Recovery/test_file_inventory.py::test_native_leaf_refusal_keeps_explicit_classification",
    "Tests/Backup_Recovery/test_tts_retained_references_roundtrip.py::test_canonical_tts_reference_blob_restores_with_fresh_native_getters",
    "Tests/Backup_Recovery/test_runtime_owner_capture.py::test_default_run_log_container_has_declared_topology",
    "Tests/Backup_Recovery/test_skills_chatbooks_capture.py::test_exact_script_output_topology_and_unsupported_config",
    "Tests/Backup_Recovery/test_related_path_admission.py",
    "Tests/Backup_Recovery/test_scheduler_native_pause_intent.py",
    "Tests/Backup_Recovery/test_mounted_console_backup.py",
    "Tests/Backup_Recovery/test_bootstrap_registry_reader.py",
    "Tests/Backup_Recovery/test_finite_library_workers.py",
    "Tests/Backup_Recovery/test_home_open_task_retirement.py",
    "Tests/Backup_Recovery/test_home_notification_retirement.py",
    "Tests/Backup_Recovery/test_first_note_backup.py",
    "Tests/Backup_Recovery/test_first_run_restore.py",
    "Tests/Backup_Recovery/test_profile_open.py",
    "Tests/Backup_Recovery/test_default_service_container.py",
    "Tests/Backup_Recovery/test_first_binding_unused_scaffolds.py",
    "Tests/Backup_Recovery/test_first_user_data_binding.py",
    "Tests/Backup_Recovery/test_bound_config_companions.py",
    "Tests/Backup_Recovery/test_config_binding_writes.py",
    "Tests/Backup_Recovery/test_bound_config_siblings.py",
    "Tests/Backup_Recovery/test_config_sibling_capture.py",
    "Tests/Backup_Recovery/test_missing_config_publication.py",
    "Tests/Backup_Recovery/test_default_config_file_lifecycle.py",
    "Tests/Backup_Recovery/test_config_file_retirement.py",
    "Tests/Backup_Recovery/test_sidebar_source_lifetime.py",
    "Tests/Backup_Recovery/test_agent_runs_recovery_schema.py",
    "Tests/Backup_Recovery/test_large_recovery_records.py::test_large_collection_completes_actual_isolated_publication",
    "Tests/Backup_Recovery/test_restore_destinations.py",
    "Tests/UI/test_backup_restore_destinations.py",
)
_RESTORE_DIAGNOSTIC_TESTS = (
    (
        "Tests/ProductionApp/test_backup_restore_end_to_end.py::"
        "test_f9_created_archive_restores_and_opens_through_actual_controls[plain]"
    ),
)
_SUPPORT_DIAGNOSTIC_TESTS = (
    "Tests/Backup_Recovery/test_restart_observation.py",
    "Tests/Backup_Recovery/test_console_progress_timer.py",
    "Tests/Backup_Recovery/test_context_policy_config_lifetime.py",
    "Tests/Backup_Recovery/test_console_config_sync_lifetime.py",
    "Tests/Backup_Recovery/test_sqlite_inside_config_scope.py",
    "Tests/Backup_Recovery/test_persona_observation_budget.py",
    "Tests/Backup_Recovery/test_db_status_maintenance.py",
    "Tests/Backup_Recovery/test_canvas_policy_maintenance.py",
    "Tests/Backup_Recovery/test_canvas_policy_worker.py",
    "Tests/Backup_Recovery/test_canvas_view_binding_lifetime.py",
    "Tests/Backup_Recovery/test_console_projection_config_lifetime.py",
    "Tests/Backup_Recovery/test_console_canvas_config_order.py",
    "Tests/Backup_Recovery/test_config_native_lock_order.py",
    "Tests/Backup_Recovery/test_core_dependency_discovery.py",
    "Tests/Backup_Recovery/test_recovery_restart_windows.py",
    "Tests/Backup_Recovery/test_windows_acl_fixture.py",
    "Tests/Backup_Recovery/test_merged_service_producer_maintenance.py",
    "Tests/Backup_Recovery/test_file_notes_maintenance.py::test_repository_probe_preserves_maintenance_and_error_lifetimes",
    "Tests/Backup_Recovery/test_notes_recovery_review.py::test_hidden_replica_tombstone_requires_fresh_pairing_before_sweep",
    "Tests/Backup_Recovery/test_thread_diagnostics.py",
    "Tests/Backup_Recovery/test_later_failure_diagnostics.py",
    "Tests/Canvas/test_profiles.py::test_native_crlf_checkout_preserves_verified_canvas_profiles",
    "Tests/Backup_Recovery/test_later_rollback_handoff.py",
    "Tests/Backup_Recovery/test_recovery_restart.py::test_actual_handoff_execs_fresh_recovery_ui",
    "Tests/Backup_Recovery/test_recovery_restart.py::test_actual_restart_keeps_launching_package_ahead_of_shadow_cwd",
    "Tests/Backup_Recovery/test_mounted_console_backup.py",
    "Tests/Backup_Recovery/test_profile_open.py::test_profile_open_requires_actual_mounted_local_reads[console_quit]",
    "Tests/Backup_Recovery/test_profile_open.py::test_profile_open_requires_actual_mounted_local_reads[console_edit]",
    "Tests/Backup_Recovery/test_large_recovery_records.py::test_large_collection_completes_actual_isolated_publication",
    "Tests/Backup_Recovery/test_bound_config_siblings.py::test_sibling_guard_rechecks_exact_config_anchor_and_foreign_owner[unsafe_parent-emoji]",
    "Tests/Backup_Recovery/test_bound_config_siblings.py::test_sibling_guard_rechecks_exact_config_anchor_and_foreign_owner[unsafe_parent-runtime]",
    "Tests/Backup_Recovery/test_bound_config_siblings.py::test_sibling_guard_rechecks_exact_config_anchor_and_foreign_owner[unsafe_parent-sidebar]",
)
_SELECTABLE_GROUP_TESTS = (
    _PRODUCT_TESTS[1] + "[selected_plain]",
    _PRODUCT_TESTS[1] + "[selected_encrypted]",
    _PRODUCT_TESTS[1] + "[plain]",
    _PRODUCT_TESTS[1] + "[encrypted_credentials]",
    "Tests/Backup_Recovery/test_selective_restore_service.py",
    "Tests/Backup_Recovery/test_settings_finish.py",
    "Tests/Backup_Recovery/test_settings_preservation_evidence.py",
    "Tests/Backup_Recovery/test_selected_owner_absence.py",
    "Tests/Backup_Recovery/test_selected_owner_finish.py",
    "Tests/Backup_Recovery/test_same_session_retirement_capture.py",
    "Tests/Backup_Recovery/test_effective_admission_roots.py",
    "Tests/Backup_Recovery/test_admission.py::test_control_root_overlap_and_nested_admission_are_refused",
    "Tests/Backup_Recovery/test_admission.py::test_missing_stable_lock_is_not_recreated",
    "Tests/Backup_Recovery/test_bootstrap.py::test_disjoint_requires_an_intact_local_binding",
    "Tests/Backup_Recovery/test_bootstrap.py::test_binding_requires_config_in_declared_admission_scope",
    "Tests/Backup_Recovery/test_bootstrap.py::test_registry_intent_never_becomes_plain_startup_permission",
    "Tests/Backup_Recovery/test_replacement_admission_recovery.py",
    "Tests/Backup_Recovery/test_eval_selective_retention.py",
    "Tests/Backup_Recovery/test_absent_sqlite_dependencies.py",
    "Tests/Backup_Recovery/test_native_discovery_sqlite_copies.py",
    "Tests/Backup_Recovery/test_restore_data_groups.py",
    "Tests/Backup_Recovery/test_rollback_group_scope.py",
    "Tests/Backup_Recovery/test_group_cli.py",
    "Tests/Backup_Recovery/test_group_issues.py",
    "Tests/Backup_Recovery/test_data_groups.py",
    "Tests/Backup_Recovery/test_archive_group_scope.py",
    "Tests/Backup_Recovery/test_group_capture.py",
    "Tests/Backup_Recovery/test_selective_discovery_scope.py",
    "Tests/Backup_Recovery/test_retirement_only_rollback.py",
    "Tests/Backup_Recovery/test_retirement_finalization.py",
    "Tests/Backup_Recovery/test_retained_config.py",
    "Tests/Backup_Recovery/test_first_binding_absent_sqlite.py",
    "Tests/Backup_Recovery/test_preserved_group_paths.py",
    "Tests/Backup_Recovery/test_preserved_absent_scope.py",
    "Tests/Backup_Recovery/test_settings_group_integration.py",
    "Tests/Backup_Recovery/test_settings_rag_group_destination.py",
    "Tests/Backup_Recovery/test_empty_group_restore.py",
    "Tests/Backup_Recovery/test_workflow_census.py",
    "Tests/Backup_Recovery/test_merged_service_producer_maintenance.py",
    "Tests/Notes/test_notes_sync_note_location.py",
    "Tests/UI/test_backup_data_groups.py",
)
_PRODUCT_SELECTIONS = {
    # Resolved per native OS below; ordinary source tests and installed F9
    # remain explicitly distinct in the existing runner receipts.
    "admission-amortization": (),
    "native-credentials-source": (
        "Tests/ProductionApp/test_native_credential_recovery.py::test_native_credential_source",
    ),
    "native-credentials-destination": (
        "Tests/ProductionApp/test_native_credential_recovery.py::test_native_credential_destinations",
    ),
    "full": (
        *_PRODUCT_TESTS,
        "Tests/Backup_Recovery/test_default_service_replacement.py",
        "Tests/Backup_Recovery/test_created_persona_subtree_rollback.py",
        "Tests/Backup_Recovery/test_eval_rollback_retention.py",
        *_SELECTABLE_GROUP_TESTS[4:],
    ),
    "restore-diagnostic": (
        *_RESTORE_DIAGNOSTIC_TESTS,
        "Tests/Backup_Recovery/test_console_guidance_readiness.py",
        "Tests/Backup_Recovery/test_console_tray_layout.py",
        "Tests/Backup_Recovery/test_console_bounded_layout.py",
    ),
    "plain": _RESTORE_DIAGNOSTIC_TESTS,
    "qodo-review": (
        _PRODUCT_TESTS[1] + "[plain]",
        "Tests/RuntimePolicy/test_server_credentials.py",
        "Tests/RuntimePolicy/test_server_credentials_lane_a.py",
        "Tests/Backup_Recovery/test_native_credential_runner.py::test_native_run_refuses_missing_transfer_root_before_effects",
        "Tests/Backup_Recovery/test_native_credential_runner.py::test_native_child_deadline_preserves_failure_and_discards_output",
        "Tests/Backup_Recovery/test_native_credential_runner.py::test_native_child_canary_streams_are_never_persisted_or_exported",
        "Tests/Backup_Recovery/test_native_credential_runner.py::test_native_child_environment_preserves_backend_and_bus",
        "Tests/Backup_Recovery/test_native_credential_runner.py::test_native_mac_child_selects_private_keychain_with_exact_home",
        "Tests/Backup_Recovery/test_native_credential_runner.py::test_native_pytest_child_keeps_environment_and_publishes_no_raw_logs",
        "Tests/Backup_Recovery/test_native_credential_runner.py::test_native_child_thread_samples_publish_only_safe_late_frames",
        "Tests/Backup_Recovery/test_thread_diagnostics.py",
        "Tests/Backup_Recovery/test_activation_mcp_remote.py",
        "Tests/Backup_Recovery/test_restore_plan.py::test_only_windows_reviewed_instance_lock_is_observed_without_body_read",
        "Tests/Backup_Recovery/test_restore_plan.py::test_malformed_instance_lock_declarations_keep_the_body_hash",
        "Tests/Backup_Recovery/test_restore_plan.py::test_windows_ordinary_and_sqlite_files_keep_exact_byte_hashes",
        "Tests/Backup_Recovery/test_restore_plan.py::test_windows_instance_lock_keeps_native_drift_refusal",
        "Tests/Backup_Recovery/test_restore_plan.py::test_windows_instance_lock_checks_pinned_before_named_and_after_state",
        "Tests/Backup_Recovery/test_restore_plan.py::test_windows_preserved_lock_uses_reviewed_config_after_publication",
        "Tests/Backup_Recovery/test_restore_plan.py::test_windows_instance_observation_binds_mode_without_normalizing_it",
        "Tests/Backup_Recovery/test_credential_profile_scopes.py",
        "Tests/Backup_Recovery/test_preserved_absent_scope.py",
        "Tests/Backup_Recovery/test_publication_finalization.py::test_pending_checks_do_not_repeat_held_namespace_overlap",
        "Tests/Backup_Recovery/test_builtin_later_snapshot.py::test_shared_legacy_later_preview_retains_authenticated_member_dependencies",
        "Tests/Backup_Recovery/test_builtin_later_snapshot.py::test_shared_builtin_later_preview_and_execution_preserve_both_profile_trees",
        "Tests/Backup_Recovery/test_builtin_later_snapshot.py::test_legacy_snapshot_alias_root_accepts_only_authenticated_own_member_removal",
        "Tests/Backup_Recovery/test_builtin_later_snapshot.py::test_legacy_snapshot_alias_root_refuses_other_dependency_changes",
        "Tests/Backup_Recovery/test_credentials.py::test_excluded_url_credentials_are_removed_from_staged_config",
        "Tests/Backup_Recovery/test_credentials.py::test_config_secret_is_removed_without_mutating_source",
        "Tests/Backup_Recovery/test_credentials.py::test_staged_current_and_history_leave_source_untouched",
        "Tests/Backup_Recovery/test_archive_writer.py::test_plaintext_roundtrip_uses_actual_reader",
        "Tests/Backup_Recovery/test_archive_writer.py::test_encrypted_roundtrip_uses_real_helper",
        "Tests/Backup_Recovery/test_archive_writer.py::test_racing_destination_is_never_overwritten",
        "Tests/Backup_Recovery/test_archive_writer.py::test_publication_refuses_replaced_parent_after_final_review",
        "Tests/Backup_Recovery/test_archive_writer.py::test_publication_interruption_preserves_real_outcome",
        "Tests/Backup_Recovery/test_archive_writer.py::test_runtime_enospc_leaves_no_published_or_temporary_artifact",
        "Tests/Backup_Recovery/test_admission.py::test_register_preintent_limit_failure_removes_only_new_locks_and_allows_retry",
        "Tests/Backup_Recovery/test_admission.py::test_register_partial_lock_failure_cleans_allocations_before_retry",
        "Tests/Backup_Recovery/test_admission.py::test_register_failed_attempt_never_removes_foreign_lock",
        "Tests/Backup_Recovery/test_admission.py::test_register_publication_failure_retains_locks_and_refuses_fresh_authority",
        "Tests/Backup_Recovery/test_admission.py::test_register_committed_cleanup_failure_retains_new_locks_and_mapping",
        "Tests/Backup_Recovery/test_admission_closed_gate.py",
        "Tests/Backup_Recovery/test_file_inventory.py::test_config_mapping_keys_are_installed_selectors_and_prose_is_untouched",
        "Tests/Backup_Recovery/test_file_inventory.py::test_config_voice_mapping_changes_only_installed_exact_selectors",
        "Tests/Backup_Recovery/test_file_inventory.py::test_remaining_installed_preferences_history_and_chatbooks_are_baseline",
        "Tests/Backup_Recovery/test_restore_destinations.py::test_restore_keeps_unused_optional_selectors_optional",
        "Tests/Backup_Recovery/test_restore_destinations.py::test_voice_selector_shape_rejects_nonstring_paths",
        "Tests/Backup_Recovery/test_retained_config.py::test_retained_voice_selectors_validate_local_paths_without_rewriting_config",
        "Tests/Backup_Recovery/test_retained_config.py::test_retained_config_rejects_selector_with_extra_database_component",
        "Tests/Backup_Recovery/test_preserved_group_paths.py::test_local_voice_selector_keeps_unselected_audio_reachable",
        "Tests/Backup_Recovery/test_activation_openai_reconnect.py::test_malformed_recovered_provider_config_refuses_before_effects",
        "Tests/Backup_Recovery/test_activation_openai_reconnect.py::test_reviewed_openai_uses_actual_handler_and_selected_connection",
        "Tests/Backup_Recovery/test_activation_openai_reconnect.py::test_actual_openai_stream_holds_source_until_native_cleanup",
        "Tests/Backup_Recovery/test_activation_openai_reconnect.py::test_enrolled_ordinary_openai_handler_preserves_transport_behavior",
        "Tests/Backup_Recovery/test_activation_openai_reconnect.py::test_actual_requests_preparation_never_resolves_unreviewed_netrc_auth",
        "Tests/Backup_Recovery/test_activation_openai_reconnect.py::test_actual_console_static_post_instance_call_keeps_awaited_native_scope",
        "Tests/Backup_Recovery/test_provider_recovery_native_files.py",
        "Tests/Backup_Recovery/test_recovery_service.py::test_failed_replacement_staging_cleans_inspection_without_journal",
        "Tests/Backup_Recovery/test_recovery_service.py::test_service_replaces_real_stored_data_and_retains_verified_originals",
        "Tests/Backup_Recovery/test_recovery_service.py::test_service_interrupted_native_replacement_can_recover_after_close",
    ),
    "selectable-groups": _SELECTABLE_GROUP_TESTS,
    "selectable-groups-diagnostic": (
        _PRODUCT_TESTS[1] + "[encrypted_credentials]",
        "Tests/Backup_Recovery/test_selected_owner_absence.py",
        "Tests/Backup_Recovery/test_selected_owner_finish.py",
        "Tests/Backup_Recovery/test_same_session_retirement_capture.py",
        "Tests/Backup_Recovery/test_effective_admission_roots.py",
        "Tests/Backup_Recovery/test_replacement_admission_recovery.py",
        "Tests/Backup_Recovery/test_retained_config.py",
        "Tests/Backup_Recovery/test_preserved_group_paths.py",
        "Tests/Backup_Recovery/test_preserved_absent_scope.py",
    ),
    "encrypted": (_PRODUCT_TESTS[1] + "[encrypted]",),
    "encrypted-credentials": (_PRODUCT_TESTS[1] + "[encrypted_credentials]",),
    "roundtrip": _PRODUCT_TESTS[2:3],
    "replacement": (
        *_PRODUCT_TESTS[3:4],
        "Tests/Backup_Recovery/test_default_service_replacement.py",
        "Tests/Backup_Recovery/test_created_persona_subtree_rollback.py",
        "Tests/Backup_Recovery/test_eval_rollback_retention.py",
    ),
    # Each bounded Windows job retains an installed-wheel workflow. The full
    # aggregate above remains available and is exactly the union of these jobs.
    "replacement-persona": (
        *_PRODUCT_TESTS[3:4],
        "Tests/Backup_Recovery/test_created_persona_subtree_rollback.py",
    ),
    "replacement-default-evals": (
        "Tests/Backup_Recovery/test_default_service_replacement.py",
        "Tests/Backup_Recovery/test_eval_rollback_retention.py",
    ),
    "rollback": _PRODUCT_TESTS[4:5],
    "support": (_PRODUCT_TESTS[0], *_PRODUCT_TESTS[5:]),
    "support-diagnostic": _SUPPORT_DIAGNOSTIC_TESTS,
    "native-close-diagnostic": (
        "Tests/Backup_Recovery/test_console_guidance_readiness.py",
        "Tests/Backup_Recovery/test_console_tray_layout.py",
        "Tests/Backup_Recovery/test_console_bounded_layout.py",
        "Tests/TTS/test_profile_native_close_contract.py::test_native_repository_close_preserves_proven_state_and_exclusion",
        "Tests/Backup_Recovery/test_db_status_maintenance.py",
        "Tests/Backup_Recovery/test_canvas_policy_maintenance.py",
        "Tests/Backup_Recovery/test_canvas_policy_worker.py",
        "Tests/Backup_Recovery/test_canvas_view_binding_lifetime.py",
        "Tests/Backup_Recovery/test_mounted_console_backup.py::test_mounted_console_complete_capture_and_resumed_writes[settings]",
        "Tests/Backup_Recovery/test_mounted_console_backup.py::test_mounted_console_complete_capture_and_resumed_writes[library]",
    ),
}


def admission_selection(system: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Finite accepted Task2/3 routes; Windows admits only native cold storage."""
    if system not in ("Darwin", "Linux", "Windows"):
        raise ValueError("unsupported_admission_platform")
    shared = (
        *_RESTORE_DIAGNOSTIC_TESTS,
        *(
            "Tests/Backup_Recovery/test_runtime_native_poll.py::" + name
            for name in (
                "test_monitor_remains_responsive_during_one_blocked_native_probe",
                "test_monitor_cancellation_waits_for_native_probe_release",
                "test_probe_failure_is_mapped_on_monitor_task_and_retried",
                "test_requested_pause_keeps_runtime_coordination_on_monitor_task",
                "test_the_unpatched_monitor_probes_about_once_a_second",
                "test_monitor_initial_probe_is_immediate",
                "test_monitor_local_pause_transition_wakes_without_transaction_churn",
                "test_monitor_deadline_counts_time_spent_in_previous_probe",
                "test_monitor_subscription_retains_snapshot_wait_race_and_cleans_callbacks",
                "test_lifecycle_pulse_finishes_while_coordinator_mutex_is_held",
            )
        ),
        *(
            "Tests/Backup_Recovery/test_mcp_source_lifetimes.py::" + name
            for name in (
                "test_guarded_json_reuses_parse_but_reads_current_bytes",
                "test_guarded_json_same_metadata_bytes_and_nested_returns",
                "test_warm_permission_parse_never_supplies_last_good_policy",
                "test_guarded_permission_strict_inventory_bytes_and_policy_stay_distinct",
                "test_guarded_parse_pause_invalidates_and_oversize_keeps_legacy_policy",
                "test_guarded_parse_migration_does_not_publish_prewrite_bytes",
                "test_guarded_parse_unknown_read_close_never_publishes_or_retries",
                "test_guarded_parse_last_owner_retirement_and_selection_error_discard",
                "test_guarded_json_replacement_after_read_refuses_detached_bytes",
                "test_guarded_parse_config_generation_change_requires_fresh_parse",
                "test_guarded_json_target_shape_fallback_never_publishes",
            )
        ),
        *(
            "Tests/MCP/test_local_store.py::" + name
            for name in (
                "test_legacy_schema_migrates_durably_and_reopens",
                "test_unknown_or_malformed_schema_is_not_rewritten",
                "test_migration_refuses_malformed_authoritative_sections",
                "test_malformed_owned_profile_preserves_source_bytes",
                "test_legacy_schema_cannot_import_owned_authority",
                "test_owned_profile_roundtrip_preserves_literal_argv",
                "test_malformed_owned_literal_headers_preserve_source",
            )
        ),
        "Tests/Backup_Recovery/test_scheduler_native_pause_intent.py::test_native_pause_intent_preserves_due_work_until_config_readmission",
        "Tests/Backup_Recovery/test_runtime_cache_revisit.py::test_monitor_revisits_only_after_foreign_operation_finishes",
        "Tests/Backup_Recovery/test_runtime_producer_settlement.py::test_actual_app_drains_sync_tail_and_idle_mcp_before_storage",
        "Tests/Backup_Recovery/test_runtime_producer_settlement.py::test_mcp_cancelled_or_uncertain_cleanup_keeps_capture_closed",
        "Tests/Backup_Recovery/test_runtime_producer_settlement.py::test_unknown_connection_ownership_keeps_resume_fenced",
    )
    cold = (
        "Tests/Backup_Recovery/test_related_path_admission.py::test_related_paths_retain_native_exclusion_until_close",
        "Tests/Backup_Recovery/test_related_path_admission.py::test_related_paths_refuse_local_pause_during_acquisition",
        "Tests/Backup_Recovery/test_related_path_admission.py::test_related_paths_recheck_pending_recovery_after_native_acquisition",
        "Tests/Backup_Recovery/test_bootstrap_registry_reader.py::test_startup_never_repairs_missing_or_unsafe_registry_lock",
    )
    if system == "Windows":
        return _NATIVE_TESTS, (
            *shared,
            *cold,
            "Tests/Backup_Recovery/test_admission_amortization_native.py::test_windows_cold_acquisition_rederives_and_refuses_changed_current_control",
        )
    native = (
        "Tests/Backup_Recovery/test_admission_closed_gate.py::test_uncancellable_busy_gate_wait_uses_native_blocking_lock",
        "Tests/Backup_Recovery/test_admission_closed_gate.py::test_normal_waits_at_closed_requested_gate_before_scanning_roots",
    )
    mutations = (
        "ancestor-made-group-writable",
        "bootstrap-root-removed",
        "config-selector-edited",
        "control-no-change",
        "data-dir-renamed-and-recreated",
        "data-dir-swapped-for-symlink",
        "enrollment-marker-replaced",
        "pending-record-from-subprocess",
        "profile-record-edited-in-place",
        "registry-intent-file",
        "registry-replaced-from-subprocess",
        "unrelated-pending-from-subprocess",
    )
    warm = tuple(
        "Tests/Backup_Recovery/test_admission_evidence_reuse.py::test_reused_evidence_matches_the_full_derivation"
        f"[{mutation}-{bound}]"
        for mutation in mutations
        for bound in ("bound", "unbound")
        if not (mutation == "profile-record-edited-in-place" and bound == "unbound")
    )
    warm += tuple(
        "Tests/Backup_Recovery/test_admission_evidence_reuse.py::" + name
        for name in (
            "test_warm_admission_reads_current_complete_control_bytes",
            "test_same_inode_bytes_cannot_hide_behind_equal_change_stamps",
            "test_warm_admission_refuses_replaced_native_lock_before_write",
            "test_borrowed_nonempty_lock_bytes_still_require_current_full_read",
            "test_lent_lock_replacement_never_authorizes_detached_description",
            "test_blocked_scope_validation_allows_unrelated_transaction",
            "test_warm_observation_rechecks_selection_and_cancellation_before_io",
            "test_candidate_observation_reserves_last_owner_before_using_pins",
            "test_prederivation_observation_cannot_confirm_a_replacement_hold",
            "test_candidate_selection_cwd_failure_does_not_leak_observation_reservation",
            "test_installed_relative_acquisition_selects_outside_coordinator_mutex",
            "test_concurrent_derivations_keep_confirmed_evidence",
            "test_cold_scope_cannot_continue_a_retired_incumbent",
            "test_counted_borrower_retains_predecessors_until_positive_close",
            "test_uncertain_predecessor_close_retains_native_exclusion",
            "test_temporary_unknown_close_fences_actual_hold",
            "test_content_error_unknown_close_is_not_optional_ineligibility",
            "test_transaction_reuses_only_its_already_locked_file_descriptions",
            "test_independent_borrow_frames_survive_another_borrowers_read_exception",
        )
    )
    return native, (
        *shared,
        *cold,
        *warm,
        "Tests/Backup_Recovery/test_mcp_source_lifetimes.py::test_guarded_parse_fork_cannot_use_inherited_evidence",
    )


_SYNTHETIC_CREDENTIALS = (
    "test-only-new-safety-password",
    "test-only-later-safety-password",
    "qualification-worker-password",
    "alpha-synthetic-history-api-value",
    "beta-synthetic-history-api-value",
    "synthetic-f9-secret",
    "private F9 archive passphrase",
    "test-only",
)
_ALLOWED_ENVIRONMENT = frozenset(
    {
        "CI",
        "COLORTERM",
        "COMSPEC",
        "GITHUB_ACTIONS",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TERM",
        "TZ",
        "WINDIR",
    }
)
_NATIVE_CREDENTIAL_BACKENDS = {
    "Darwin": "keyring.backends.macOS.Keyring",
    "Linux": "keyring.backends.SecretService.Keyring",
    "Windows": "keyring.backends.Windows.WinVaultKeyring",
}
_NATIVE_CREDENTIAL_ENVIRONMENT = (
    "PYTHON_KEYRING_BACKEND",
    "TLDW_NATIVE_CREDENTIAL_ROOT",
    "TLDW_NATIVE_MAC_KEYCHAIN",
    "DBUS_SESSION_BUS_ADDRESS",
    "KEYRING_PROPERTY_PREFERRED_COLLECTION",
    "TLDW_CREDENTIAL_TRANSFER_ROOT",
    "RUNNER_ENVIRONMENT",
    "RUNNER_OS",
)
_LINUX_CREDENTIAL_COLLECTION = "/org/freedesktop/secrets/collection/login"
_NATIVE_SQLITE_OWNERS = frozenset(
    {
        "db.chachanotes.primary",
        "chat.attachments",
        "notes.sync_bindings",
        "notes.file_notes",
        "study.local",
        "quiz.local",
        "recovered.media",
    }
)
_NATIVE_SQLITE_ISSUES = frozenset(
    {
        "cancelled",
        "invalid_domain_reference",
        "invalid_managed_membership",
        "invalid_recovered_asset",
        "invalid_recovered_reference",
        "invalid_recovered_tombstone",
        "invalid_sqlite_integrity",
        "missing_required_asset",
        "recovered_operation_pending",
        "sqlite_resource_limit",
        "sqlite_security_unavailable",
        "sqlite_validation_unavailable",
        "unsupported_domain_reference",
        "unsupported_schema",
        "unsupported_schema_policy",
        "unsupported_schema_version",
        "unsupported_sqlite_owner",
    }
)
_NATIVE_INVENTORY_ISSUES = frozenset(
    {
        "unsupported_owner",
        "invalid_status",
        "unsupported",
        "unavailable",
        "missing_required",
        "duplicate_logical_id",
        "unvalidated_deletion",
        "missing_identity",
        "unsupported_path_kind",
        "undeclared_alias",
        "shared_identity_mismatch",
        "overlapping_owner_roots",
        "dependency_unavailable",
        "config_parse_failure",
        "config_discovery_failure",
        "invalid_shared_declaration",
        "shared_identity_unavailable",
    }
)


def validate_native_credential_environment() -> str:
    """Refuse personal stores, fallback backends and foreign SecretService buses.

    This check performs no credential writes and never creates or unlocks a
    collection. Call it in every fixture-writing process before native access.
    """
    environment = os.environ
    system = platform.system()
    expected = _NATIVE_CREDENTIAL_BACKENDS.get(system)
    if expected is None or environment.get("PYTHON_KEYRING_BACKEND") != expected:
        raise RuntimeError("native_credential_backend_selection_required")
    if system == "Linux":
        root_value = environment.get("TLDW_NATIVE_CREDENTIAL_ROOT", "")
        root = Path(root_value)
        if not root_value or not root.is_absolute() or root.resolve() != root:
            raise RuntimeError("native_credential_private_session_required")
        info = root.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise RuntimeError("native_credential_session_not_private")
        address = environment.get("DBUS_SESSION_BUS_ADDRESS", "")
        if not re.fullmatch(
            re.escape(f"unix:path={root / 'bus'}") + r"(?:,guid=[0-9a-f]{32})?",
            address,
        ):
            raise RuntimeError("native_credential_foreign_session_bus")
        bus = (root / "bus").lstat()
        if not stat.S_ISSOCK(bus.st_mode) or bus.st_uid != os.getuid():
            raise RuntimeError("native_credential_foreign_session_socket")
        if (
            environment.get("KEYRING_PROPERTY_PREFERRED_COLLECTION")
            != _LINUX_CREDENTIAL_COLLECTION
        ):
            raise RuntimeError("native_credential_private_collection_required")
        import secretstorage
        from jeepney.bus_messages import message_bus

        with closing(secretstorage.dbus_init()) as connection:
            # A collection query can auto-activate another daemon before the
            # explicitly unlocked private daemon has finished starting.
            owner = connection.send_and_get_reply(
                message_bus.NameHasOwner("org.freedesktop.secrets")
            )
            if len(owner.body) != 1 or owner.body[0] is not True:
                raise RuntimeError("native_credential_service_not_running")
            collection = secretstorage.Collection(
                connection, _LINUX_CREDENTIAL_COLLECTION
            )
            if collection.is_locked():
                raise RuntimeError("native_credential_session_locked")
    elif not (
        environment.get("GITHUB_ACTIONS") == "true"
        and environment.get("RUNNER_ENVIRONMENT") == "github-hosted"
        and environment.get("RUNNER_OS")
        == {"Darwin": "macOS", "Windows": "Windows"}[system]
    ):
        raise RuntimeError("native_credential_disposable_runner_required")

    import keyring

    module, name = expected.rsplit(".", 1)
    backend = keyring.get_keyring()
    if type(backend) is not getattr(importlib.import_module(module), name):
        raise RuntimeError("native_credential_fallback_backend_refused")
    if backend.priority <= 0:
        raise RuntimeError("native_credential_backend_unavailable")
    if system == "Linux" and (
        getattr(backend, "preferred_collection", None) != _LINUX_CREDENTIAL_COLLECTION
    ):
        raise RuntimeError("native_credential_collection_continuity_lost")
    return expected


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    """Write one deterministic JSON receipt."""
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(
                json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
            )
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _run_git(workspace: Path, *arguments: str) -> str:
    """Run a fixed read-only Git query for the source receipt."""
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("Git is unavailable for source identity capture")
    completed = subprocess.run(  # nosec B603 - fixed executable and arguments
        [executable, *arguments],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    return completed.stdout.rstrip("\r\n")


def _tracked_files(workspace: Path) -> tuple[str, ...]:
    """Return exact tracked path names without Git's display quoting."""
    return tuple(
        relative
        for relative in _run_git(workspace, "ls-files", "-z").split("\0")
        if relative
    )


def _create_private_root(evidence_root: Path) -> Path:
    """Create the test-only private root below a platform-trusted ancestor."""
    if os.name != "nt":
        private_root = evidence_root / "private"
        private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        return private_root

    from tldw_chatbook.Utils.windows_files import WindowsOS

    windows = WindowsOS()
    trusted_temp = Path.home() / "AppData" / "Local" / "Temp"
    private_root = trusted_temp / f"tldw-backup-platform-{os.getpid()}"
    windows.mkdir(private_root, 0o700)
    info = windows.stat(private_root)
    if info.st_uid != windows.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise OSError("windows_private_root_not_private")
    return private_root


def _copy_tracked_source(workspace: Path, private_root: Path) -> tuple[Path, str]:
    """Copy exact tracked HEAD bytes into the private runtime without secrets."""
    if _run_git(workspace, "status", "--porcelain=v1"):
        raise RuntimeError("source_checkout_not_clean")
    archive_path = private_root / "tracked-head.tar"
    source_copy = private_root / "source"
    if os.name == "nt":
        from tldw_chatbook.Utils.windows_files import WindowsOS

        WindowsOS().mkdir(source_copy, 0o700)
    else:
        source_copy.mkdir(mode=0o700)

    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("Git is unavailable for tracked source copy")
    subprocess.run(  # nosec B603 - fixed Git archive of the selected HEAD
        [executable, "archive", "--format=tar", "-o", str(archive_path), "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        timeout=60,
    )
    tracked = set(_tracked_files(workspace))
    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        archived = {member.name.rstrip("/") for member in members if member.isfile()}
        if archived != tracked or any(
            not (member.isfile() or member.isdir()) for member in members
        ):
            raise RuntimeError("tracked_source_archive_mismatch")
        archive.extractall(source_copy, filter="data")
    copied = {
        str(path.relative_to(source_copy)).replace("\\", "/")
        for path in source_copy.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if copied != tracked:
        raise RuntimeError("tracked_source_copy_mismatch")
    return source_copy, _sha256(archive_path)


def _source_receipt(
    workspace: Path, source_copy: Path, archive_sha256: str
) -> dict[str, object]:
    """Identify and hash every tracked file in the private execution copy."""
    tracked = _tracked_files(workspace)
    files = {}
    for relative in tracked:
        candidate = source_copy / relative
        if candidate.is_file() and not candidate.is_symlink():
            files[relative] = _sha256(candidate)
    if len(files) != len(tracked):
        raise RuntimeError("source_receipt_file_count_mismatch")
    package = importlib.metadata.distribution("tldw_chatbook")
    return {
        "schema": 2,
        "git_head": _run_git(workspace, "rev-parse", "HEAD"),
        "git_status": _run_git(workspace, "status", "--porcelain=v1"),
        "distribution_version": package.version,
        "execution_source": "private_tracked_head_copy",
        "source_archive_sha256": archive_sha256,
        "files": files,
    }


def _native_identity(private_root: Path) -> dict[str, object]:
    """Record the production adapter's native identity and operation decisions."""
    receipt: dict[str, object] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "os_name": os.name,
    }
    try:
        from tldw_chatbook.Backup_Recovery import native_files, qualification

        native_root = private_root / "native-identity-root"
        native_root.mkdir(mode=0o700)
        with native_files.pinned_directory(native_root) as descriptor:
            receipt["native_identity"] = qualification.native_identity(descriptor)
        receipt["operations"] = {
            operation: list(qualification.qualified_for(operation, native_root))
            for operation in (
                "publish_new",
                "publish_file",
                "publish_directory",
                "admission",
            )
        }
    except Exception as error:  # noqa: BLE001 - preserve platform failure receipt
        receipt["identity_error"] = f"{type(error).__name__}: {error}"
    return receipt


def _windows_ancestor_receipt(workspace: Path, private_root: Path) -> dict[str, object]:
    """Classify native owner/DACL data without exporting SIDs or local paths."""
    entries: list[dict[str, object]] = []
    receipt: dict[str, object] = {
        "schema": 1,
        "entries": entries,
    }
    if os.name != "nt":
        receipt["unsupported"] = "native_windows_required"
        return receipt

    runner_home = Path.home().resolve(strict=True)
    root_candidates = {
        "workspace": workspace,
        "private_root": private_root,
        "runner_home": runner_home,
        "runner_local_temp": runner_home / "AppData" / "Local" / "Temp",
        "system_drive": Path(os.environ.get("SYSTEMDRIVE", runner_home.drive) + "\\"),
    }
    roots = {
        name: selected.resolve(strict=True)
        for name, selected in root_candidates.items()
        if selected.is_dir()
    }
    receipt["roots"] = list(roots)

    import ctypes as C

    from tldw_chatbook.Utils.windows_files import _P, WindowsOS, _native

    windows, native = WindowsOS(), _native()
    principal_aliases: dict[str, str] = {
        native.user_sid: "CURRENT_USER",
        "S-1-5-18": "LOCAL_SYSTEM",
        "S-1-5-32-544": "BUILTIN_ADMINISTRATORS",
        "S-1-3-4": "OWNER_RIGHTS",
        "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464": (
            "TRUSTED_INSTALLER"
        ),
    }

    def principal_alias(sid: str) -> str:
        if sid not in principal_aliases:
            principal_aliases[sid] = f"UNRECOGNIZED_{len(principal_aliases) - 3}"
        return principal_aliases[sid]

    receipt["current_principal"] = principal_alias(native.user_sid)
    paths: list[Path] = []
    for resolved in roots.values():
        for candidate in reversed((resolved, *resolved.parents)):
            if candidate not in paths:
                paths.append(candidate)

    for path in paths:
        roles = {}
        for name, root in roots.items():
            if path == root:
                roles[name] = "self"
            elif path in root.parents:
                roles[name] = f"ancestor_{root.parents.index(path) + 1}"
        entry: dict[str, object] = {"roles": roles}
        descriptor = None
        file_descriptor = None
        try:
            file_descriptor = windows.open(path, windows.O_RDONLY | windows.O_DIRECTORY)
            handle = native.handle(file_descriptor)
            info = windows.fstat(file_descriptor)
            entry.update(
                projected_mode=stat.S_IMODE(info.st_mode),
                projected_mode_octal=oct(stat.S_IMODE(info.st_mode)),
                projected_uid=info.st_uid,
            )

            owner, dacl, descriptor = _P(), _P(), _P()
            result = native.advapi.GetSecurityInfo(
                handle,
                1,
                5,
                C.byref(owner),
                None,
                C.byref(dacl),
                None,
                C.byref(descriptor),
            )
            if result:
                raise C.WinError(result)
            entry["owner_principal"] = principal_alias(native.sid_string(owner))
            aces = []
            if dacl.value:
                count = struct.unpack_from("<H", C.string_at(dacl, 8), 4)[0]
                for index in range(count):
                    ace = _P()
                    native.check(native.advapi.GetAce(dacl, index, C.byref(ace)))
                    kind, flags, length = struct.unpack("<BBH", C.string_at(ace, 4))
                    if length < 8:
                        raise OSError("malformed_windows_acl")
                    mask = struct.unpack("<I", C.string_at(ace.value + 4, 4))[0]
                    trustee_sid = (
                        native.sid_string(ace.value + 8) if kind in {0, 1} else None
                    )
                    aces.append(
                        {
                            "type": kind,
                            "flags": flags,
                            "mask": mask,
                            "trustee_principal": (
                                principal_alias(trustee_sid) if trustee_sid else None
                            ),
                        }
                    )
            entry["aces"] = aces
        except Exception as error:  # noqa: BLE001 - retain per-ancestor outcome
            entry["error"] = {
                "type": type(error).__name__,
                "errno": getattr(error, "errno", None),
                "winerror": getattr(error, "winerror", None),
            }
        finally:
            if descriptor is not None and descriptor.value:
                native.kernel.LocalFree(descriptor)
            if file_descriptor is not None:
                windows.close(file_descriptor)
        entries.append(entry)
    return receipt


def _private_environment(
    workspace: Path, private_root: Path, *, native_credentials: bool = False
) -> dict[str, str]:
    """Build a credential-free, offline environment rooted below runner temp."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _ALLOWED_ENVIRONMENT
    }
    directories = {
        "HOME": private_root / "home",
        "USERPROFILE": private_root / "home",
        "XDG_CONFIG_HOME": private_root / "xdg-config",
        "XDG_DATA_HOME": private_root / "xdg-data",
        "XDG_CACHE_HOME": private_root / "xdg-cache",
        "XDG_STATE_HOME": private_root / "xdg-state",
        "TEMP": private_root / "tmp",
        "TMP": private_root / "tmp",
        "TMPDIR": private_root / "tmp",
    }
    for directory in set(directories.values()):
        directory.mkdir(parents=True, mode=0o700)
    selector = private_root / "xdg-config" / "config.toml"
    selector.write_text(
        '[general]\nusers_name="windows-qualification"\n'
        "[first_run]\nsetup_completed=true\n"
        "[splash_screen]\nenabled=false\n",
        encoding="utf-8",
    )
    environment.update({key: str(path) for key, path in directories.items()})
    environment.update(
        TLDW_CONFIG_PATH=str(selector),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONIOENCODING="utf-8",
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        PYTHONUTF8="1",
        PYTHONPATH=str(workspace),
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TRANSFORMERS_OFFLINE="1",
    )
    if native_credentials:
        validate_native_credential_environment()
        environment.update(
            {
                name: os.environ[name]
                for name in _NATIVE_CREDENTIAL_ENVIRONMENT
                if name in os.environ
            }
        )
    return environment


def _redact(text: str) -> str:
    """Remove fixed synthetic credentials and password-shaped output fields."""
    for value in _SYNTHETIC_CREDENTIALS:
        text = text.replace(value, "[REDACTED_SYNTHETIC_CREDENTIAL]")
    return re.sub(
        r"(?i)(password(?:_confirm)?\s*[=:]\s*)[^\s,;\]\}]+",
        r"\1[REDACTED]",
        text,
    )


def _sanitize_file(
    source: Path, destination: Path, *, private_root: Path | None = None
) -> None:
    """Copy a UTF-8 evidence file while applying the fixed redaction policy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = _redact(source.read_text(encoding="utf-8", errors="replace"))
    if private_root is not None:
        replacements = (
            (str(private_root), "[PRIVATE_ROOT]"),
            (str(Path.home()), "[RUNNER_HOME]"),
        )
        for path_value, replacement in replacements:
            variants = {
                path_value,
                path_value.replace("\\", "/"),
                path_value.replace("/", "\\"),
            }
            variants.update(value.replace("\\", "\\\\") for value in tuple(variants))
            for value in sorted(variants, key=len, reverse=True):
                content = content.replace(value, replacement)
    destination.write_text(
        content,
        encoding="utf-8",
    )


def _junit_result(path: Path) -> dict[str, object]:
    """Read counts and skip identities from one JUnit document."""
    document = ET.parse(path)
    cases = [
        node for node in document.iter() if node.tag.rsplit("}", 1)[-1] == "testcase"
    ]
    skipped = []
    failed = []
    for case in cases:
        identity = "::".join(
            filter(None, (case.attrib.get("classname"), case.attrib.get("name")))
        )
        child_tags = {child.tag.rsplit("}", 1)[-1] for child in case}
        if "skipped" in child_tags:
            skipped.append(identity)
        if child_tags & {"failure", "error"}:
            failed.append(identity)
    return {"collected": len(cases), "skipped": skipped, "failed": failed}


def _native_failure_metadata(record: Mapping[str, object]) -> dict[str, object]:
    """Project strictly bounded exception/code metadata, never messages or locals."""
    kind, frames = record["error_class"], record["frames"]
    if (
        not isinstance(kind, str)
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", kind)
        or not isinstance(frames, list)
        or len(frames) > 64
    ):
        raise RuntimeError("unsafe_native_failure_metadata")
    projected = []
    for frame in frames:
        filename, function, line = frame["file"], frame["function"], frame["line"]
        if (
            not isinstance(filename, str)
            or not re.fullmatch(
                r"[A-Za-z0-9_.-]{1,125}\.py|<(?:string|stdin)>|<frozen [A-Za-z_][A-Za-z0-9_.]{0,99}>",
                filename,
            )
            or not isinstance(function, str)
            or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]{0,127}|<(?:module|lambda|genexpr|listcomp|dictcomp|setcomp)>",
                function,
            )
            or type(line) is not int
            or not 1 <= line <= 1_000_000
        ):
            raise RuntimeError("unsafe_native_failure_frame")
        projected.append({"file": filename, "function": function, "line": line})
    result = {"error_class": kind, "frames": projected}
    for field, allowed in (("errno", {13}), ("winerror", {5, 32, 33})):
        value = record.get(field)
        if type(value) is int and value in allowed:
            result[field] = value
    target_kind = record.get("target_kind")
    if isinstance(target_kind, str) and target_kind in {
        "declared_shm",
        "declared_wal",
        "declared_main",
        "declared_instance_lock",
        "other",
        "ambiguous",
    }:
        result["target_kind"] = target_kind
    if "issue" in record:
        from tldw_chatbook.Backup_Recovery.recovery_service import issue_code

        issue = record["issue"]
        result["issue"] = (
            issue
            if isinstance(issue, str)
            and issue
            in {
                "cancelled",
                "review_required",
                "compression_review_required",
                "encryption_unavailable",
                "encryption_failed",
            }
            else issue_code(ValueError(issue))
        )
    for field, allowed in (
        ("sqlite_owner", _NATIVE_SQLITE_OWNERS),
        ("sqlite_issue", _NATIVE_SQLITE_ISSUES),
    ):
        value = record.get(field)
        if isinstance(value, str) and value in allowed:
            result[field] = value
    inventory = record.get("inventory")
    if isinstance(inventory, Mapping):
        from tldw_chatbook.Backup_Recovery.data_groups import group_for_owner
        from tldw_chatbook.Backup_Recovery.inventory import BLOCKING

        issues = inventory.get("issues", ())
        issues = (
            {
                value
                for value in issues
                if isinstance(value, str) and value in _NATIVE_INVENTORY_ISSUES
            }
            if isinstance(issues, (list, tuple))
            else set()
        )
        blocking = set()
        rows = inventory.get("blocking", ())
        for row in rows if isinstance(rows, (list, tuple)) else ():
            if not isinstance(row, Mapping):
                continue
            owner, status = row.get("owner"), row.get("status")
            if (
                isinstance(owner, str)
                and isinstance(status, str)
                and (
                    group_for_owner(owner) is not None
                    or owner in {"unknown", "sqlite.transient"}
                )
                and status in BLOCKING
            ):
                blocking.add((owner, status))
        result["inventory"] = {
            "issues": sorted(issues)[:32],
            "blocking": [
                {"owner": owner, "status": status}
                for owner, status in sorted(blocking)[:64]
            ],
        }
    return result


def _record_native_failure(
    root: Path, error: BaseException, *, metadata: Mapping[str, object] | None = None
) -> None:
    """Record metadata without allowing an observation error to mask the failure."""
    try:
        from Tests.Backup_Recovery.thread_diagnostics import _error_metadata
        from tldw_chatbook.Backup_Recovery.recovery_service import issue_code

        optional = dict(metadata or {})
        optional.pop("target_kind", None)
        try:
            from tldw_chatbook.Backup_Recovery import archive_reader, restore_plan
            from tldw_chatbook.Backup_Recovery.capture import _item_validator
            from tldw_chatbook.Backup_Recovery.config_adapter import _InstanceLock
            from tldw_chatbook.Backup_Recovery.models import Inventory
            from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
            from tldw_chatbook.Backup_Recovery.publication import _sidecar_main

            trace = error.__traceback__ if isinstance(error, OSError) else None
            for _ in range(64):
                if trace is None:
                    break
                observed = trace.tb_next
                hashed = observed.tb_next if observed is not None else None
                if (
                    trace.tb_frame.f_code is restore_plan._fingerprint.__code__
                    and observed is not None
                    and observed.tb_frame.f_code is restore_plan._observed.__code__
                    and hashed is not None
                    and hashed.tb_frame.f_code is archive_reader._hash.__code__
                    and hashed.tb_next is None
                ):
                    path = trace.tb_frame.f_locals.get("path")
                    target = trace.tb_frame.f_locals.get("target")
                    if (
                        isinstance(path, Path)
                        and path == observed.tb_frame.f_locals.get("path")
                        and path == hashed.tb_frame.f_locals.get("path")
                        and type(target) is Inventory
                    ):
                        owners = {row.owner_id: row for row in install_adapters()}
                        matches = [row for row in target.items if row.path == path]
                        kinds, mains = set(), []
                        for item in matches:
                            if (
                                sum(
                                    row.logical_id == item.logical_id
                                    for row in target.items
                                )
                                != 1
                            ):
                                kinds.add("ambiguous")
                                continue
                            if item.owner == "sqlite.transient":
                                candidates = [
                                    row
                                    for row in target.items
                                    if item.dependencies == (row.logical_id,)
                                ]
                                try:
                                    main = _sidecar_main(item, target.items, owners)
                                except ValueError:
                                    kinds.add("ambiguous")
                                    continue
                                if len(candidates) != 1 or main.status != "included":
                                    kinds.add("ambiguous")
                                    continue
                                kinds.add(
                                    "declared_shm"
                                    if path == Path(str(main.path) + "-shm")
                                    else "declared_wal"
                                )
                            elif item.owner == "runtime.instance_lock":
                                parts = item.logical_id.split(":")
                                config_id = (
                                    item.logical_id.removesuffix(item.owner) + "config"
                                )
                                configs = [
                                    row
                                    for row in target.items
                                    if row.logical_id == config_id
                                ]
                                meta = item.metadata
                                kinds.add(
                                    "declared_instance_lock"
                                    if type(owners.get(item.owner)) is _InstanceLock
                                    and len(parts) == 3
                                    and parts[0] == "profile"
                                    and parts[1]
                                    and parts[2] == item.owner
                                    and path.name == ".instance.lock"
                                    and item.status == "intentionally_excluded"
                                    and item.dependencies == (config_id,)
                                    and len(configs) == 1
                                    and configs[0].owner == "config"
                                    and configs[0].status == "included"
                                    and configs[0].path is not None
                                    and meta is not None
                                    and (
                                        meta.root_id,
                                        meta.parent_id,
                                        meta.relative_path,
                                        meta.kind,
                                        meta.policy,
                                    )
                                    == (item.logical_id, None, "", "file", "private")
                                    else "ambiguous"
                                )
                                continue
                            else:
                                adapter = owners.get(item.owner)
                                policy = (
                                    _item_validator(adapter, item).schema_policy()
                                    if adapter
                                    else None
                                )
                                if (
                                    item.status != "included"
                                    or policy is None
                                    or not policy.schema_sql
                                ):
                                    kinds.add("other")
                                    continue
                                main = item
                                kinds.add("declared_main")
                            mains.append(main)
                        if len(mains) > 1 and not all(
                            row.path == mains[0].path
                            and row.shared_group
                            and row.shared_group == mains[0].shared_group
                            for row in mains
                        ):
                            kinds.add("ambiguous")
                        optional["target_kind"] = (
                            next(iter(kinds))
                            if len(kinds) == 1
                            else "ambiguous"
                            if kinds
                            else "other"
                        )
                    break
                trace = trace.tb_next
        except Exception:  # noqa: BLE001 - optional classification cannot hide the failure.
            optional.pop("target_kind", None)
        record = _native_failure_metadata(
            {**optional, **_error_metadata(error), "issue": issue_code(error)}
        )
        _write_json(root / f"{os.getpid()}-{uuid4().hex}.json", record)
    except Exception:  # noqa: BLE001 - preserve the original private failure.
        return


def install_native_failure_hook() -> None:
    """Observe uncaught child failures before the product script starts."""
    root = Path(os.environ["TLDW_NATIVE_FAILURE_ROOT"])
    previous = sys.excepthook

    def observe(kind, error, trace):
        _record_native_failure(root, error)
        previous(kind, error, trace)

    sys.excepthook = observe


class NativeCredentialFailures:
    """Record outer pytest failures that never reach sys.excepthook."""

    def pytest_exception_interact(self, node, call, report):
        if call.excinfo is not None:
            _record_native_failure(
                Path(os.environ["TLDW_NATIVE_FAILURE_ROOT"]), call.excinfo.value
            )


def _publish_native_failures(private_root: Path, artifacts: Path) -> None:
    """Revalidate private observations and publish only class and code locations."""
    failures = []
    for path in sorted((private_root / "native-failures").glob("*.json")):
        if path.is_symlink():
            raise RuntimeError("unsafe_native_failure_receipt")
        failures.append(
            _native_failure_metadata(json.loads(path.read_text(encoding="utf-8")))
        )
    children = []
    for path in sorted(
        (private_root / "native-failures" / "child-stacks").glob("*.json")
    ):
        label = re.fullmatch(
            r"(setup|capture|transfer|rollback|negative|read|read-rollback)--(default|retargeted)--[1-9][0-9]{0,9}",
            path.stem,
        )
        if path.is_symlink() or label is None:
            raise RuntimeError("unsafe_native_child_diagnostic")
        snapshots = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(snapshots, list):
            raise TypeError("unsafe_native_child_samples")
        samples = []
        for snapshot in snapshots[-4:]:
            if not isinstance(snapshot, list) or len(snapshot) > 32:
                raise RuntimeError("unsafe_native_child_threads")
            samples.append(
                [
                    {
                        "frames": _native_failure_metadata(
                            {
                                "error_class": "ThreadSnapshot",
                                "frames": thread["frames"],
                            }
                        )["frames"]
                    }
                    for thread in snapshot
                ]
            )
        children.append({"route": label[1], "role": label[2], "samples": samples})
    if failures or children:
        _write_json(
            artifacts / "native-failures.json",
            {
                "schema": 1,
                "failures": failures,
                **({"children": children} if children else {}),
            },
        )


def _run_pytest_phase(
    *,
    workspace: Path,
    private_root: Path,
    artifacts: Path,
    environment: dict[str, str],
    phase: str,
    tests: tuple[str, ...],
    noconftest: bool,
    timeout_seconds: int,
    native_credentials: bool = False,
) -> dict[str, object]:
    """Run one fixed pytest phase and retain its sanitized log and JUnit receipt."""
    prefix = "native-" if phase == "native" else ""
    raw_log = private_root / f"{prefix}pytest-output.log"
    raw_junit = private_root / f"{prefix}pytest.xml"
    bootstrap = (
        "from Tests.network_guard import install; install(); "
        "import keyring; from keyring.backends.null import Keyring; "
        "keyring.set_keyring(Keyring()); import pytest, sys; "
        "raise SystemExit(pytest.main(sys.argv[1:]))"
    )
    if native_credentials:
        failure_root = private_root / "native-failures"
        failure_root.mkdir(mode=0o700, exist_ok=True)
        environment = {**environment, "TLDW_NATIVE_FAILURE_ROOT": str(failure_root)}
        (workspace / "sitecustomize.py").write_text(
            "from Tests.Backup_Recovery.run_platform_product import install_native_failure_hook\n"
            "install_native_failure_hook()\n",
            encoding="utf-8",
        )
        bootstrap = (
            "from Tests.network_guard import install; install(); "
            "from Tests.Backup_Recovery.run_platform_product import "
            "validate_native_credential_environment, NativeCredentialFailures; "
            "validate_native_credential_environment(); import pytest, sys; "
            "raise SystemExit(pytest.main(sys.argv[1:], plugins=[NativeCredentialFailures()]))"
        )
    command = [sys.executable, "-c", bootstrap]
    if noconftest:
        command.append("--noconftest")
    if environment.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1":
        command.extend(("-p", "pytest_asyncio.plugin", "-p", "pytest_timeout"))
    command.extend(
        (
            *tests,
            "-vv",
            "--tb=no" if native_credentials else "--tb=long",
            *(("--show-capture=no",) if native_credentials else ()),
            f"--timeout={timeout_seconds if native_credentials else 2400}",
            f"--basetemp={private_root / f'{phase}-pytest'}",
            f"--junitxml={raw_junit}",
        )
    )
    with raw_log.open("w", encoding="utf-8") as output:
        try:
            completed = subprocess.run(  # nosec B603 - fixed interpreter/tests
                command,
                cwd=workspace,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
            pytest_returncode = completed.returncode
        except subprocess.TimeoutExpired:
            output.write(f"\n{phase.upper()} PYTEST PHASE TIMED OUT\n")
            pytest_returncode = 124

    if native_credentials:
        _publish_native_failures(private_root, artifacts)
    if not native_credentials:
        _sanitize_file(raw_log, artifacts / raw_log.name, private_root=private_root)
    junit = {"collected": 0, "skipped": [], "failed": [], "parse_error": None}
    if raw_junit.is_file():
        if not native_credentials:
            _sanitize_file(
                raw_junit, artifacts / raw_junit.name, private_root=private_root
            )
        try:
            junit.update(_junit_result(raw_junit))
        except (OSError, ET.ParseError) as error:
            junit["parse_error"] = f"{type(error).__name__}: {error}"
    else:
        junit["parse_error"] = "pytest did not produce JUnit XML"
    return {
        "tests": list(tests),
        "pytest_returncode": pytest_returncode,
        "junit": junit,
    }


def _native_runtime_receipt(source: Mapping[str, object]) -> dict[str, object]:
    """Validate and project only public native runtime and artifact identity."""
    fields = (
        "schema",
        "status",
        "system",
        "release",
        "machine",
        "python",
        "backend",
        "revision",
        "wheel_sha256",
    )
    receipt = {name: source[name] for name in fields}
    if (
        type(receipt["schema"]) is not int
        or receipt["schema"] != 1
        or receipt["status"] != "passed"
        or receipt["system"] not in _NATIVE_CREDENTIAL_BACKENDS
        or receipt["backend"] != _NATIVE_CREDENTIAL_BACKENDS[receipt["system"]]
    ):
        raise RuntimeError("invalid_native_credential_runtime_receipt")
    for name in ("release", "machine", "python"):
        if not isinstance(receipt[name], str) or not re.fullmatch(
            r"[A-Za-z0-9_.+()-]{1,128}", receipt[name]
        ):
            raise RuntimeError("invalid_native_credential_runtime_field")
    for name, length in (("revision", 40), ("wheel_sha256", 64)):
        if not isinstance(receipt[name], str) or not re.fullmatch(
            rf"[0-9a-f]{{{length}}}", receipt[name]
        ):
            raise RuntimeError("invalid_native_credential_identity_hash")
    return receipt


def _publish_native_credential_artifacts(
    transfer_root: Path, artifacts: Path, product_selection: str
) -> int:
    """Publish encrypted synthetic transfers and strict value-free receipts only."""
    outbound = transfer_root / "outbound"
    if product_selection == "native-credentials-source":
        pending = []
        for system in _NATIVE_CREDENTIAL_BACKENDS:
            basename = f"source-{system.lower()}"
            path = outbound / f"{basename}.json"
            if not path.is_file():
                continue
            archive = outbound / f"{basename}.age"
            if path.is_symlink() or archive.is_symlink() or not archive.is_file():
                raise RuntimeError("unsafe_native_credential_transfer")
            source = json.loads(path.read_text(encoding="utf-8"))
            receipt = _native_runtime_receipt(source)
            if receipt["system"] != system or source["archive"] != archive.name:
                raise RuntimeError("native_credential_transfer_identity_mismatch")
            digest = _sha256(archive)
            with archive.open("rb") as stream:
                encrypted = stream.read(22) == b"age-encryption.org/v1\n"
            if not encrypted or source["archive_sha256"] != digest:
                raise RuntimeError("native_credential_transfer_not_verified_encrypted")
            receipt.update(archive=archive.name, archive_sha256=digest)
            pending.append((archive, path, receipt))
        for archive, path, receipt in pending:
            shutil.copyfile(archive, artifacts / archive.name)
            _write_json(artifacts / path.name, receipt)
        return len(pending)

    path = outbound / "destination-results.json"
    if not path.is_file():
        return 0
    if path.is_symlink():
        raise RuntimeError("unsafe_native_credential_destination_receipt")
    source = json.loads(path.read_text(encoding="utf-8"))
    receipt = _native_runtime_receipt(source)
    results = []
    for result in source["results"]:
        source_system = result["source_system"]
        if (
            source_system not in _NATIVE_CREDENTIAL_BACKENDS
            or result["destination_system"] != receipt["system"]
            or not isinstance(result["archive_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", result["archive_sha256"])
        ):
            raise RuntimeError("invalid_native_credential_direction_receipt")
        projected = {
            name: result[name]
            for name in ("source_system", "destination_system", "archive_sha256")
        }
        for name in ("isolated", "original_retained", "replacement", "rollback"):
            if result[name] is not True:
                raise RuntimeError("invalid_native_credential_equality_receipt")
            projected[name] = result[name]
        for name in ("captured", "manual_required", "unavailable"):
            if type(result[name]) is not int or result[name] < 0:
                raise RuntimeError("invalid_native_credential_count_receipt")
            projected[name] = result[name]
        if result["captured"] == 0 or result["unavailable"] != 0:
            raise RuntimeError("native_credential_capture_not_complete")
        results.append(projected)
    if (
        len(results) != 3
        or {row["source_system"] for row in results} != set(_NATIVE_CREDENTIAL_BACKENDS)
        or type(source["negative_checks"]) is not int
        or source["negative_checks"] < 0
    ):
        raise RuntimeError("incomplete_native_credential_destination_receipt")
    receipt.update(results=results, negative_checks=source["negative_checks"])
    _write_json(artifacts / path.name, receipt)
    return len(results)


def _installed_receipts(private_root: Path) -> list[dict[str, object]]:
    """Hash every file in each wheel installation built by product tests."""
    results = []
    resolved_private = private_root.resolve()
    for source_receipt in sorted(private_root.rglob("native-package.json")):
        source = json.loads(source_receipt.read_text(encoding="utf-8"))
        installed = Path(source["installed"]).resolve()
        if not installed.is_relative_to(resolved_private):
            raise RuntimeError("installed package escaped the private evidence root")
        files = {
            str(path.relative_to(installed)).replace("\\", "/"): _sha256(path)
            for path in sorted(installed.rglob("*"))
            if path.is_file()
            and not path.is_symlink()
            and "__pycache__" not in path.parts
        }
        results.append(
            {
                "fixture_receipt": str(
                    source_receipt.relative_to(private_root)
                ).replace("\\", "/"),
                "wheel_sha256": source["sha256"],
                "installed_files": files,
            }
        )
    return results


def _collect_safe_logs(private_root: Path, artifacts: Path) -> int:
    """Collect only text logs; configs, archives and fixture payloads stay private."""
    count = 0
    for phase in ("native-pytest", "product-pytest"):
        phase_root = private_root / phase
        for source in sorted(phase_root.rglob("*.log")):
            if source.is_symlink() or not source.is_file():
                continue
            relative = source.relative_to(phase_root)
            visible_relative = Path(
                *(
                    f"_dot_{part[1:]}" if part.startswith(".") else part
                    for part in relative.parts
                )
            )
            destination = artifacts / "test-logs" / phase / visible_relative
            if destination.exists():
                raise RuntimeError("safe_log_artifact_name_collision")
            _sanitize_file(
                source,
                destination,
                private_root=private_root,
            )
            count += 1
    return count


def _artifact_hashes(artifacts: Path) -> dict[str, str]:
    """Hash final artifact files without including the hash receipt itself."""
    return {
        str(path.relative_to(artifacts)).replace("\\", "/"): _sha256(path)
        for path in sorted(artifacts.rglob("*"))
        if path.is_file() and path.name != "artifact-sha256.json"
    }


def run(
    workspace: Path, evidence_root: Path, *, product_selection: str = "full"
) -> int:
    """Execute the finite qualification and retain safe failure evidence."""
    product_tests = _PRODUCT_SELECTIONS[product_selection]
    native_tests = _NATIVE_TESTS
    if product_selection == "admission-amortization":
        native_tests, product_tests = admission_selection(platform.system())
    native_credentials = product_selection.startswith("native-credentials-")
    if native_credentials:
        if not os.environ.get("TLDW_CREDENTIAL_TRANSFER_ROOT"):
            raise RuntimeError("native_credential_transfer_root_required")
        validate_native_credential_environment()
    workspace = workspace.resolve()
    evidence_root = evidence_root.resolve()
    artifacts = evidence_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_root = _create_private_root(evidence_root)
    source_copy, archive_sha256 = _copy_tracked_source(workspace, private_root)

    if native_credentials:
        environment = _private_environment(
            source_copy, private_root, native_credentials=True
        )
        environment["TLDW_NATIVE_CREDENTIAL_REVISION"] = _run_git(
            workspace, "rev-parse", "HEAD"
        )
        transfer_root = Path(environment["TLDW_CREDENTIAL_TRANSFER_ROOT"])
        phase = _run_pytest_phase(
            workspace=source_copy,
            private_root=private_root,
            artifacts=artifacts,
            environment=environment,
            phase="product",
            tests=product_tests,
            noconftest=True,
            timeout_seconds=(
                180 if product_selection == "native-credentials-destination" else 80
            )
            * 60,
            native_credentials=True,
        )
        installed = _installed_receipts(private_root)
        junit = phase["junit"]
        failed = bool(
            phase["pytest_returncode"]
            or junit["parse_error"]
            or junit["skipped"]
            or junit["failed"]
            or junit["collected"] < len(product_tests)
            or not installed
        )
        published = 0
        if not failed:
            published = _publish_native_credential_artifacts(
                transfer_root, artifacts, product_selection
            )
            failed = published != (
                1 if product_selection == "native-credentials-source" else 3
            )
        summary = {
            "schema": 1,
            "product_selection": product_selection,
            "status": "failed" if failed else "passed",
            "revision": environment["TLDW_NATIVE_CREDENTIAL_REVISION"],
            "source_archive_sha256": archive_sha256,
            "backend": environment["PYTHON_KEYRING_BACKEND"],
            "pytest_returncode": phase["pytest_returncode"],
            "collected": junit["collected"],
            "failed": len(junit["failed"]),
            "skipped": len(junit["skipped"]),
            "junit_available": junit["parse_error"] is None,
            "installed_package_receipts": len(installed),
            "published_directions": published,
        }
        _write_json(artifacts / "summary.json", summary)
        _write_json(artifacts / "artifact-sha256.json", _artifact_hashes(artifacts))
        return 1 if failed else 0

    _write_json(
        artifacts / "source-receipt.json",
        _source_receipt(workspace, source_copy, archive_sha256),
    )
    try:
        ancestor_receipt = _windows_ancestor_receipt(workspace, private_root)
    except Exception as error:  # noqa: BLE001 - preserve diagnostic failure
        ancestor_receipt = {
            "schema": 1,
            "error": {
                "type": type(error).__name__,
                "errno": getattr(error, "errno", None),
                "winerror": getattr(error, "winerror", None),
            },
        }
    _write_json(artifacts / "windows-ancestor-security.json", ancestor_receipt)
    _write_json(artifacts / "native-identity.json", _native_identity(private_root))
    environment = _private_environment(source_copy, private_root)
    native_phase = _run_pytest_phase(
        workspace=source_copy,
        private_root=private_root,
        artifacts=artifacts,
        environment=environment,
        phase="native",
        tests=native_tests,
        noconftest=True,
        timeout_seconds=10 * 60,
    )
    product_phase = _run_pytest_phase(
        workspace=source_copy,
        private_root=private_root,
        artifacts=artifacts,
        environment=environment,
        phase="product",
        tests=product_tests,
        noconftest=False,
        timeout_seconds=(120 if product_selection == "support" else 80) * 60,
    )

    installed = _installed_receipts(private_root)
    _write_json(artifacts / "installed-package-receipt.json", installed)
    log_count = _collect_safe_logs(private_root, artifacts)
    phases = {"native": native_phase, "product": product_phase}
    effective_failure = not installed
    for phase in phases.values():
        phase_junit = phase["junit"]
        phase_tests = phase["tests"]
        if not isinstance(phase_junit, dict) or not isinstance(phase_tests, list):
            raise TypeError("invalid_internal_phase_receipt")
        effective_failure |= bool(
            phase["pytest_returncode"]
            or phase_junit["parse_error"]
            or phase_junit["skipped"]
            or phase_junit["failed"]
            or phase_junit["collected"] < len(phase_tests)
        )
    summary = {
        "schema": 2,
        "product_selection": product_selection,
        "tests": [*native_tests, *product_tests],
        "phases": phases,
        "effective_returncode": 1 if effective_failure else 0,
        "installed_package_receipts": len(installed),
        "collected_test_logs": log_count,
    }
    _write_json(artifacts / "summary.json", summary)
    _write_json(artifacts / "artifact-sha256.json", _artifact_hashes(artifacts))
    return summary["effective_returncode"]


def main(arguments: Iterable[str] | None = None) -> int:
    """Parse command-line arguments and run the native qualification."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--product-selection",
        choices=tuple(_PRODUCT_SELECTIONS),
        default="full",
    )
    options = parser.parse_args(arguments)
    return run(
        options.workspace,
        options.evidence_root,
        product_selection=options.product_selection,
    )


if __name__ == "__main__":
    raise SystemExit(main())
